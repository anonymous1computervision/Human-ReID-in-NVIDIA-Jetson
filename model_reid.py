import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from glob import glob
from random import shuffle
from utils.reid_dataset import ReIDDataset
from utils import misc
import numpy as np
import os
import pprint


class ReidNetwork():
    def __init__(self, sess, embedding_dim, train, height, width, **kwargs):
        self.sess = sess
        self.embedding_dim = embedding_dim
        self.height = height
        self.width = width
        if train:
            self.data_dir = kwargs.get('data_dir')
            assert self.data_dir is not None, 'pass in data_dir to train'
            self.margin = kwargs.get('margin')
            self.threshold = kwargs.get('threshold')
            self.write_images = kwargs.get('write_images', False)
            self.is_train = tf.placeholder(dtype=tf.bool)
            self.make_training_graph()
        else:
            #much simpler graph with just testing, no triplets to worry about
            self.test_model_dir = kwargs.get('test_model_dir')
            self.in_tensor = kwargs.get('in_tensor', None)
            self.make_test_graph()

    def training_network(self, image):
        _, endpoints = resnet_v1.resnet_v1_50(
            image,
            num_classes=702,
            is_training=True,
            global_pool=True,
            output_stride=None,
            reuse=tf.AUTO_REUSE,
            scope='resnet_v1_50')
        return endpoints

    def testing_network(self, image):
        _, endpoints = resnet_v1.resnet_v1_50(
            image,
            num_classes=702,
            is_training=False,
            global_pool=True,
            output_stride=None,
            reuse=tf.AUTO_REUSE,
            scope='resnet_v1_50')
        return endpoints

    def get_embedding(self, image, is_eval):
        if not is_eval:
            endpoints = tf.cond(self.is_train,
                                lambda: self.training_network(image),
                                lambda: self.testing_network(image))
        else:
            print('using no cond')
            endpoints = self.testing_network(image)

        end_convs = tf.identity(endpoints['resnet_v1_50/block4'])
        pooled = tf.reduce_max(end_convs, axis=1)
        pooled = tf.reduce_max(pooled, axis=1)
        pooled = tf.reshape(pooled, [-1, 2048])

        with tf.variable_scope('resnet_v1_50/linear', reuse=tf.AUTO_REUSE):
            self.matrix = tf.get_variable('matrix', [2048, self.embedding_dim],
                                          tf.float32)
            bias = tf.get_variable('bias', [self.embedding_dim])
            linear_out = tf.matmul(pooled, self.matrix) + bias

        embedding = tf.nn.l2_normalize(linear_out, axis=1, name='embedding')
        return embedding

    def make_test_graph(self):
        if self.in_tensor is None:
            self.image = tf.placeholder(
                tf.float32, [None, self.height, self.width, 3], 'input')
        else:
            self.image = tf.identity(self.in_tensor, 'input')

        self.embedding = self.get_embedding(self.image, True)
        self.saver = tf.train.Saver(tf.trainable_variables())
        self.saver.restore(self.sess, self.test_model_dir)

        print('Restored test network from {}'.format(self.test_model_dir))

    def get_out_tensor(self):
        return self.embedding

    def make_training_graph(self):
        self.query_placeholder = tf.placeholder(
            tf.float32, [None, self.height, self.width, 3])
        self.p_placeholder = tf.placeholder(tf.float32,
                                            [None, self.height, self.width, 3])
        self.n_placeholder = tf.placeholder(tf.float32,
                                            [None, self.height, self.width, 3])

        self.q = self.get_embedding(self.query_placeholder, False)
        self.p = self.get_embedding(self.p_placeholder, False)
        self.n = self.get_embedding(self.n_placeholder, False)

        qTdiff = tf.diag_part(tf.matmul(self.q, tf.transpose(self.n)))
        qTsim = tf.diag_part(tf.matmul(self.q, tf.transpose(self.p)))
        self.diff_expanded = qTsim - qTdiff
        self.loss = tf.reduce_mean(
            tf.maximum(0.0, self.margin - self.diff_expanded))
        self.loss_sum = tf.summary.scalar('Loss', self.loss)

        self.diff = tf.reduce_mean(self.diff_expanded)
        self.accuracy = tf.reduce_mean(
            tf.to_float(tf.greater(self.diff_expanded, 0)))

        tf.summary.histogram('linear layer', self.matrix)

        if self.threshold:
            self.thresh_loss = tf.reduce_mean(
                tf.maximum(0.0, self.threshold - qTsim))
            self.loss_with_thresh = self.loss + self.thresh_loss
            self.threshold_sum = tf.summary.scalar('Thresh_loss',
                                                   self.thresh_loss)
            self.lwt_sum = tf.summary.scalar('Loss with thresh',
                                             self.loss_with_thresh)
        if self.write_images:
            tf.summary.image('Query', self.query_placeholder)
            tf.summary.image('Positive', self.p_placeholder)
            tf.summary.image('Negative', self.n_placeholder)

        self.summary = tf.summary.merge_all()

    def train(self, config):
        self.dataset = ReIDDataset(
            data_dir=self.data_dir,
            N=config.N,
            k=config.k,
            batch_size=config.batch_size,
            sample_num=config.sample_num,
            is_train_pholder=self.is_train,
            in_pholder=self.query_placeholder,
            emb_op=self.q,
            sess=self.sess)
        opt = tf.train.AdamOptimizer(
            learning_rate=config.learning_rate, beta1=config.momentum)
        # This is a different saver than the test saver because different variables
        # have to be restored.
        self.saver = tf.train.Saver(max_to_keep=25)
        tvs = tf.trainable_variables()
        tvs = [var for var in tvs if 'logits' not in var.name]

        if not self.threshold:
            train_step = opt.minimize(self.loss, var_list=tvs)
        else:
            train_step = opt.minimize(self.loss_with_thresh, var_list=tvs)

        if config.restore:
            model_dir = 'models/reid/m{}'.format(config.id)
            self.sess.run(tf.global_variables_initializer())
            path = tf.train.latest_checkpoint(model_dir)
            print(path)
            start_epoch = 1 + int(path.split('.ckpt')[1].split('-')[1])
            print('restoring model in {} with epoch {}'.format(
                model_dir, start_epoch))
            self.saver.restore(self.sess, path)
        else:
            start_epoch = 0
            try:
                config.id = 1 + max([
                    int(f.split('/')[-1][1:]) for f in glob('./models/reid/m*')
                ])
            except ValueError:
                print(
                    'There is nothing in models/reid/ or there is a malformatted file, using id 0.'
                )
                config.id = 0
            print("This model's id is {}.".format(config.id))
            model_dir = 'models/reid/m{}'.format(config.id)
            os.mkdir(model_dir)
            config_dict = misc.flags_to_dict(config)
            with open(os.path.join(model_dir, 'config.txt'), 'w') as f:
                pprint.pprint(config_dict, f)
            print('Attempting to restore ID model')

            def is_restorable(var):
                name = var.name
                if ('block' in name or 'conv1' in var.name):
                    return True
                return False

            to_restore = [var for var in tvs if is_restorable(var)]
            init_saver = tf.train.Saver(to_restore)
            assert config.init_model is not None, 'trying to restore reid model without pretrained id model. pass in path to init model.'
            global_vars = tf.global_variables()
            to_init_vars = [
                var for var in global_vars if var not in to_restore
            ]
            tf.variables_initializer(to_init_vars).run()

            init_saver.restore(self.sess, config.init_model)
            print('Restored ID model')
        model_path = os.path.join(model_dir, 'reid_model.ckpt')

        self.log_dir = os.path.join(model_dir, 'logs')
        val_dir = os.path.join(self.log_dir, 'val')
        train_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        print(self.log_dir)
        val_writer = tf.summary.FileWriter(val_dir, self.sess.graph)

        self.dataset.get_data_partition()
        val_triplets = self.dataset.get_random_triplets(
            self.dataset.val_x, self.dataset.val_y)
        ##Train Loop
        for epoch in range(start_epoch, config.num_epochs):
            p = config.p_0 + (epoch / config.num_epochs) * (
                config.p_max - config.p_0)
            print('Occlusion proportion is {:.3f}'.format(p))
            batch_triplets = self.dataset.hard_triplet_mine(p)
            for idx in range(config.k):
                triplets = np.array(
                    batch_triplets[idx * config.batch_size:(idx + 1) *
                                   config.batch_size])
                query = np.squeeze(triplets[:, 0, ...])
                positive = np.squeeze(triplets[:, 1, ...])
                negative = np.squeeze(triplets[:, 2, ...])
                loss, acc, diff, _, summary = self.sess.run(
                    [
                        self.loss, self.accuracy, self.diff, train_step,
                        self.summary
                    ],
                    feed_dict={
                        self.query_placeholder: query,
                        self.p_placeholder: positive,
                        self.n_placeholder: negative,
                        self.is_train: True
                    })
                train_writer.add_summary(summary, idx + epoch * config.k)
                if not idx % (config.k // 4):
                    print(
                        'epoch {0:d} idx {1:d} loss {2:.5f} acc {3:.5f} diff {4:.5f}'.
                        format(epoch, idx, loss, acc, diff))
            len_val = len(val_triplets) // 2
            shuffle(val_triplets)
            triplets = np.array(val_triplets[:len_val])
            query = np.squeeze(triplets[:, 0, ...])
            positive = np.squeeze(triplets[:, 1, ...])
            negative = np.squeeze(triplets[:, 2, ...])
            loss, acc, diff, summary = self.sess.run(
                [self.loss, self.accuracy, self.diff, self.summary],
                feed_dict={
                    self.query_placeholder: query,
                    self.p_placeholder: positive,
                    self.n_placeholder: negative,
                    self.is_train: False
                })
            val_writer.add_summary(summary, epoch * config.k)

            print('VAL epoch {} loss {} acc {} diff {} '.format(
                epoch, loss, acc, diff))
            if epoch % config.save_itr == 0 or epoch == config.num_epochs - 1:
                self.saver.save(
                    self.sess,
                    model_path,
                    global_step=epoch,
                    write_meta_graph=not bool(epoch))
        print('Done Training')
