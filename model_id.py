import tensorflow as tf
import numpy as np
import os
import random
import pprint
from glob import glob
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from utils.classify_dataset import ClassifyDataset


class ID_ResNet():
    """Class for re-training and inferring a resnet 50 V1"""

    def __init__(self, sess, data_dir, n_labels, h, w, batch_size, crop_prop,
                 write_images):
        self.sess = sess
        self.data_dir = data_dir
        self.n_labels = n_labels
        self.dataset = ClassifyDataset(
            self.data_dir,
            h=h,
            w=w,
            batch_size=batch_size,
            crop_prop=crop_prop)
        self.write_images = write_images
        self.inputs, self.labels = self.dataset.next_element
        self.labels = tf.reshape(self.labels, [-1, self.n_labels])
        self.is_train = tf.placeholder(dtype=tf.bool)
        self.make_graph()

    def network_train(self, inputs):
        net, end_points = resnet_v1.resnet_v1_50(
            inputs,
            num_classes=self.n_labels,
            is_training=True,
            global_pool=True,
            output_stride=None,
            reuse=tf.AUTO_REUSE,
            scope="resnet_v1_50")
        return net, end_points

    def network_test(self, inputs):
        net, end_points = resnet_v1.resnet_v1_50(
            inputs,
            num_classes=self.n_labels,
            is_training=False,
            global_pool=True,
            output_stride=None,
            reuse=tf.AUTO_REUSE,
            scope="resnet_v1_50")
        return net, end_points

    def make_graph(self):
        self.train_net, _ = self.network_train(self.inputs)
        self.test_net, _ = self.network_test(self.inputs)
        y_hat = tf.cond(self.is_train, lambda: self.train_net,
                        lambda: self.test_net)
        y_hat = tf.reshape(y_hat, [-1, 702])
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=y_hat, labels=self.labels))
        correct_prediction = tf.equal(
            tf.argmax(y_hat, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        loss_sum = tf.summary.scalar('loss', self.loss)
        acc_sum = tf.summary.scalar('accuracy', self.accuracy)
        if self.write_images:
            img_sum = tf.summary.image('train_images', self.inputs)
        self.summary = tf.summary.merge_all()

    def train(self, learning_rate, momentum, weight_decay, model_id, val_itr,
              save_itr, num_blocks, restore, num_epochs):
        tvs = tf.trainable_variables()
        if weight_decay is not None:
            weights_norm = tf.reduce_sum(
                tf.stack([tf.nn.l2_loss(var) for var in tvs]))
            self.loss += weight_decay * weights_norm
        #pprint.pprint(tvs)
        opt = tf.train.AdamOptimizer(
            learning_rate=learning_rate, beta1=momentum)
        self.saver = tf.train.Saver(max_to_keep=10)
        train_tvs = tvs[:]
        print(train_tvs is tvs)
        for i in range(1, 5 - num_blocks):
            string = "block" + str(i)
            print("excluding block {}".format(string))
            train_tvs = [var for var in train_tvs if string not in var.name]
        train_tvs = train_tvs[2:]
        #pprint.pprint(train_tvs)

        train_step = opt.minimize(self.loss, var_list=train_tvs)

        if restore:
            model_dir = 'models/id/m{}'.format(model_id)
            print(model_dir)
            path = tf.train.latest_checkpoint(model_dir)
            print(path)
            start_epoch = 1 + int(path.split('.ckpt')[1].split('-')[1])
            print("restoring model in {} with epoch".format(
                model_dir, start_epoch))
            self.saver.restore(self.sess, path)
        else:
            try:
                model_id = 1 + max([
                    int(f.split('/')[-1][1:]) for f in glob('./models/id/m*')
                ])
            except ValueError:
                model_id = 0
            model_dir = 'models/id/m{}'.format(model_id)
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
            start_epoch = 0
            print("Initializing new variables..")
            self.sess.run(tf.global_variables_initializer())
            vars_ = [
                var for var in tvs
                if 'biases' not in var.name and 'logits' not in var.name
            ]
            pprint.pprint(vars_)
            saver = tf.train.Saver(vars_)
            path = 'pretrained/resnet_v1_50.ckpt'
            saver.restore(self.sess, path)
        log_dir = os.path.join(model_dir, 'logs')
        train_writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        val_writer = tf.summary.FileWriter(
            os.path.join(log_dir, 'val'), self.sess.graph)

        idx = 0
        print("Starting training")

        for epoch in range(start_epoch, num_epochs):
            self.sess.run(self.dataset.training_init_op)
            while True:
                try:
                    batch_loss, batch_acc, batch_sum, _ = self.sess.run(
                        [self.loss, self.accuracy, self.summary, train_step],
                        {self.is_train: True})
                    train_writer.add_summary(batch_sum, idx)
                    idx += 1
                    if batch_loss > 100:
                        print("Big loss", idx)
                    if not idx % 100:
                        print("idx: %2d, loss = %.5f, accuracy = %.5f" % \
                                (idx, batch_loss, batch_acc))
                    if idx % save_itr == 1:
                        self.saver.save(
                            self.sess,
                            os.path.join(model_dir, 'model.ckpt'),
                            global_step=idx,
                            write_meta_graph=not bool(epoch))
                except tf.errors.OutOfRangeError:
                    break

            self.sess.run(self.dataset.validation_init_op)

            while True:
                try:
                    val_loss, val_acc, val_sum = self.sess.run(
                        [self.loss, self.accuracy, self.summary],
                        {self.is_train: False})
                    val_writer.add_summary(val_sum, idx)
                    print("idx: %2d, VAL loss = %.5f, VAL accuracy = %.5f" % \
                            (idx, val_loss, val_acc))
                    #validate
                except tf.errors.OutOfRangeError:
                    break
        print("Done Training")
