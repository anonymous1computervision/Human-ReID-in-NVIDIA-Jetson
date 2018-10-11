import tensorflow as tf
import os
import random
import numpy as np
import model_reid
import eval_reid
import pprint
import time
from utils import dataset, misc

flags = tf.app.flags
flags.DEFINE_integer("num_epochs", 25, "number of epochs to train")
flags.DEFINE_integer("save_itr", 1, "num epochs between saves")
flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_integer("embedding_dim", 2048, "dimension of embedding")
flags.DEFINE_integer("N", 5000, "number of examples for hard mining")
flags.DEFINE_integer("sample_num", 25,
                     "approx number of hard triplets to sample from")
flags.DEFINE_integer("k", 16, "number of updates before new N fetch")
flags.DEFINE_integer("id", -1, "id to restore from, if none does most recent")
flags.DEFINE_integer("height", 256, "height")
flags.DEFINE_integer("width", 128, "width")
flags.DEFINE_float("p_0", 0.0, "initial proportion of noised image")
flags.DEFINE_float("p_max", 0.5, "final proportion of noised image")
flags.DEFINE_float("learning_rate", 1e-5, "learning rate")
flags.DEFINE_float("momentum", 0.9, "beta1 for adam")
flags.DEFINE_float("margin", 0.1, "margin for triplet loss")
flags.DEFINE_float("threshold", None,
                   "threshold loss for training, None default")
flags.DEFINE_string("data_dir", "./data/bounding_box_train/",
                    "directory where data is")
flags.DEFINE_string("init_model", None, "where the model to init from is")
flags.DEFINE_string("test_model_dir", None, "where the model for testing is")
flags.DEFINE_string("test_dir", "./data/bounding_box_test/",
                    "where the images to test on are")
flags.DEFINE_string("query_dir", "./data/bounding_box_query/",
                    "where the queries at")
flags.DEFINE_boolean("restore", True, "whether to restore an existing model")
flags.DEFINE_boolean("train", True, "training?")
flags.DEFINE_boolean("write_images", False, "write input triplets to t-board?")

FLAGS = flags.FLAGS


def main(_):
    with tf.Session() as sess:
        if FLAGS.train:
            model = model_reid.ReidNetwork(
                sess=sess,
                embedding_dim=FLAGS.embedding_dim,
                train=True,
                data_dir=FLAGS.data_dir,
                height=FLAGS.height,
                width=FLAGS.width,
                margin=FLAGS.margin,
                threshold=FLAGS.threshold,
                write_images=FLAGS.write_images,
            )
            model.train(FLAGS)
        else:
            print(FLAGS.test_dir)
            print(FLAGS.query_dir)
            print(FLAGS.test_model_dir)
            model = model_reid.ReidNetwork(
                sess=sess,
                embedding_dim=FLAGS.embedding_dim,
                train=False,
                test_model_dir=FLAGS.test_model_dir,
                height=FLAGS.height,
                width=FLAGS.width)
            dh = dataset.ImageDataset(
                FLAGS.test_dir, h=FLAGS.height, w=FLAGS.width)
            acc = eval_reid.top_k_acc(model, sess, dh, FLAGS.query_dir, 5)
            return


if __name__ == '__main__':
    tf.app.run()
