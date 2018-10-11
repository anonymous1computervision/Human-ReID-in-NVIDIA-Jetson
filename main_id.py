import os
import tensorflow as tf
import model_id

FLAGS = tf.app.flags
FLAGS.DEFINE_integer("num_epochs", 25, "num iters to train to")
FLAGS.DEFINE_integer("save_itr", 250, "num iters between saves")
FLAGS.DEFINE_integer("val_itr", 50, "num iters between val evals")
FLAGS.DEFINE_integer("batch_size", 128, "batch size")
FLAGS.DEFINE_integer("blocks", 2, "number of blocks to train")
FLAGS.DEFINE_integer("height", 256, "height of input images")
FLAGS.DEFINE_integer("width", 128, "width of input images")
FLAGS.DEFINE_integer("y_dim", 702, "number of classes")
FLAGS.DEFINE_integer("id", -1, "model from which to restore")
FLAGS.DEFINE_float("learning_rate", 3e-4, "learning rate for Adam")
FLAGS.DEFINE_float("momentum", 0.9,
                   "b1 term for Adam, momentum for other opts")
FLAGS.DEFINE_float("crop_prop", 0.9,
                   "approx proportion of image left after crop")
FLAGS.DEFINE_float("weight_decay", 1e-5, "weight decay / lambda")
FLAGS.DEFINE_string("data_dir", "./data/bounding_box_train/",
                    "directory where data is")
FLAGS.DEFINE_boolean("restore", True, "whether to restore an existing model")
FLAGS.DEFINE_boolean("write_images", False,
                     "whether to write training images to tboard")

FLAGS = FLAGS.FLAGS


def main(_):

    with tf.Session() as sess:
        model = model_id.ID_ResNet(
            sess,
            n_labels=FLAGS.y_dim,
            data_dir=FLAGS.data_dir,
            batch_size=FLAGS.batch_size,
            h=FLAGS.height,
            w=FLAGS.width,
            crop_prop=FLAGS.crop_prop,
            write_images=FLAGS.write_images)
        model.train(
            learning_rate=FLAGS.learning_rate,
            momentum=FLAGS.momentum,
            weight_decay=FLAGS.weight_decay,
            model_id=FLAGS.id,
            val_itr=FLAGS.val_itr,
            save_itr=FLAGS.save_itr,
            num_blocks=FLAGS.blocks,
            restore=FLAGS.restore,
            num_epochs=FLAGS.num_epochs)


if __name__ == '__main__':
    tf.app.run()
