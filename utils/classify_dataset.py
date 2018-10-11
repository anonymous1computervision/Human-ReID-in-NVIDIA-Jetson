import tensorflow as tf
from . import dataset
import random

class ClassifyDataset(dataset.ImageDataset):
    def __init__(self, data_dir, **kwargs):
        super(ClassifyDataset, self).__init__(data_dir=data_dir)
        self.h = kwargs.get('h', 256)
        self.w = kwargs.get('w', 128)
        self.batch_size = kwargs.get('batch_size', 64)
        self.crop_prop = kwargs.get('crop_prop', 0.9)
        self.num_parallel_calls = kwargs.get('num_parallel_calls', 6)
        self.mean = tf.constant([123.67, 116.28, 103.53], shape=[1,1,1,3])
        self.std = tf.constant([58.40, 56.12, 57.38], shape=[1,1,1,3])
        self._get_datasets()

    def _get_datasets(self):
        train, val = self.split_train_val(self.all_img_files)
        train_y = tf.constant([self.get_id_from_path(fp) for fp in train])
        val_y = tf.constant([self.get_id_from_path(fp) for fp in val])
        val_len = len(val)

        training_set = tf.data.Dataset.from_tensor_slices((train, train_y))
        training_set = training_set.map(self._parse_image_train, num_parallel_calls=self.num_parallel_calls)
        training_set = training_set.shuffle(buffer_size=1000)
        self.training_set = training_set.batch(self.batch_size)
        self.training_set = self.training_set.prefetch(buffer_size=self.batch_size)

        validation_set = tf.data.Dataset.from_tensor_slices((val, val_y))
        validation_set = validation_set.map(self._parse_image_val, num_parallel_calls=self.num_parallel_calls)
        validation_set = validation_set.shuffle(buffer_size=val_len)
        self.validation_set = validation_set.batch(val_len)

        iterator = tf.data.Iterator.from_structure(self.training_set.output_types,
                                                self.training_set.output_shapes)
        self.next_element = iterator.get_next()
        self.training_init_op = iterator.make_initializer(self.training_set)
        self.validation_init_op = iterator.make_initializer(self.validation_set)

    def _parse_image_val(self, filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.cast(tf.image.resize_images(image_decoded, [self.h,self.w]), tf.float32)
        return tf.squeeze((image_resized - self.mean) / self.std), tf.one_hot(label, self.count)

    def _parse_image_train(self, filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.cast(tf.image.decode_jpeg(image_string), tf.float32)
        new_h = tf.cast(tf.scalar_mul(self.crop_prop, tf.cast(tf.shape(image_decoded)[0], tf.float32)), tf.int32)
        new_w = tf.cast(tf.scalar_mul(self.crop_prop, tf.cast(tf.shape(image_decoded)[1], tf.float32)), tf.int32)
        image_cropped = tf.random_crop(image_decoded, [new_h, new_w, 3])
        image_resized = tf.image.resize_images(image_cropped, [self.h, self.w])
        image_maybe_flipped = tf.image.random_flip_left_right(image_resized)
        norm = (image_maybe_flipped - self.mean) / self.std
        return tf.squeeze(norm), tf.one_hot(label, self.count)

