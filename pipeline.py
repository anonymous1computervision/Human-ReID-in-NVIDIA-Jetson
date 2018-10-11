import numpy as np
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.tools import optimize_for_inference_lib
import random
import os
import model_reid
import cv2
import time
import sys
from glob import glob
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
from tf_trt_models.detection import download_detection_model
from tf_trt_models.detection import build_detection_graph

EMBEDDING_DIM = 2048
TEST_MODEL_DIR = ''
OBJ_MODEL_DIR = './obj_data/'
OBJ_MODEL_NAME = 'ssd_mobilenet_v1_coco'
OBJ_TRT_PATH = './obj_data/' + OBJ_MODEL_NAME + '_trt.pb'
OBJ_PREFIX = 'object_detection'
REID_CKPT = './models/m74/reid_model.ckpt-200'
REID_TRT_PATH = './reid_data/trt.pb'
REID_PREFIX = 'reid'
IMG_PATH = './test_imgs/'
H = 256
W = 128


def obj_det_graph(obj_model, obj_model_dir, trt_graph_path):
    make_new = 'obj' in sys.argv
    if os.path.exists(trt_graph_path) and not make_new:
        trt_graph = tf.GraphDef()
        with open(trt_graph_path, 'rb') as f:
            trt_graph.ParseFromString(f.read())
    else:
        config_path, checkpoint_path = download_detection_model(
            obj_model, obj_model_dir)
        frozen_graph, input_names, output_names = build_detection_graph(
            config=config_path, checkpoint=checkpoint_path)
        print("Making a TRT graph for the object detection model")
        trt_graph = trt.create_inference_graph(
            input_graph_def=frozen_graph,
            outputs=output_names,
            max_batch_size=1,
            max_workspace_size_bytes=1 << 26,
            precision_mode='FP32',
            minimum_segment_size=50)
        with open(trt_graph_path, 'wb') as f:
            f.write(trt_graph.SerializeToString())

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config)

    tf.import_graph_def(trt_graph, name=OBJ_PREFIX)
    tf_input = tf_sess.graph.get_tensor_by_name(OBJ_PREFIX + '/input:0')
    tf_scores = tf_sess.graph.get_tensor_by_name(OBJ_PREFIX + '/scores:0')
    tf_boxes = tf_sess.graph.get_tensor_by_name(OBJ_PREFIX + '/boxes:0')
    tf_classes = tf_sess.graph.get_tensor_by_name(OBJ_PREFIX + '/classes:0')
    return tf_sess, tf_scores, tf_boxes, tf_classes, tf_input


def boxes_to_reid(image, scores, boxes, classes):
    h, w = image.shape[:2]
    score_thresh = 0.36
    persons = []
    #make sure the input image is normalized (?)
    for i in range(len(scores)):
        if scores[i] > score_thresh and classes[i] == 0:
            box = boxes[i] * np.array([
                image.shape[0], image.shape[1], image.shape[0], image.shape[1]
            ])
            box = box.astype(np.int32)
            person_slice = image[box[0]:box[2], box[1]:box[3], :]
            resized = cv2.resize(person_slice, (W, H))
            persons += [resized]
    return np.array(persons)


MEAN = np.array([0.4399, 0.4309, 0.4458]).reshape(1, 1, 3)
STD = np.array([0.1998, 0.2028, 0.1905]).reshape(1, 1, 3)


def reid_preprocess(image):
    tf_x_float = tf.cast(image, tf.float32)
    tf_x_div = tf_x_float / 255.0
    return (tf_x_div - MEAN) / STD


def reid_graph(sess, trt_graph_path, reid_ckpt_path=None):
    make_new = False
    for argv in sys.argv:
        if 'ckpt' in argv:
            make_new = True
            reid_ckpt_path = argv
    if not os.path.exists(trt_graph_path) or make_new:
        assert reid_ckpt_path is not None, 'reid_ckpt_path cannot be none'
        bounding_box_placeholder = tf.placeholder(tf.float32, [None, H, W, 3],
                                                  'reid_input')
        tf_preprocessed = reid_preprocess(bounding_box_placeholder)
        reid_model = model_reid.ReidNetwork(
            sess=sess,
            embedding_dim=EMBEDDING_DIM,
            train=False,
            in_tensor=tf_preprocessed,
            test_model_dir=reid_ckpt_path,
            height=H,
            width=W)
        tf_embedding = reid_model.get_out_tensor()
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_node_names=["embedding"])

        print("Making a TRT graph for the reidentification model")
        trt_graph = trt.create_inference_graph(
            input_graph_def=frozen_graph,
            outputs=["embedding"],
            max_batch_size=10,
            max_workspace_size_bytes=1 << 26,
            precision_mode='FP16',
            minimum_segment_size=50)
        print("writing to {}".format(trt_graph_path))
        with open(trt_graph_path, 'wb') as f:
            f.write(trt_graph.SerializeToString())
    else:
        trt_graph = tf.GraphDef()
        with open(trt_graph_path, 'rb') as f:
            trt_graph.ParseFromString(f.read())

    tf.import_graph_def(trt_graph, name=REID_PREFIX)
    tf_input = sess.graph.get_tensor_by_name(REID_PREFIX + '/reid_input:0')
    tf_output = sess.graph.get_tensor_by_name(REID_PREFIX + '/embedding:0')
    return tf_input, tf_output


def execute(images):
    sess, tf_scores, tf_boxes, tf_classes, tf_input = obj_det_graph(
        OBJ_MODEL_NAME, OBJ_MODEL_DIR, OBJ_TRT_PATH)
    reid_in, reid_out = reid_graph(sess, REID_TRT_PATH, REID_CKPT)
    for image in images:
        start_obj = time.time()
        scores, boxes, classes = sess.run(
            [tf_scores, tf_boxes, tf_classes],
            feed_dict={tf_input: image[None, ...]})
        end_obj = time.time()
        boxes = boxes[0]
        scores = scores[0]
        classes = classes[0]
        batch = boxes_to_reid(image, scores, boxes, classes)
        print("{} people identified".format(batch.shape[0]))
        if not batch.shape[0]:
            continue
        end_process = time.time()
        embeddings = sess.run(reid_out, feed_dict={reid_in: batch})
        end_reid = time.time()
        print("obj det: {} \nprocess {} \nreid {}".format(
            end_obj - start_obj, end_process - end_obj,
            end_reid - end_process))


def open_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


if __name__ == '__main__':
    execute([open_image(pth) for pth in glob('./test_imgs/ped*')])
