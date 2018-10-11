import tensorflow as tf
import numpy as np
import random
import time
from utils import misc
from glob import glob


def top_k_acc(model, sess, dh, query_dir, n=5):
    """Evaluate top k accuracy on the query and testing set"""
    filenames = dh.all_img_files
    label_dict = misc.get_idx_dict(
        [dh.get_class_from_path(fp) for fp in filenames])
    label_dict_keys = list(label_dict.keys())[:]
    label_dict_keys.sort()
    embedding_dict = {}

    imgs = [misc.parse_image(filename, dh.h, dh.w) for filename in filenames]
    start = time.time()
    gallery_embs = np.squeeze(
        np.array(
            [sess.run(model.embedding, {model.image: img}) for img in imgs]))
    end = time.time()
    print("Speed = {0:.2f} imgs/sec".format(len(gallery_embs) / (end - start)))
    print("Got Embeddings")
    query_files = glob(query_dir + '*.jpg')
    query_imgs = [
        misc.parse_image(filename, dh.h, dh.w) for filename in query_files
    ]
    query_embs = np.array(
        [sess.run(model.embedding, {model.image: img}) for img in query_imgs])
    dots = np.dot(np.squeeze(query_embs), np.squeeze(gallery_embs.T))
    print(dots.shape)
    for k in range(1, n + 1):
        top_k = np.argpartition(-dots, kth=k)[:, :k]
        indexed = np.array(filenames)[top_k]
        ids = [[dh.get_class_from_path(x) for x in y] for y in list(indexed)]
        query_ids = [dh.get_class_from_path(f) for f in query_files]
        acc = sum([(query_ids[i] in x)
                   for i, x in enumerate(ids)]) / float(len(query_ids))
        print("Rank ", k, " accuracy: ", acc)
