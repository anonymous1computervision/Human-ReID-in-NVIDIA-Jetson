import numpy as np
import cv2
import os
from glob import glob
import random
MEAN = np.array([0.4399, 0.4309, 0.4458])
STD = np.array([0.1998, 0.2028, 0.1905])
MEAN = np.reshape(MEAN, [1,1,3])
STD = np.reshape(STD, [1,1,3])

def scale_to_fit(image, shape=(256, 128)):
    target_size = list(shape)
    target_ratio = target_size[1]/target_size[0]
    ratio = image.shape[1]/image.shape[0]
    if ratio < target_ratio:
        rsz_prop = target_size[0]/image.shape[0]
        resize_shape = [round(rsz_prop * image.shape[0]), round(rsz_prop * image.shape[1])]
        rsz = cv2.resize(image, (resize_shape[1], resize_shape[0]))
        assert rsz.shape[0] == target_size[0]
        noisy = np.random.normal(0, 1., target_size + [3])
        left = (target_size[1] - rsz.shape[1])//2
        right = left + rsz.shape[1]
        noisy[:, left:right, :] = rsz
        return noisy
    if ratio > target_ratio:
        rsz_prop = target_size[1]/image.shape[1]
        resize_shape = [round(rsz_prop * image.shape[0]), round(rsz_prop * image.shape[1])]
        rsz = cv2.resize(image, (resize_shape[1], resize_shape[0]))
        assert rsz.shape[1] == target_size[1]
        noisy = np.random.normal(0, 1., target_size + [3])
        left = (target_size[0] - rsz.shape[0])//2
        right = left + rsz.shape[0]
        noisy[left:right, :, :] = rsz
        return noisy

def get_image(filename, shape=None, scale_fit=False, rescale=False):
    img = cv2.imread(filename)
    assert img is not None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    if rescale:
        img = img * 1.0/255.0
    if shape is not None:
        if scale_fit:
            return scale_to_fit(img, shape)
        ret = cv2.resize(img, (shape[1], shape[0]))
        return ret
    else:
        return img

def preprocess_image(img, mean, std, occlusion=0):
    if occlusion > 0:
        size = img.shape[0] * img.shape[1]
        del_size = int(occlusion * size)
        assert img.shape[1] >= 16
        w = np.random.randint(.5 * np.sqrt(del_size), 2 * np.sqrt(del_size))
        w = min(w, img.shape[1])
        h = min(del_size // w, img.shape[0])
        start_h = np.random.randint(0, img.shape[0] - h + 1)
        start_w = np.random.randint(0, img.shape[1] - w + 1)

        mean_img = np.mean(img, axis=(0,1)).reshape(1,1,3)
        std_img = np.std(img, axis=(0,1)).reshape(1,1,3)

        noise = np.random.normal(mean_img, std_img, [h, w, 3])
        img[start_h:start_h+h,start_w:start_w+w,:] = noise
    return (img - np.expand_dims(mean, 1)) / np.expand_dims(std, 1)

def parse_image(filename, h, w, occlusion=0.):
    image = get_image(filename, shape=(h, w), rescale=True)
    return preprocess_image(image, MEAN, STD, occlusion)

def get_idx_dict(labels):
    idx_dict = {}
    for idx in range(len(labels)):
        class_ = labels[idx]
        if class_ not in idx_dict:
            idx_dict[class_] = [idx]
        else:
            idx_dict[class_] += [idx]
    return idx_dict

def flags_to_dict(flags):
    ret = {}
    for attr in dir(flags):
        if 'help' in attr:
            continue
        value = getattr(flags, attr)
        ret[attr] = value
    return ret






