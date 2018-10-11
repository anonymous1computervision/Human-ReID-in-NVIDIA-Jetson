from glob import glob
import numpy as np
import random

class ImageDataset(object):
    def __init__(self, data_dir, **kwargs):
        self.data_dir = data_dir
        self.h = kwargs.get('h', 256)
        self.w = kwargs.get('w', 128)
        self.all_img_files = self.get_all_jpgs()
        self.id_dict, self.count = self.get_classes_dict()

    def get_class_from_path(self, path):
        #IF USING DIFFERENT DATASET: change this to be correct.
        #this assumes that all images are in some directory, d
        #and their paths are d/<<ID>>_<OTHER_INFO>.jpg, as in Duke or Market.
        return int(path.split('/')[-1].split('_')[0])

    def get_classes_dict(self):
        id_list = []
        count = 0
        for img_file in self.all_img_files:
            person_id = self.get_class_from_path(img_file)
            if person_id not in id_list:
                id_list += [person_id]
                count += 1
        id_list.sort()
        id_dict = {}
        for i in range(len(id_list)):
            id_dict[id_list[i]] = i
        return id_dict, count

    def get_all_jpgs(self):
        return glob(self.data_dir + "*.jpg")

    def split_train_val(self, data, p=0.9):
        random.seed(106)
        random.shuffle(data)
        split_index = int(len(data) * p)
        train = data[:split_index]
        val = data[split_index:]
        return train, val
    def get_id_from_path(self, path):
        class_ = self.get_class_from_path(path)
        return self.id_dict[class_]

