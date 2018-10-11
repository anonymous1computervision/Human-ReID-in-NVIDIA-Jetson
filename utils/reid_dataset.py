import numpy as np
import time
import random
from . import dataset
from . import misc

class ReIDDataset(dataset.ImageDataset):
    def __init__(self, data_dir, **kwargs):
        super(ReIDDataset, self).__init__(data_dir)
        self.N = kwargs.get('N')
        self.k = kwargs.get('k')
        self.batch_size = kwargs.get('batch_size')
        self.sample_num = kwargs.get('sample_num')

        self.sess = kwargs.get('sess')
        self.is_train_pholder = kwargs.get('is_train_pholder')
        self.in_pholder = kwargs.get('in_pholder')
        self.emb_op = kwargs.get('emb_op')

    def get_random_triplets(self, x, y, max_length=64*32):
        print("Getting validation triplets...")
        assert len(x) == len(y)
        if len(x) > max_length:
            x, y = zip(*random.sample(list(zip(self.train_x, self.train_y)), max_length))
        imgs = [misc.parse_image(ex, self.h, self.w, 0) for ex in x]
        idx_dict = misc.get_idx_dict(y)
        batch = []
        for idx in range(len(x)):
            q_label = y[idx]
            positives = idx_dict[q_label]
            negatives = []
            for i in idx_dict:
                if not i == q_label:
                    negatives += idx_dict[i]
            positives = np.array(positives)
            negatives = np.array(negatives)
            cn = random.randrange(len(negatives)//3)
            choices = np.array(np.meshgrid(range(len(positives)), cn)).T.reshape(-1, 2)
            batch += [[idx, positives[x[0]], negatives[x[1]]] for x in list(choices)]

        img_batch = [[imgs[i] for i in x] for x in batch]
        return img_batch

    def hard_triplet_mine(self, p):
        start_time = time.time()
        x, y = zip(*random.sample(list(zip(self.train_x, self.train_y)), self.N))
        idx_dict = misc.get_idx_dict(y)

        imgs = np.array([misc.parse_image(img, self.h, self.w, p) for img in x])
        def feed(img):
            return {self.in_pholder:img, self.is_train_pholder:False}
        embedded = np.array([self.sess.run(self.emb_op, feed(img)) for img in imgs])

        query_indices = random.sample(range(self.N), self.k * self.batch_size)
        batch = []
        for idx in query_indices:
            qlabel = y[idx]
            positives = idx_dict[qlabel]
            negatives = []
            for i in idx_dict:
                if not i == qlabel:
                    negatives += idx_dict[i]
            positives = np.array(positives)
            negatives = np.array(negatives)
            emb_pos = np.squeeze(embedded[positives])
            emb_neg = np.squeeze(embedded[negatives])
            q = embedded[idx]
            qpt = np.dot(q, emb_pos.T)
            qnt = np.dot(q, emb_neg.T)
            num_pos = min(len(positives)-1, int(np.sqrt(self.sample_num)))
            num_neg = self.sample_num // (num_pos + 1) + 1
            if num_pos > 0:
                min_qpt = np.argpartition(qpt, kth=num_pos)[:num_pos]
            else:
                min_qpt = np.array([[1]])
            max_qnt = np.argpartition(-qnt, kth=num_neg)[:num_neg]
            choices = np.array(np.meshgrid(min_qpt, max_qnt)).T.reshape(-1, 2)
            chosen = np.random.randint(low=0, high=choices.shape[0])
            if len(positives) == 1:
                c_pos = 0
            else:
                c_pos = choices[chosen][0]
            batch += [[idx, positives[c_pos], negatives[choices[chosen][1]]]]
        end_time = time.time()
        print('{:3f} seconds for hard triplet mine'.format(end_time-start_time))
        img_batch = [[imgs[i] for i in x] for x in batch]
        return img_batch

    def get_data_partition(self):
        train, val = self.split_train_val(self.all_img_files, 0.97)
        train_y = [self.get_id_from_path(fp) for fp in train]
        val_y = [self.get_id_from_path(fp) for fp in val]
        self.train_x = train
        self.train_y = train_y
        self.val_x = val
        self.val_y = val_y
