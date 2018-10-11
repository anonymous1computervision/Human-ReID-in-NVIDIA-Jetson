import pipeline
import numpy as np
import random
import sys
import cv2
import time
import argparse
from argus_camera import ArgusCamera

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=2048)
parser.add_argument('--obj_model', type=str, default='ssd_mobilenet_v1_coco')
args = parser.parse_args()


class Demo():
    '''Class that runs the demo'''

    def __init__(self, score_thresh=.4, query_size=(128, 256), in_=(256, 128)):

        self.score_thresh = score_thresh
        self.query_size = query_size
        self.in_h = in_[0]
        self.in_w = in_[1]
        self.query = None
        self.img_boxes = []
        self.query_embedding = None
        self.embeddings = []
        self.people = []
        self.frames = 0
        self.times = [.1]
        self.FPS_BUFFER_SIZE = 8
        self.COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 150, 255),
                       (0, 255, 255), (200, 0, 200), (255, 191, 0), (180, 105,
                                                                     255)]
        self.RESOLUTION = (1080, 720)
        self.MAX_QUERY = self.RESOLUTION[0] // self.query_size[0]
        self.camera = ArgusCamera(
            stream_resolution=self.RESOLUTION,
            video_converter_resolution=self.RESOLUTION)

    def animate(self):
        """Called repeatedly. Does obj det, reid, and displays results"""

        start_time = time.time()
        image = self.get_next_image()
        self.frames += 1
        display = np.array(image)
        if self.query is not None:
            display[:self.query.shape[0], :self.query.shape[1], :] = self.query

        # run object detection
        scores, boxes, classes = self.sess.run(
            [self.tf_scores, self.tf_boxes, self.tf_classes],
            feed_dict={self.tf_input: image[None, ...]})
        scores = scores[0]
        boxes = boxes[0]
        classes = classes[0]
        persons = []
        self.people = []
        self.img_boxes = []

        # Iterate through the detected bounding boxes, add to batch"
        for i in range(len(scores)):
            if scores[i] > self.score_thresh and classes[i] == 0:
                box = boxes[i] * np.array([
                    display.shape[0], display.shape[1], display.shape[0],
                    display.shape[1]
                ])
                box = box.astype(np.int32)
                self.img_boxes += [box]
                person_slice = image[box[0]:box[2], box[1]:box[3], :]
                rescaled = cv2.resize(person_slice, (128, 256))
                persons += [rescaled]
                self.people += [rescaled]
        display = display[:, :, ::-1]
        overlay = display.copy()
        output = display.copy()

        # If there are queries, draw their overlay
        if self.query_embedding is not None:
            for i in range(self.query_embedding.shape[0]):
                cv2.rectangle(
                    overlay, (self.query_size[0] * i, 0),
                    (self.query_size[0] * (i + 1), self.query_size[1]),
                    self.COLORS[i], -1)

        # draw fps
        avg_fps = len(self.times) / sum(self.times)
        cv2.putText(output, "{:.2f} fps".format(avg_fps),
                    (130, self.RESOLUTION[1] - 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 1, cv2.LINE_AA)

        #if there are no people, add overlay, display image, update fps buffer and exit
        if len(persons) == 0:
            cv2.addWeighted(overlay, 0.22, output, 0.78, 0, output)
            cv2.imshow("image", output)
            if len(self.times) < self.FPS_BUFFER_SIZE:
                self.times = [time.time() - start_time] + self.times
            else:
                self.times = [time.time() - start_time] + self.times[:-1]
            cv2.waitKey(1)
            return

        #run reid
        batch = np.array(persons)
        self.embeddings = self.sess.run(
            self.reid_out, feed_dict={self.reid_in: batch})

        #run matching algorithm
        return_vec = None
        if self.query is not None:
            return_vec, sims = self.matching(self.query_embedding,
                                             self.embeddings)

        #Draw relevant info on bounding boxes
        for i, box in enumerate(self.img_boxes):
            if return_vec is not None and return_vec[i] != -1:
                cv2.putText(output, "{:.3f}".format(
                    sims[i]), (box[1] + 10, box[2] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1,
                            cv2.LINE_AA)
                color = self.COLORS[return_vec[i]]
                cv2.rectangle(overlay, (box[1], box[0]), (box[3], box[2]),
                              color, -1)
            else:
                color = (255, 255, 255)
                cv2.rectangle(overlay, (box[1], box[0]), (box[3], box[2]),
                              color, -1)
        #Add overlay, show image, update fps buffer, and exit
        cv2.addWeighted(overlay, 0.22, output, 0.78, 0, output)
        cv2.imshow("image", output)
        cv2.waitKey(1)
        if len(self.times) < self.FPS_BUFFER_SIZE:
            self.times = [time.time() - start_time] + self.times
        else:
            self.times = [time.time() - start_time] + self.times[:-1]

    def onrelease(self, event, x, y, flags, param):
        """Runned on release, add image and its embeddings to current queries"""
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, box in enumerate(self.img_boxes):
                if y < box[2] and y > box[0] and \
                    x < box[3] and x > box[1]:
                    if self.query is None:
                        self.query = cv2.resize(self.people[i],
                                                self.query_size)
                    else:
                        this_query = cv2.resize(self.people[i],
                                                self.query_size)
                        if (self.query.shape[1] + this_query.shape[1]
                            ) / self.query_size[0] > self.MAX_QUERY:
                            self.query = self.query[:, this_query.shape[1]:, :]
                        self.query = np.concatenate(
                            [self.query, this_query], axis=1)
                    if self.query_embedding is None:
                        self.query_embedding = np.reshape(
                            self.embeddings[i], (-1, args.embedding_dim))
                        print("Initialized Query Embedding")
                        return
                    else:
                        similarity = np.dot(self.query_embedding,
                                            self.embeddings[i].T)
                        if self.query_embedding.shape[0] < self.MAX_QUERY:
                            self.query_embedding = np.vstack(
                                [self.query_embedding, self.embeddings[i]])
                        else:
                            self.query_embedding = np.vstack(
                                [self.query_embedding[1:], self.embeddings[i]])
        if event == cv2.EVENT_RBUTTONDOWN:
            #Right click exits demo
            self.going = False
        return

    def get_next_image(self):
        """Read from camera"""
        return np.array(self.camera.read())[:, :, :3]

    def matching(self, query_embeddings, camera_embeddings):
        """A greedy algorithm to match query and camera images via embeddings.

        Arguments:
        query_embeddings -- embeddings of query
        camera_embeddings -- embeddings of people captured by camera
        Returns:
        return vec -- a vector which matches the n'th embedding to one of the queries.
        return sims -- a vector with the similarity scores for each matchings.
        """
        query_embeddings = np.reshape(query_embeddings,
                                      (-1, args.embedding_dim))
        camera_embeddings = np.reshape(camera_embeddings,
                                       (-1, args.embedding_dim))
        query_num = query_embeddings.shape[0]
        camera_num = camera_embeddings.shape[0]
        dot_matrix = np.dot(query_embeddings, camera_embeddings.T)
        dot_matrix = np.reshape(dot_matrix, (query_num, camera_num))

        #since not all will be matched, default is -1:
        return_vector = -np.ones(shape=camera_num)
        return_sims = -np.ones(shape=camera_num)
        for i in range(camera_num):
            argmax = np.unravel_index(np.argmax(dot_matrix), dot_matrix.shape)
            max_sim = dot_matrix[argmax]
            if dot_matrix[argmax] < 0.3:
                break
            dot_matrix[argmax[0], :] = -10000.
            dot_matrix[:, argmax[1]] = -10000.
            return_vector[argmax[1]] = argmax[0]
            return_sims[argmax[1]] = max_sim
        return return_vector.astype(np.int32), return_sims

    def demo(self):
        """Runs the demo"""

        #get some graphs
        self.sess, self.tf_scores, self.tf_boxes, self.tf_classes, self.tf_input\
            = pipeline.obj_det_graph(
                args.obj_model,
                './obj_data',
                './obj_data/'+args.obj_model+'_trt.pb'
        )
        self.reid_in, self.reid_out = pipeline.reid_graph(
            self.sess, './reid_data/trt.pb')

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.onrelease)
        self.going = True
        while self.going:
            self.animate()
        cv2.destroyAllWindows()


demo = Demo()
demo.demo()
