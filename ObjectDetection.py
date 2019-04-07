import os

import numpy as np
from keras import backend as K
from keras.models import load_model
from TrafficSurveillance.models.keras_yolo import yolo_eval, yolo_head

vehicles = {'car', 'bus', 'truck', 'person', 'motorbike', 'train', 'bicycle'}

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score_threshold": 0.3,
        "iou_threshold": 0.5,
        "model_image_size": (608, 608)
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.model = self._get_model()
        self.sess = K.get_session()
        self.input_image_shape = K.placeholder(shape=(2, ))
        self.boxes, self.scores, self.classes = self.generate()


    def _get_model(self):
        return load_model(self.model_path)

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        yolo_outputs = yolo_head(self.model.output, self.anchors, len(self.class_names))
        boxes, scores, classes = yolo_eval(
            yolo_outputs,
            self.input_image_shape,
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold)

        return boxes, scores, classes

    def detect_image(self, image):

        image_data = np.array(image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
                    [self.boxes, self.scores, self.classes],
                    feed_dict={
                        self.model.input: image_data,
                        self.input_image_shape: [image.shape[0], image.shape[1]],
                        K.learning_phase(): 0
                    })

        detected_boxes = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            if predicted_class not in vehicles:
                continue
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))
            detected_boxes.append((top, bottom, left, right))

        return detected_boxes

    def session_close(self):
        self.sess.close()