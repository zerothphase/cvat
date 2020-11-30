
import numpy as np
from PIL import Image
import cv2

class ModelLoader:

    def __init__(self, model_proto=None, model_caffe=None):
        self.net = cv2.dnn.readNetFromCaffe(model_proto, model_caffe)
        self.threshold = 0.5

    def infer(self, image):
        '''
        image: PIL image
        '''
        w, h = image.size

        # TODO assert image type (RGB vs GRAY)

        # top right bottom left
        np_image = np.array(image)
        blob = cv2.dnn.blobFromImage(np_image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence >= self.threshold:
                # x1, y1, x2, y2
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                box = box.astype("int")
                faces.append((box, confidence))

        return faces
