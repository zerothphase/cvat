
from PIL import Image
import cv2
import sys
import numpy as np
from insightface_coordReg import Handler

class ModelLoader:

    def __init__(self, model_prefix=None):
        self.detector = Handler(model_prefix, 0, ctx_id=0, det_size=640)
        self.threshold = 0.5

    def infer(self, image):
        '''
        image: PIL image
        '''
        w, h = image.size
        image = image.convert("RGB")

        # top right bottom left
        img = np.array(image)


        faces, landmarks_preds = self.detector.get(img, get_all=True)

        if faces is None:
            return [], []

        faces_res = []
        for i in range(faces.shape[0]):
            box = faces[i, :4].astype(np.int).tolist()

            # x1, y1, x2, y2
            faces_res.append(box)

        landmarks_res = []
        for pred in landmarks_preds:
            landmarks = []
            pred = np.round(pred).astype(np.int)
            for i in range(pred.shape[0]):
                x, y = pred[i]
                landmarks.append(int(x))
                landmarks.append(int(y))
            landmarks_res.append(landmarks)
        return faces_res, landmarks_res
