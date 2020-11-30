
import numpy as np
from PIL import Image
import face_recognition

class ModelLoader:

    def __init__(self):
        self.model = None


    def infer(self, image):
        '''
        image: PIL image
        '''
        # width, height = image.size
        # if width > 1920 or height > 1080:
        #     image = image.resize((width // 2, height // 2), Image.ANTIALIAS)

        # top right bottom left
        faces = face_recognition.face_locations(np.array(image))
        return faces
