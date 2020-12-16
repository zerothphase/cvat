import json
import base64
import io
from PIL import Image
import yaml
from model_loader import ModelLoader


def init_context(context):
    context.logger.info("Init context...  0%")
    model_prefix = './2d106det'
    model_handler = ModelLoader(model_prefix=model_prefix)
    setattr(context.user_data, 'model_handler', model_handler)
    functionconfig = yaml.safe_load(open("/opt/nuclio/function.yaml"))
    labels_spec = functionconfig['metadata']['annotations']['spec']
    labels = {item['id']: item['name'] for item in json.loads(labels_spec)}
    setattr(context.user_data, "labels", labels)
    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"].encode('utf-8')))
    threshold = float(data.get("threshold", 0.5))
    image = Image.open(buf)

    faces, faces_landmarks = context.user_data.model_handler.infer(image)

    results = []
    for face, landmarks in zip(faces, faces_landmarks):
        face_class = 1
        landmarks_class = 2
        face_label = context.user_data.labels.get(face_class, "unknown")
        landmarks_label = context.user_data.labels.get(landmarks_class, "unknown")

        results.append({
            "label": face_label,
            "points": face,
            "type": "rectangle",
        })

        results.append({
            "label": landmarks_label,
            "points": landmarks,
            "type": "points",
        })

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)
