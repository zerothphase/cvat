import json
import base64
import io
from PIL import Image
import yaml
from model_loader import ModelLoader


def init_context(context):
    context.logger.info("Init context...  0%")
    # model_path = "/opt/nuclio/faster_rcnn/frozen_inference_graph.pb"
    model_handler = ModelLoader()
    setattr(context.user_data, 'model_handler', model_handler)
    functionconfig = yaml.safe_load(open("/opt/nuclio/function.yaml"))
    labels_spec = functionconfig['metadata']['annotations']['spec']
    labels = {item['id']: item['name'] for item in json.loads(labels_spec)}
    setattr(context.user_data, "labels", labels)
    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run face_recognition cnn model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"].encode('utf-8')))
    threshold = float(data.get("threshold", 0.5))
    image = Image.open(buf)

    boxes = context.user_data.model_handler.infer(image)

    results = []
    for i in range(len(boxes)):
        obj_class = 1
        obj_label = context.user_data.labels.get(obj_class, "unknown")
        # top right bottom left
        xtl = boxes[i][3]
        ytl = boxes[i][0]
        xbr = boxes[i][1]
        ybr = boxes[i][2]

        results.append({
            "confidence": 0.99,
            "label": obj_label,
            "points": [xtl, ytl, xbr, ybr],
            "type": "rectangle",
        })

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)
