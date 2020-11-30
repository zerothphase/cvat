import json
import base64
import io
from PIL import Image
import yaml
from model_loader import ModelLoader


def init_context(context):
    context.logger.info("Init context...  0%")
    model_proto = "/opt/nuclio/deploy.prototxt"
    model_caffe = "/opt/nuclio/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    model_handler = ModelLoader(model_proto=model_proto, model_caffe=model_caffe)
    setattr(context.user_data, 'model_handler', model_handler)
    functionconfig = yaml.safe_load(open("/opt/nuclio/function.yaml"))
    labels_spec = functionconfig['metadata']['annotations']['spec']
    labels = {item['id']: item['name'] for item in json.loads(labels_spec)}
    setattr(context.user_data, "labels", labels)
    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run opencv dnn model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"].encode('utf-8')))
    threshold = float(data.get("threshold", 0.5))
    image = Image.open(buf)

    boxes = context.user_data.model_handler.infer(image)

    results = []
    for i in range(len(boxes)):
        # note: need to convert numpy types to python types to serialize
        obj_class = 1
        confidence = str(boxes[i][1])
        obj_label = context.user_data.labels.get(obj_class, "unknown")
        xtl = int(boxes[i][0][0])
        ytl = int(boxes[i][0][1])
        xbr = int(boxes[i][0][2])
        ybr = int(boxes[i][0][3])

        results.append({
            "confidence": confidence,
            "label": obj_label,
            "points": [xtl, ytl, xbr, ybr],
            "type": "rectangle",
        })

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)
