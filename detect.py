from PIL import Image
from flask import Flask, request, json
import numpy as np
import requests
import tensorflow as tf
import base64
import cv2

with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
sess=tf.Session()
sess.graph.as_default()
tf.import_graph_def(graph_def, name='')

app = Flask(__name__)
@app.route("/")
def main():
    return "Welcome!"
@app.route('/detect',methods=['POST'])
def detect():
    _image = request.form['image']
    img=data_uri_to_cv2_img(_image)
    rows = img.shape[0]
    cols = img.shape[1]
    inp = cv2.resize(img, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
    # Visualize detected bounding boxes.
    num_detections = int(out[0][0])
    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > 0.5:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            retval, buffer = cv2.imencode('.jpg', img[int(y):int(bottom), int(x):int(right)])
            r = requests.post(url = "http://127.0.0.1:5000/predict", data = {'image':base64.b64encode(buffer)}) 
            return json.dumps({'x':int(x),'y':int(y),'right':int(right),'bottom':int(bottom),'sign':int(r.text)}) 
    return json.dumps({'x':0,'y':0,'right':0,'bottom':0,'sign':0}) 
def data_uri_to_cv2_img(uri):
    #encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(uri), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)