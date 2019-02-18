from PIL import Image
from flask import Flask, request, json
import numpy as np
import tensorflow as tf
import base64
import cv2

with tf.gfile.FastGFile('frozen_inference_graph (4).pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
sess=tf.Session()
sess.graph.as_default()
tf.import_graph_def(graph_def, name='')

app = Flask(__name__)
@app.route("/")
def main():
    return "Welcome!"
@app.route('/predict',methods=['POST'])
def predict():
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
        if score > 0.5:
            return str(classId)
    return "0"
def data_uri_to_cv2_img(uri):
    #encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(uri), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)