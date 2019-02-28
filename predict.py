from flask import Flask, request, json
import numpy as np
import tensorflow as tf
import base64
import cv2

with tf.gfile.FastGFile('trained_graph.pb', 'rb') as f:
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
    img = data_uri_to_cv2_img(request.form['image'])
    buffer = cv2.imencode('.jpg', img)[1].tostring()
    predictions = sess.run(sess.graph.get_tensor_by_name('final_result:0'), \
                    {'DecodeJpeg/contents:0': buffer})
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    return str(top_k[0])
def data_uri_to_cv2_img(uri):
    #encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(uri), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)