import cv2 as cv
import requests 
import base64
import numpy as np
import json
from pygame import mixer
mixer.init()
cap = cv.VideoCapture(1)
while True:
    ret, img = cap.read()
    retval, buffer = cv.imencode('.jpg',img,[int(cv.IMWRITE_JPEG_QUALITY), 70])
    r = requests.post(url = "http://127.0.0.1:5001/detect", data = {'image':base64.b64encode(buffer)}) 
    jo = json.loads(r.text)
    if jo['sign']!=0 and jo['sign']!=4 and jo['sign']!=6 and not mixer.music.get_busy(): 
        mixer.music.load('sound/'+str(jo['sign'])+'.mp3')
        mixer.music.play()
    cv.rectangle(img, (jo["x"], jo["y"]), (jo["right"], jo["bottom"]), (125, 255, 51), thickness=2)
    cv.imshow("img",img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break