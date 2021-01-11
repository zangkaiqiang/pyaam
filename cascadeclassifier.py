# -*- coding: utf-8 -*-
# CreateBy: kai
# CreateAt: 2021/1/11
import cv2
import matplotlib.pyplot as plt
import os.path

def detect(filename, cascade_file = "data/haarcascades/haarcascade_frontalface_default.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    i=0
    for (x, y, w, h) in faces:
        i+=1
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        temp=image[y:y+h,x:x+w,:]
        # cv2.imwrite('%s_%d.jpg'%(os.path.basename(filename).split('.')[0],i),temp)
    # cv2.imshow("AnimeFaceDetect", image)
    # cv2.waitKey(0)
    # cv2.imwrite("out.png", image)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

detect('pyaam/data/muct/jpg/i000qa-fn.jpg')