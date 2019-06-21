from imutils import paths
import cv2
import time
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from cv2 import *
#cascPath = "/home/shikha/Desktop/opencv-master/data/haarcascades/haarcascade_frontalface_alt.xml"
#faceCascade = cv2.CascadeClassifier("/home/shikha/Desktop/opencv-master/data/haarcascades/haarcascade_frontalface_alt.xml")
#smileCascade=cv2.CascadeClassifier("/home/shikha/Desktop/opencv-master/data/haarcascades/haarcascade_smile.xml")
#lowerbodyCascade=cv2.CascadeClassifier("/home/shikha/Desktop/opencv-master/data/haarcascades/haarcascade_lowerbody.xml")
MODE="COCO"
if MODE is "COCO":
    protoFile = "line_vec.prototxt"   #/home/shikha/
    weightsFile = "/home/shikha/Downloads/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
files=glob.glob("/home/shikha/port/*.PNG")
ap = argparse.ArgumentParser()
args=vars(ap.parse_args())
img1 = cv2.imread("/home/shikha/stylo.jpg",1)
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1Copy=np.copy(img1)
img1Width=img1.shape[1]
img1Height=img1.shape[0]
threshold=0.1
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
files=glob.glob("/home/shikha/port/*.PNG")
detector = cv2.CascadeClassifier("/home/shikha/Desktop/opencv-master/data/haarcascades/haarcascade_frontalface_alt.xml")
detector1 = cv2.CascadeClassifier("/home/shikha/Desktop/opencv-master/data/haarcascades/haarcascade_smile.xml")
detector2=cv2.CascadeClassifier("/home/shikha/Desktop/opencv-master/data/haarcascades/haarcascade_lowerbody.xml")
scale_factor = 1.2
min_neighbours = 5
min_size = (30,30)
faces_coord = detector.detectMultiScale(gray, scaleFactor= scale_factor, minNeighbors = min_neighbours, minSize = min_size)
min_neighbours = 20
smile_coord = detector1.detectMultiScale(gray, scaleFactor= scale_factor, minNeighbors = min_neighbours)
scale_factor = 1.1
min_neighbours=2
lowerbody_coord=detector2.detectMultiScale(gray, scaleFactor= scale_factor, minNeighbors = min_neighbours)
s_c = smile_coord[[0]]
print( "Type: " + str(type(faces_coord)))
print(faces_coord)
print("Length: "+ str(len(faces_coord)))
print( "Type: " + str(type(smile_coord)))
print(smile_coord)
print("Length: "+ str(len(smile_coord)))
print( "Type: " + str(type(lowerbody_coord)))
print(lowerbody_coord)
print("Length: "+ str(len(lowerbody_coord)))
for (x, y, w, h) in faces_coord:
    cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 2)
for (px, py, pw, ph) in s_c:
    cv2.rectangle(img1, (px, py),(px+pw, py+ph),(0,125,0), 2)
for (jx, jy, jw, jh) in lowerbody_coord:
    cv2.rectangle(img1, (jx, jy),(jx+jw, jy+jh),(0,0,255), 2)

def pose_detection():
                   X_data=[]
files=glob.glob("/home/shikha/port/*.PNG")
for myFile in files:
        print(files)
        image=cv2.imread(files)
        X_data.append(myFile)
        print(X_data)
        

# define dnn network
img1=cv2.imread("stylo.jpg")
t=time.time()
inWidth=250
inHeight=250
inpblob=cv2.dnn.blobFromImage(img1,1.0/255,(inWidth,inHeight),(0,0,0),swapRB=False)
print("First Blob: {}".format(inpblob.shape))
net.setInput(inpblob)
print("time taken by network : {:.3f}".format(time.time() - t))
output=net.forward()
H=output.shape[2]
W=output.shape[3]
print(output.shape)

#store detected keypoints of image
points = []
for i in range (18):
    probMap = output[0,i,:,:]
    minval, prob, minloc, point = cv2.minMaxLoc(probMap)
    x=(img1Width*point[0])/W
    y=(img1Width*point[1])/H
    threshold=0.1

if (prob > threshold):
    cv2.circle(img1, (int (x),int(y)),8,(0,255,255),thickness=-1,lineType=cv2.FILLED)
    points.append((int (x),int(y)))
else:
    points.append(None)
img1=cv2.imread("stylo.jpg",1)
cv2.imshow("stylo.jpg", img1)
imwrite("/home/shikha/port/thumbnail_2.png", img1)
waitKey(0)

print ("Also working")

