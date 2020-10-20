# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 01:36:08 2020

@author: Ankit
"""


import cv2
import numpy as np

cap=cv2.VideoCapture(0)
classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

whT =320


## Model Files
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

confThreshold=0.5
nmsThreshold=0.3
def findObjects(outputs,img):
    hT, wT, cT = img.shape     ###height width and channnel
    bbox = []         ## have deatils of bounding box
    classIds = []     ## have info about classid
    confs = []         ##confidence values
    
    for output in outputs:     ## becoz we have 3 output
        for det in output:     ##we go to each output and we call each of the box as a detection
            scores = det[5:]   ## to find the value of highest probability i.e, which class have highest probability ,
                                   ## so we remove 1st 5 element
            classId = np.argmax(scores)    ## find index of max values
            confidence = scores[classId]
            
            #filtering objects
            if confidence > confThreshold:              ##if confidence is greater than 50 percent
                                                        ##we save the image
                ## saving width and height and x and y
                w,h = int(det[2]*wT) , int(det[3]*hT)      
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    
    ## tell which bounding box to keep
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]     #extract x y w h
        print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


while True:
    success, img = cap.read()
    
    ##converting image to blob
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    
    ## get all layer names
    layersNames = net.getLayerNames()
    print(layersNames)
    
    ##get the index of ouput layers
    print(net.getUnconnectedOutLayers())
    
    ##now use the index and refer it back to the layer name and extract the names from these indexes
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    print(outputNames)
    
    ## now send the image as forward pass and find the output of these layers which came before
    outputs = net.forward(outputNames)
    print(len(outputs))  ## to know how many different output we are getting
    print(type(outputs))  ## to know the type of output
    print(type(outputs))  ## now we know output is list now let's check what's inside it...and we get numpy array
    
    ## let's get shape of all 3 channels
    print(outputs[0].shape)    ##let's say (xxx,85) xxx is number of bounding boxes 
                               #80 is probability of prediction of each class and + 5 is the center x,
                                 #center y , width ,height and confidence that there is a object in the box
    print(outputs[0][0])  #let's visualize whaqt is told up in the comment
                          #that's 1st output 1st element
    
    print(outputs[1].shape)
    print(outputs[2].shape)

    
    
    ###calling above function
    
    findObjects(outputs,img)
    
    cv2.imshow('Image', img)
    cv2.waitKey(1)