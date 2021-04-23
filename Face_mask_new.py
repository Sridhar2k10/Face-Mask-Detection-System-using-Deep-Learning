# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 09:14:29 2020

@author: Sridhar K
"""
import glob
import cv2
import xml.etree.ElementTree as ET

data = []
target = [] 
img_size = 100

img_paths = glob.glob('C:/Users/Sridhar K/Desktop/FACE_DEtect/images/*.*')
#img_paths.sort()
tree_paths = glob.glob('C:/Users/Sridhar K/Desktop/FACE_DEtect/annotations/*.*')
#tree_paths.sort()

for i in range(len(img_paths)):
    img_path = img_paths[i]
    img = cv2.imread(img_path)
    
    tree_path = tree_paths[i]
    tree = ET.parse(tree_path)
    objs = tree.findall('object')
    for obj in objs:
        label = obj.find('name').text
        box = obj.find('bndbox')
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        
        face = img[ymin:ymax, xmin:xmax]
        resized_face = cv2.resize(face,(img_size,img_size))
        data.append(resized_face)
        target.append(label)
    
import numpy as np

data = np.array(data)/255.0
data = np.reshape(data,(data.shape[0],img_size,img_size,3))
target = np.array(target)

# from keras.utils import np_utils

# new_target = np_utils.to_categorical(target)

target = target.reshape(len(target), 1)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
columntransformer = ColumnTransformer([("Label", OneHotEncoder(), [0])], remainder = 'passthrough')
new_target = columntransformer.fit_transform(target)

np.save('data',data)
np.save('target',new_target)

#target=new_target

data=np.load('C:/Users/Sridhar K/Desktop/FACE_DEtect/data.npy')
target=np.load('C:/Users/Sridhar K/Desktop/FACE_DEtect/target.npy')
#loading the save numpy arrays in the previous code

from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint

model=Sequential()

model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The first CNN layer followed by Relu and MaxPooling layers

model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#The second convolution layer followed by Relu and MaxPooling layers

model.add(Flatten())
model.add(Dropout(0.5))
#Flatten layer to stack the output convolutions from second convolution layer
model.add(Dense(128,activation='relu'))
#Dense layer of 128 neurons
model.add(Dense(3,activation='softmax'))
#The Final layer with two outputs for two categories

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])     

#from keras.utils.vis_utils import plot_model
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#visulization of cnn model

from sklearn.model_selection import train_test_split

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)  

checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history=model.fit(train_data,train_target,epochs=20,callbacks=[checkpoint],validation_split=0.2)

from matplotlib import pyplot as plt

plt.plot(history.history['loss'],'r',label='training loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'],'r',label='training accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.xlabel('# epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

print(model.evaluate(test_data,test_target))

from keras.models import load_model
import cv2
import numpy as np


prototxtPath = 'C:/Users/Sridhar K/Desktop/FACE_DEtect/deploy.prototxt'
weightsPath = 'C:/Users/Sridhar K/Desktop/FACE_DEtect/res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

model = load_model('C:/Users/Sridhar K/Desktop/FACE_DEtect/model-015.model')


def detect_face(img, faceNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = img.shape[:2]
	blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),(104.0, 177.0, 123.0))
	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	# initialize our list of faces, their corresponding locations
	locs = []
    # loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]
		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			locs.append((startX, startY, endX-startX, endY-startY))
	# return a tuple of the face locations
	return locs

source=cv2.VideoCapture(0)

labels_dict={2:'NO MASK',1:'MASK',0:'MASK WORN INCORECTLY'}
color_dict={2:(0,0,255),1:(0,255,0),0:(255,0,0)}
flag = 0

while(True):

    ret,img=source.read()
    #gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=detect_face(img, faceNet)
    
    if len(faces) == 0:
        if flag == 0:
            cv2.putText(img, 'No face detected', (img.shape[1]//2 - int(img.shape[1]*0.1875), img.shape[0]//2),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            flag = 1
        else:
            flag = 0
            
    else:
        flag=0
          
        for x,y,w,h in faces:
        
            #face_img=gray[y:y+h,x:x+w]
            face_img=img[y:y+h,x:x+w]
            resized=cv2.resize(face_img,(100,100))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,100,100,3))
            result=model.predict(reshaped)
    
            label=np.argmax(result,axis=1)[0]
          
            cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
            cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
            cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            
        
    cv2.imshow('LIVE',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break
        
cv2.destroyAllWindows()
source.release()