import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19
from gaborfilter import create_gaborfilter,apply_filter
from wdl import weber
def preprocess(img):
    cv.imwrite("temp.jpg",img)
    gfilters = create_gaborfilter()
    myimage = cv.imread("temp.jpg")
    image_g = apply_filter(myimage, gfilters)
    cv.imwrite("temp2.jpg",image_g)
    grayscale_image = cv.imread('temp2.jpg', 0)
    descriptor = weber(grayscale_image)
    cv.imwrite("temp3.jpg",descriptor)
    myimage = cv.imread("temp3.jpg")
    return myimage
 
data=ImageDataGenerator(validation_split=0.2,preprocessing_function=preprocess)

training=data.flow_from_directory("dataset/train",
                                  class_mode = "categorical",
                                  target_size = (224,224),
                                  color_mode="rgb",
                                  batch_size=32,
                                  shuffle = True,
                                  subset='training',
                                  seed = 32)
validation=data.flow_from_directory("dataset/train",
                                  class_mode = "categorical",
                                  target_size = (224,224),
                                  color_mode="rgb",
                                 batch_size=32,
                                  shuffle = True,
                                  subset='validation',
                                  seed = 32)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Conv2D,MaxPooling2D
model=Sequential()

model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dense(200,activation="relu"))
model.add(Dense(2,activation="softmax"))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history2=model.fit(training,validation_data=validation,epochs=5, steps_per_epoch=len(training), validation_steps=len(validation) )
r=model.evaluate(validation)
def plots(x,y,title,x_label,y_label,xlimit,ylimit=[0,2]):
    plt.plot(x)
    plt.plot(y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(xlimit[0],xlimit[1])
    plt.ylim(ylimit[0],ylimit[1])
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


plots(history2.history['accuracy'],history2.history['val_accuracy'],'accuracy of the model','Number of epochs','accuracy',[0,4],[.7,1])
plots(history2.history['loss'],history2.history['val_loss'],'loss of the model','Number of epochs','loss',[0,4],[0,1])
