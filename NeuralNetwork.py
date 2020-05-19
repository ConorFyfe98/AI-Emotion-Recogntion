import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import matplotlib.pyplot as plt

#initalise classifiers
faceDetectionOne = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDetectionTwo = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDetectionThree = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDetectionFour = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

#Declare directories
TRAIN_DIR ='C:/Users/test/Desktop/EmotionApp/TrainData'
TEST_DIR ='C:/Users/test/Desktop/EmotionApp/TestData'

IMG_SIZE = 350
LR = 1e-3

MODEL_NAME = '6conv34epochs-{}-{}.model'.format(LR, '6conv-basic')

def label_img(img):
    #emotion label is before space in file name
    labelled_emotion = img.split(' ')[0]

    #return one hot arraybased on emotion label
    if labelled_emotion == 'surprise': return [1,0,0,0,0,0]
    elif labelled_emotion == 'sadness': return [0,1,0,0,0,0]
    elif labelled_emotion == 'joy': return [0,0,1,0,0,0]
    elif labelled_emotion == 'anger': return [0,0,0,1,0,0]
    elif labelled_emotion == 'disgust': return [0,0,0,0,1,0]
    elif labelled_emotion == 'fear': return [0,0,0,0,0,1]

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        #Get full path of image
        path = os.path.join(TRAIN_DIR,img)
        #Read in image, ensure grayscale, and resized
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    #Save training data to be used again
    np.save('train_data_emotions.npy', training_data)
    return training_data

def process_test_data():
        testing_data = []
        for img in tqdm(os.listdir(TEST_DIR)):
            path = os.path.join(TEST_DIR,img)
            #ID of the image
            img_num = img.split('.')[0]
            #Read image, ensure grayscale, and resized
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)


            faceOne = faceDetectionOne.detectMultiScale(img, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            faceTwo = faceDetectionTwo.detectMultiScale(img, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            faceThree = faceDetectionThree.detectMultiScale(img, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            faceFour = faceDetectionFour.detectMultiScale(img, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

            #Return first face found, if no face found return empty
            if len(faceOne) == 1:
                facePosition = faceOne
            elif len(faceTwo) == 1:
                facePosition = faceTwo
            elif len(faceThree) == 1:
                facePosition = faceThree
            elif len(faceFour) == 1:
                facePosition = faceFour
            else:
                facePosition = ""

            #get coordinates and size of rectangle containing face, Slice image to coordinates.
            for (x, y, w, h) in facePosition:
                img = img[y:y+h, x:x+w]
                try:
                    #Resize face so all images have same size
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                except:
                   pass 
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            testing_data.append([np.array(img), img_num])
        np.save('test_data_emotions.npy',testing_data)
        return testing_data

#Load train data if it already exists 
def create_data():
    if os.path.isfile('train_data_emotions.npy'):
        train_data = np.load('train_data_emotions.npy')
        print('loaded')
        return train_data
    else:
        train_data = create_train_data()
        print('new')
        return train_data

train_data = create_data()

#Reset TensorBoard graph
tf.reset_default_graph()

#Input layer
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

#Add convolution and pooling layer
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

#Fully connected and dropout layer
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

#Output layer with 6 classifications
convnet = fully_connected(convnet, 6, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

#If meta file for model exists, loaf model
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded')
    

#split traingin and testing data
train = train_data[:-5]
test = train_data[-5:]

#X is the feature sets of images, Y is the emotion label fo the image
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]


test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]


#Train the network
model.fit({'input': X}, {'targets': Y}, n_epoch=34, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)

#if dont have file doesnt exist
test_data = process_test_data()


fig = plt.figure(figsize=(10,10))

#use 29 images for testing
for num, data in enumerate(test_data[:29]):

    #img num is the id of the image, data is image
    img_num = data[1]
    img_data = data[0]

    #5 by 7 grid
    y = fig.add_subplot(5,7,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)

    #Use the model from memory, Break down list of predictions to receive probability
    model_out = model.predict([data])[0]
    surpriseNum = model.predict([data])[0,0]
    sadnessNum = model.predict([data])[0,1]
    joyNum = model.predict([data])[0,2]
    angerNum = model.predict([data])[0,3]
    disgustNum = model.predict([data])[0,4]
    fearNum = model.predict([data])[0,5]

    #Print results for each emotion and probability
    print("----------------------")
    print("Suprise: ",surpriseNum)
    print("Sadness: ",sadnessNum)
    print("joy: ",joyNum)
    print("Anger: ",angerNum)
    print("Disgust: ",disgustNum)
    print("Fear: ",fearNum)
    
    #Find highest probability and convert to percentage
    highest = 0
    if surpriseNum > highest:
        highest = surpriseNum
        surprisePercentage = (surpriseNum*100)
        str_label=('Surprise ' + str(surprisePercentage)[:2] + '%')
    if sadnessNum > highest:
        highest = sadnessNum
        sadnessPercentage = (sadnessNum*100)
        str_label = ('sadness ' + str(sadnessPercentage)[:2] + '%')
    if joyNum > highest:
        highest = joyNum
        joyPercentage = (joyNum*100)
        str_label = ('Joy ' + str(joyPercentage)[:2] + '%')
    if angerNum > highest:
        highest = angerNum
        angerPercentage = (angerNum*100)
        str_label = ('Anger ' + str(angerPercentage)[:2] + '%')
    if disgustNum > highest:
        highest = disgustNum
        disgustPercentage = (disgustNum*100)
        str_label = ('Disgust ' + str(disgustPercentage)[:2] + '%')
    if fearNum > highest:
        highest = fearNum
        fearPercentage = (fearNum*100)
        str_label = ('Fear ' + str(fearPercentage)[:2] + '%')

    #display image and label with emotion and prediction percentage
    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    #hide x and y axis
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()


#References
#Title: Classifying Cats vs Dogs with a Convolutional Neural Network on Kaggle
#Author: Harrison Kelly 
#Date: February 22nd 2017 
#Availability: https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/

