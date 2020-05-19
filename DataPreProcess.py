import cv2
import glob

#initalise classifiers
faceDetectionOne = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDetectionTwo = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDetectionThree = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDetectionFour = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

#List of emotions to be detected
emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

def detect_faces(emotion):
    #Get a list of all the images in each emotion folder
    files = glob.glob("sorted_emotions\\%s\\*" %emotion) 
    filenumber = 0
    for f in files:

        #Read in image and convert to grayscale
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #Use all 4 classifiers to detect face
        print("Detecting Face...")
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
        print("Cropping to Face...")
        for (x, y, w, h) in facePosition:
            img = img[y:y+h, x:x+w]

            #Try write the image to the training data folder
            try:
                print("Saving image...")
                #Resize face so all images have same size
                preProcessedImg = cv2.resize(img, (350, 350))
                #Write image
                cv2.imwrite("TrainData\\%s (%s).jpg" %(emotion, filenumber), preProcessedImg)
            except:
               pass 
        filenumber += 1 #Increment image number
for emotion in emotions:
    detect_faces(emotion) #Call function

#References
#Title: Emotion Recognition With Python, OpenCV and a Face Dataset
#Author: palkab
#Date: Apr 01, 2016  
#Availability: http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/
