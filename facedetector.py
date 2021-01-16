import cv2
from random import randrange

#Load some pre-trained data  on face frontals from opencv
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#To capture video from webcam
webcam = cv2.VideoCapture(0)

#Iterate forever over frames
while True:
    #Read the current frame
    successful_frame_read, frame=webcam.read()

    #Convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #Draw rectangles around the faces
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame , (x,y) , (x+w,y+h) , (randrange(256),randrange(256),randrange(256) , 20))

    cv2.imshow("Face Detector" , frame)
    key=cv2.waitKey(1)

    # Stop if Q key is pressed
    if key==81 or key==113:
        break

#Release VideoCamera Object
webcam.release()


#Choose an image to detect faces in
#img = cv2.imread("sk.jpg")
#Convert to grayscale
#grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
"""
#Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#Draw rectangles around the faces
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img , (x,y) , (x+w,y+h) , (0,255,0) , 2)

"""


#Display the face with images
#cv2.imshow("Face Detector" , img)
#cv2.waitKey()