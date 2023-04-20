import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np

#Define constant variable
WIDTH = 1920
HEIGTH = 1080
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7

#Define the a dictionary that represent the label of each hand signs
labels = {0:'okay',
           1:'stop',
           2:'thumps up',
           3:'thumps down',
           4:'one',
           5:'three'}

#Get the mediapipe solutions for hand recognition
mp_hands = mp.solutions.hands
#Get the mediapipe drawing utils
mp_drawing = mp.solutions.drawing_utils
#Creating a hands object that can handle one hands maximum.
hands = mp_hands.Hands(max_num_hands=MAX_NUM_HANDS, min_detection_confidence=MIN_DETECTION_CONFIDENCE)
#Load the Convolutional neural network with weight and bias saved
convNN = load_model('handGestureModel.h5')


#Define a webcam which is a VideoCapture instance
webcam = cv2.VideoCapture(0)

#Loop while the webcam is active
while webcam.isOpened():
  #Get the state and the frame
  state, frame = webcam.read()
  #Resize the frame using x and y variable defined beforehand
  frame = cv2.resize(frame, (WIDTH,HEIGTH))
  #Flip the image
  frame = cv2.flip(frame, 1)
  
  #Change the color from bgr to rgb
  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  #Get the hand inside the frame
  handDetectionData = hands.process(frame)
  #Change back the color from rgb to bgr
  cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

  #Initialize label name to 'None'
  labelName = "None"
  
  #Check if there is hand detected on the videoCapture
  if handDetectionData.multi_hand_landmarks:
    #Define a list that will be feed to the model
    input = [[]]
    #Loop through each hand detected
    for hand in handDetectionData.multi_hand_landmarks:
      #Get the points that segment the hand
      points = hand.landmark
      #Loop through the 23 points
      for point in points:
        #Add the x and y value multiply by the screen width and height to the input
        input[0].append([point.x*WIDTH])
        input[0].append([point.y*HEIGTH])

      #Draw the landmarks on the detected hand
      mp_drawing.draw_landmarks(frame, hand, connections=mp_hands.HAND_CONNECTIONS)

      #get the prediction from the model after feeding the input
      prediction = convNN.predict(input)
      #get the labelID of the prediction
      labelID = np.argmax(prediction)
      #Get the labelName based on the classId
      labelName = labels[labelID]
      #Display the className on the top left of the video capture
      cv2.putText(frame, labelName, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0,0,255), 2, cv2.LINE_AA)
      
  #Name the window   
  cv2.namedWindow("window", cv2.WINDOW_NORMAL)
  #Display the camera capture
  cv2.imshow('window', frame)
  
  #Check if we press the key 'q' so that we can exit the video capture
  if cv2.waitKey(5) & 0xFF == ord('q'):
    break

#Stop the webcam capture
webcam.release()
cv2.destroyAllWindows()