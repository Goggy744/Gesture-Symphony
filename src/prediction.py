import cv2
from keras.models import load_model
import numpy as np
import mediapipe as mp

  
def setup():
    """ Description
    Function that setup the module mediapipe and load the model from train.py
    :rtype: a tuple that contains hands object, drawing tools from mp and the model to make prediction
    """
    
    #Get the mediapipe solutions for hand recognition
    handsSolutions = mp.solutions.hands
    #Get the mediapipe drawing utils
    mp_drawing = mp.solutions.drawing_utils
    #Creating a hands object that can handle one hands maximum.
    handsObject = handsSolutions.Hands(max_num_hands=MAX_NUM_HANDS, min_detection_confidence=MIN_DETECTION_CONFIDENCE)
    #Load the Convolutional neural network with weight and bias saved
    convNN = load_model('handGestureModel.h5')
    
    return handsSolutions, handsObject, mp_drawing, convNN

def frameFormating(camera):
  """ Description
  Function that the videoCapture frames and format them
  :rtype: the formatted frame
  """
  #Get the state and the frame
  _, frame = camera.read()
  #Resize the frame using x and y variable defined beforehand
  frame = cv2.resize(frame, (WIDTH,HEIGTH))
  #Flip the image
  frame = cv2.flip(frame, 1)
  #Return the frame
  return frame

def detectHand(frame):
  """ Description
  Function that detect hand on a frame
  :type frame: a OpenCV frame
  :param frame: A single image from a video Capture
  :rtype: return a object from mediapipe that contains the detected hand
  """
  #Change the color from bgr to rgb
  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  #Get the hand inside the frame
  handDetectionData = handsObject.process(frame)
  #Change back the color from rgb to bgr
  cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
  
  #Return the data
  return handDetectionData

def getInput(handDetectionData):
  """ Description
  Function that return a valid input that can be feed into a model
  :type handDetectionData: mediapipe object that contains information on detected hand
  :param handDetectionData: Informations on the detected hands
  :rtype: a list
  """
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
  
  #Return the input
  return input
  
def getPrediction(input):
  """ Description
  Function that return the prediction of a model based on the input
  :type input: a list
  :param input: The input that can be feed into the model
  :rtype: a string
  """
  #get the prediction from the model after feeding the input
  prediction = convNN.predict(input)
  #get the labelID of the prediction
  labelID = np.argmax(prediction)
  #Get the labelName based on the classId
  labelName = labels[labelID]
  #return the prediction
  return labelName

if __name__ == "__main__":
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
  
    
  #Setup the module mediapipe and load the model
  handsSolutions, handsObject, mpDrawing, convNN = setup()

  #Define a webcam which is a VideoCapture instance
  webcam = cv2.VideoCapture(0)

  #Loop while the webcam is active
  while webcam.isOpened():
    #Get the formated frame from the webcam capture
    formatedFrame = frameFormating(webcam)

    #Search for a hand in the formated frame
    handDetectionData = detectHand(formatedFrame)
    
    #Check if there is hand detected on the videoCapture
    if handDetectionData.multi_hand_landmarks:
      #Get the input from the detected hand
      input = getInput(handDetectionData)
      #Draw the landmarks on the detected hand
      mpDrawing.draw_landmarks(formatedFrame,
                                handDetectionData.multi_hand_landmarks[0],
                                connections=handsSolutions.HAND_CONNECTIONS)
      #Get the prediction
      labelName = getPrediction(input)
      
      #Display the className on the top left of the video capture
      cv2.putText(formatedFrame, labelName, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0,0,255), 2, cv2.LINE_AA)
        
    #Name the window   
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    #Display the camera capture
    cv2.imshow('window', formatedFrame)
    
    #Check if we press the key 'q' so that we can exit the video capture
    if cv2.waitKey(5) & 0xFF == ord('q'):
      break

  #Stop the webcam capture
  webcam.release()
  cv2.destroyAllWindows()