from prediction import *
import pygame
import cv2

#Initialize the pygame module
pygame.init()

#Define constant variable
WIDTH = 1920
HEIGHT = 1080
SCREEN_SIZE = (WIDTH, HEIGHT)
SCREEN = pygame.display.set_mode(SCREEN_SIZE)
FPS = 144

#Define a python class for the usage of the camera
class Camera:
  
  def __init__(self, ) -> None:
    self.c = cv2.VideoCapture(0)
    self.state = True
    self.currentFrame = None
  
  def setCurrentFrame(self):
    """ Description
    Function that set the current frame

    :rtype: None
    """
    #Set the current frame using the imported function from prediction.py
    self.currentFrame = frameFormating(self.c)
    
  def stopVideoCapture(self):
    self.state = False
    
    
  def capture(self):
    """ Description
    Function that activates the video capture
  
    :rtype: None
    """
    #Loop until the state is set to false
    while self.state:
      #Call the method setCurrentFrame
      self.setCurrentFrame()
      
      #Check if we press the key 'q' so that we can exit the video capture
      if cv2.waitKey(5) & 0xFF == ord('q'):
        self.stopVideoCapture()
    
    self.c.release()
    cv2.destroyAllWindows()
    
    
    
  
#Define a python class for the application
class Application:
  
  def __init__(self, screen, fps):
    """ Description
    Constructor of the class Application
    
    :type screen: a Surface object of the pygame library
    :param screen: The screen of the application
  
    :type fps: an int
    :param fps: the amount of frame per second that the application will display
    
    :rtype:None
    """
    self.screen = screen
    self.fps = fps
    self.clock = pygame.time.Clock()
    self.state = True
    
  def shutdown(self):
    """ Description
    Function that change the state of the application
    
    :rtype:None
    """
    #Change the state to false
    self.state = False    
  
  
  def run(self):
    """ Description
    Function that runs the application
    
    :rtype: None
    """
    #Loop until the state is set to false
    while self.state:
      #Enable the framerate to be equal to the value of fps
      self.clock.tick(self.fps)
      
      #Loop through the app event
      for event in pygame.event.get():
        #Check if the event is the QUIT event
        if event.type == pygame.QUIT:
          #Call the shutdown method
          self.shutdown()
    
    #Close the application
    pygame.quit()
    