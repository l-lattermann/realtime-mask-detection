import cv2 as cv
import sys

# Create class video source
class VideoSource:

    # Constructor
    def __init__(self, src):
        self.src = src
        
        # Open video source
        try:
            self.cap = cv.VideoCapture(src)
        except:
            print("Error opening video source")
            sys.exit(1)

    # Read frame
    def read(self):
        return self.cap.read()

    # Release video source
    def release(self):
        self.cap.release()