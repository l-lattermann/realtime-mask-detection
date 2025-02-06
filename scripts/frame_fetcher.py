import threading
import cv2

def t():
    print("Hello")

class FrameFetcher:
    """
        Class to fetch frames in a separate thread
        :param src: source of the video feed
    """
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()  # Thread safety

        # Start the background frame fetching thread
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        """Continuously fetch frames in a separate thread"""
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:  # Ensure thread safety
                    self.ret, self.frame = ret, frame

    def get_frame(self):
        """Get the latest available frame"""
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        """Stop the background thread and release the camera"""
        self.stopped = True
        self.thread.join()
        self.cap.release()
