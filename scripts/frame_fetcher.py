import threading
from queue import Queue
import cv2

class FrameFetcher:
    """
    Class to fetch frames in a separate thread.
    Uses a queue to store frames and prevent blocking.
    """
    def __init__(self, src=0, queue_size=10):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.ret, self.frame = self.cap.read()
        self.queue = Queue(maxsize=queue_size)
        self.stopped = False
        self.lock = threading.Lock()  # Thread safety for `self.ret` and `self.frame`

        # Start the background frame fetching thread
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        """Continuously fetch frames and store in queue."""
        while not self.stopped:
            cv2.waitKey(2)
            if self.queue.empty():  
                ret, frame = self.cap.read()
                print("cap read")

                if not ret:
                    print("no ret")
                    continue 

                with self.lock:
                    self.ret, self.frame = ret, frame 

                self.queue.put((ret, frame))  
            
            

    def get_frame(self):
        """Retrieve the latest available frame from the queue."""
        with self.lock:
            if not self.queue.empty():
                return self.queue.get()
            return self.ret, self.frame  # Return last valid frame if queue is empty

    def stop(self):
        """Stop the background thread and release the camera."""
        self.stopped = True
        self.thread.join()
        self.cap.release()