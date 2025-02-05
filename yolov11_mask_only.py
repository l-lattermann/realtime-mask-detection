from ultralytics import YOLO
import cv2 
from matplotlib import pyplot as plt
import time
import threading



# Put stats bar on frame
def put_stats_bar(frame, stats_dict: dict, bar_height=60,font_scale=0.5, font_thickness=1):
    """
        Add a stats bar at the bottom of the frame
        :param frame: frame to add the stats bar
        :param stats_dict: dictionary containing the stats
        :param update_interval: interval to update the stats
        :param frame_count: frame count
        :param bar_height: height of the stats bar
        :param font_scale: font scale of the text

        returns: None
    """

    # Add a rectangle at the bottom of the frame
    h, w, _ = frame.shape  # Get frame dimensions
    cv2.rectangle(frame, (0, h - bar_height), (w, h), (0, 0, 0), -1)  # Black rectangle

    stats_text = "" # Initialize stats text
    # Convert stats dict to text
    for stat_name, stat in stats_dict.items():
        if isinstance(stat, (float, int)):
            stats_text += f"{stat_name}: {stat:.2f}   "    # Add stats to the text, rounded to 2 decimal places
        else:
            stats_text += f"{stat_name}: {stat}   "    # Add stats to the text

    # Add stats text to the frame    
    text_color = (255, 255, 255)  # White text
    cv2.putText(frame, stats_text, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    # Add navigation text to the frame
    navigation_text = f'Controls: IOU: "i" = +, "u" = -], Confidence: ["c" = +, "d" = -], Pred. Framerate: ["p" = +, "o" = -], Model: ["1" = M1, "2" = M2]c'
    cv2.putText(frame, navigation_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

def put_bounding_boxes(frame, result, model, colorcode=(0,255,0), color_static=True):
    """
        Add bounding boxes to the frame
        :param frame: frame to add the bounding boxes
        :param result: result from the model
        :param model: model used for inference
        :param colorcode: color code for the bounding boxes
        :param color_static: flag to use static color

        returns: None
    """

    for i in range(len(result[0].boxes.cls)):
        x1, y1, x2, y2 = map(int, result[0].boxes.xyxy[i])
        clss, conf = result[0].boxes.cls[i], result[0].boxes.conf[i]
        text = f"{model.names[int(clss)]} ({conf:.2f})"

        if not color_static:
            if int(clss) == 0:
                colorcode = (255, 165, 0)
            elif int(clss) == 1:
                colorcode = (0, 0, 255)
            elif int(clss) == 2:
                colorcode = (0,255,0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colorcode, 2)  # Red for model 2
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colorcode, 2)

def put_distance_line(frame, result, distance_threshold=150, avg_mask_box_size=25):
    """
        Add distance line between bounding boxes
        :param frame: frame to add the distance line
        :param result: result from the model
        :param distance_threshold: threshold for the distance
        :param avg_mask_box_size: average mask box size

        returns: None
    """
    
    boxes = []   # Set to store bounding box tuples (center, scale)
    for i in range(len(result[0].boxes.cls)):
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, result[0].boxes.xyxy[i])

        # Calculate the number of pixels per cm and the center of the bounding box
        pxl_per_cm = (y2-y1)/avg_mask_box_size  # pixels per cm
        box_center_x = (x1+x2)/2  # x center coordinate
        box_center_y = (y1+y2)/2  # y center coordinate
        boxes.append((pxl_per_cm, box_center_x, box_center_y))  # Add tuple to the list

    # Sort the boxes by scale    
    boxes_sort_scale = sorted(boxes, key=lambda x: x[0])    # Sort the boxes by scale
    for i in range(len(boxes_sort_scale)-1):

        # Calculate the distance between the bounding boxes
        pxl_distance_x = int(boxes_sort_scale[i+1][1]-boxes_sort_scale[i][1])  # X distance
        pxl_distance_y = int(boxes_sort_scale[i+1][2]-boxes_sort_scale[i][2])  # Y distance
        avg_scale = int((boxes_sort_scale[i][0] + boxes_sort_scale[i+1][0]) / 2) # Calculate the average scale

        # Calculate euclidean distance
        distance = (pxl_distance_x**2 + pxl_distance_y**2)**0.5 / avg_scale  # Calculate the distance between the bounding boxes

        # Get line start and end points
        pt1= (int(boxes_sort_scale[i][1]), int(boxes_sort_scale[i][2]))     # Get the first point
        pt2= (int(boxes_sort_scale[i+1][1]), int(boxes_sort_scale[i+1][2])) # Get the second point
        pt_middle = (int((pt1[0]+pt2[0])/2), int((pt1[1]+pt2[1])/2))    # Get the middle point

        # Check distance threshold
        colorcode = (0, 165, 0)
        if distance < distance_threshold:
            colorcode = (0, 0, 255)

        # Add line and text to the frame
        cv2.line(frame, pt1, pt2, colorcode, 2)
        cv2.putText(frame, f"{distance:.2f} cm", pt_middle, cv2.FONT_HERSHEY_SIMPLEX, 1, colorcode, 2)

def wait_for_key(stats_dict: dict, model_name_dict: dict):
    """
        Wait for key press and adjust the stats accordingly. 
        :param stats_dict: dictionary containing the stats
        :param model_name_dict: dictionary containing the model names

        returns: False if 'q' is pressed, else True
    """
    # Check for key press
    key = cv2.waitKey(1) & 0xFF  # Capture key press once

    # Define adjustments
    adjustments = {
        ord('i'): ("IOU", 0.01),
        ord('u'): ("IOU", -0.01),
        ord('c'): ("Conf.", 0.01),
        ord('d'): ("Conf.", -0.01),
        ord('p'): ("Pred. Framerate", 1),
        ord('o'): ("Pred. Framerate", -1),
        ord('1'): ("Model", 1),
        ord('2'): ("Model", 2)
    }

    # Define adjustment ranges
    min_pred_framerate, max_pred_framerate = 1, 60  # Pred. Framerate range
    min_conf, max_conf = 0.1, 1.0  # Confidence range
    min_iou, max_iou = 0.1, 1.0  # NMS IoU range

    if key == ord('q'):  # Quit
        return False
    
    elif key in adjustments:
        var, delta = adjustments[key]
        
        # Apply adjustment with clamping
        if var == "IOU":
            iou = max(min_iou, min(max_iou, stats_dict["IOU"] + delta))
            stats_dict[var] = iou
        elif var == "Conf.":
            conf = max(min_conf, min(max_conf, stats_dict["Conf."] + delta))
            stats_dict[var] = conf
        elif var == "Pred. Framerate":
            pred_framerate = max(min_pred_framerate, min(max_pred_framerate, stats_dict["Pred. Framerate"] + delta))
            stats_dict[var] = pred_framerate
        elif var == "Model":
            if delta == 1:
                model = list(model_name_dict.keys())[0]
                stats_dict["Model"] = "YOLOv8n"
            elif delta == 2:
                model = list(model_name_dict.keys())[1]
                stats_dict["Model"] = "YOLOv11n"
    return True

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

def main():
    # Load models
    v8n_mask = YOLO("models/yolo/train_runs/yolov8n_mask_100epochs/train/weights/best.pt")
    v11n_mask = YOLO("models/yolo/train_runs/yolo11n_mask_100epochs/train/weights/best.pt")

    # Create model list
    model_dict = {"YOLOv8n":v8n_mask, "YOLOv11n": v11n_mask}

    # Initial values
    start_time = time.time()    # Start time
    frame_count = 0 # Frame count
    stats_dict = {} # Stats dictionary
    next_stats_update = {}  # Initialize next update
    stats_update_interval = 3  # Stats update interval

    # Add model variables to the stats dictionary
    stats_dict = {"Conf.": 0.7, "IOU": 0.5, "Pred. Framerate": 1, "Model": "YOLOv8n"}   # Initial values

    # Initialize the threaded frame fetcher
    fetcher = FrameFetcher(0)

    while True:
        # Fetch the latest frame
        frame_fetch_time = time.time()    # Start time
        ret, frame = fetcher.get_frame()
        if not ret: # If the frame is not available,
            break   # Break the loop
        frame_count += 1  # Increment frame count
        stats_dict["Frame fetch time"] = (time.time() - frame_fetch_time)*1000    # Add frame fetch time to stats

        # Reset frame count and start time every 100 frames
        if frame_count == 100:
            frame_count = 1
            start_time = time.time()

        # Define the model
        model_name = stats_dict["Model"]  # Get the model name
        model = model_dict[model_name]  # Get the model

        # Check every pred_framerate frames
        if frame_count % stats_dict["Pred. Framerate"] == 0:
            # Perform inference for person and mask models
            inf_time = time.time()    # Start time 
            result = model(frame, conf=stats_dict["Conf."], iou=stats_dict["IOU"], stream=True, verbose=False)
            result = list(result)   # Convert generator to list
            inf_time = (time.time() - inf_time)*1000    # End time
            stats_dict["Inference time"] = inf_time

        # Add bounding boxes to the frame
        box_time = time.time()    # Start time
        put_bounding_boxes(frame, result, model, color_static=False)
        stats_dict["Annotation time"]=(time.time() - box_time)*1000    # End time

        # Add distance line to the frame
        line_time = time.time()    # Start time
        put_distance_line(frame=frame, result=result, distance_threshold=150, avg_mask_box_size=25)
        stats_dict["Line time"]=(time.time() - line_time)*1000    # End time

        # Calculate FPS (frames per second)
        elapsed_time = time.time() - start_time  # Total time since start
        stats_dict["FPS"] = frame_count / elapsed_time  # Frames per second

        # Add stats bar to the frame
        if frame_count % stats_update_interval == 0:
            next_stats_update = stats_dict.copy()    # Update stats
        put_stats_bar(frame, stats_dict=next_stats_update)
        
        # Key commands
        menu_time = time.time()  # Start time
        if not wait_for_key(stats_dict, model_dict):  # Wait for key press
            break
        stats_dict["Menu time"] = (time.time() - menu_time) * 1000  # End time 
        
        # Display the frame
        cv2.imshow("Detections", frame) 

    fetcher.stop()  # Stop the frame fetcher
    cv2.destroyAllWindows() # Close all OpenCV windows




if __name__ == "__main__":
    main()