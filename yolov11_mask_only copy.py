from ultralytics import YOLO
import cv2 
import time

from scripts import cam_calibration as cc
from scripts import cv2_functions as cv2f
from scripts import frame_fetcher as ff


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
    stats_dict = {"Conf.": 0.7, "IOU": 0.5, "Pred. Framerate": 1, "f camera in pxl": 100,"Model": "YOLOv8n", "Dist. test": False}   # Initial values

    # Initialize the threaded frame fetcher
    fetcher = ff.FrameFetcher(0)

    # Calibrate the camera
    avg_mask_size = 18 # Define average mask size
    cc.calibrate_cam(avg_mask_size, fetcher, model_dict["YOLOv8n"], stats_dict)

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
            stats_dict["Inference time"] = (time.time() - inf_time)*1000    # End time

        # Add bounding boxes to the frame
        box_time = time.time()    # Start time
        cv2f.put_bounding_boxes(frame, result, model, color_static=False)
        stats_dict["Annotation time"]=(time.time() - box_time)*1000    # End time

        # Add distance line to the frame
        line_time = time.time()    # Start time
        cv2f.put_distance_line(frame=frame, result=result, stats_dict=stats_dict, distance_threshold=150, avg_mask_size=avg_mask_size)
        stats_dict["Line time"]=(time.time() - line_time)*1000    # End time

        # Calculate FPS (frames per second)
        elapsed_time = time.time() - start_time  # Total time since start
        stats_dict["FPS"] = frame_count / elapsed_time  # Frames per second

        # Add stats bar to the frame
        if frame_count % stats_update_interval == 0:
            next_stats_update = stats_dict.copy()    # Update stats
        cv2f.put_stats_bar(frame, stats_dict=next_stats_update)
        
        # Key commands
        menu_time = time.time()  # Start time
        if not cv2f.wait_for_key(stats_dict, model_dict):  # Wait for key press
            break
        stats_dict["Menu time"] = (time.time() - menu_time) * 1000  # End time 

        # Optional: Test distance function
        if stats_dict["Dist. test"]:
            print("Results: ", len(result[0].boxes.cls))
            cv2f.test_distance_line(frame, result, stats_dict, avg_mask_size)
        
        # Display the frame
        cv2.imshow("Detections", frame) 

    fetcher.stop()  # Stop the frame fetcher
    cv2.destroyAllWindows() # Close all OpenCV windows

if __name__ == "__main__":
    main()