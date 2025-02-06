from ultralytics import YOLO
import cv2 
import time

from scripts import cam_calibration as cc
from scripts import cv2_functions as cv2f
from scripts.frame_fetcher import FrameFetcher

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
    stats_dict = {"Conf.": 0.7, "IOU": 0.5, "Inf. FPS": 1, "f": 100,"Model": "YOLOv8n", "Dist. test": False, "Blur": False}   # Initial values

    # Initialize the threaded frame fetcher
    fetcher = FrameFetcher("data_sets/video_data/3205619-hd_1920_1080_25fps.mp4")

    # Calibrate the camera
    avg_mask_size = 18 # Define average mask size
    
    # Initialize the timer
    tm = cv2.TickMeter()
    frame_time = 20 # Frame time in milliseconds

    while True:
        # Start the timer
        tm.start()

        # Fetch the latest frame
        frame_fetch_time = time.time()    # Start time
        ret, frame = fetcher.get_frame()
        if not ret:
            print("No frame")
            continue  # Skip if no frame is available
        print("Frame fetched")
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
        if frame_count % stats_dict["Inf. FPS"] == 0:
            # Perform inference for person and mask models
            inf_time = time.time()    # Start time 
            result = model(frame, conf=stats_dict["Conf."], iou=stats_dict["IOU"], stream=True, verbose=False)
            result = list(result)   # Convert generator to list
            stats_dict["Inf. time"] = (time.time() - inf_time)*1000    # End time

        # Add bounding boxes to the frame
        box_time = time.time()    # Start time
        cv2f.put_bounding_boxes(frame, result, model, color_static=False)
        stats_dict["Box time"]=(time.time() - box_time)*1000    # End time

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
        if not cv2f.wait_for_key(stats_dict):  # Wait for key press
            break
        stats_dict["Menu time"] = (time.time() - menu_time) * 1000  # End time 

        # Optional: Test distance function
        if stats_dict["Dist. test"]:
            cv2f.test_distance_line(frame, result, stats_dict, avg_mask_size)

        # Blur the faces
        if stats_dict["Blur"]:
            # Check if faces are detected
            if len(result[0].boxes.cls) > 0:
                # Save last face coordinates
                last_face_cords = result
                cv2f.blur_face(frame, result)
            # If no faces are detected, blur at last face coordinates
            elif len(result[0].boxes.cls) == 0 and "last_face_cords" in locals():
                cv2f.blur_face(frame, last_face_cords)
        
        # Display the frame
        cv2.imshow("Detections", frame)
        tm.stop()
        elapsed_tick_time = tm.getTimeMilli()
        tm.reset()
        sleep_time = max(0, (frame_time - elapsed_tick_time))
        #time.sleep(sleep_time/1000) 
        print(f"Frame time: {frame_time} ms, Elapsed time: {elapsed_tick_time} s, Sleep time: {sleep_time} s")
        print(f"Framecount: {frame_count}")

    fetcher.stop()  # Stop the frame fetcher
    cv2.destroyAllWindows() # Close all OpenCV windows

if __name__ == "__main__":
    main()