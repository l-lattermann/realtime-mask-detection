import cv2


def calibrate_cam(avg_mask_size, fetcher, model, stats_dict: dict):
    """
        Calibrate the camera focal length
        :param stats_dict: dictionary containing the stats

        returns: None
    """
    # Initialize focal length
    f = 0

    while True:
        # Fetch the latest frame
        ret, frame = fetcher.get_frame()
        if not ret: # If the frame is not available,
            break   # Break the loop

        # Mirrow the frame vertically
        frame = cv2.flip(frame, 1)

        # Put center point on the frame
        h_frame, w_frame, _ = frame.shape  # Get frame dimensions
        cv2.circle(frame, (int(w_frame/2), int(h_frame/2)), 5, (0, 0, 255), -1)  # Red circle
        cv2.rectangle(frame, (int(w_frame/2)-50, int(h_frame/2)-50), (int(w_frame/2)+50, int(h_frame/2)+50), (0, 0, 0), 2)  # Black rectangle

        # Add stats text to the frame 
        cv2.putText(frame, "Camera calibration:", (10, h_frame - 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        # Add navigation text to the frame
        calibration_text = f'Stand 100 cm away from the camera and put your nose on the red dot.'
        cv2.putText(frame, calibration_text, (10, h_frame - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'k' to calibrate the camera", (10, h_frame - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF  # Capture key press once
        if key == ord('k'):  # Calibrate the camera
            # Detect mask
            result = model(frame, conf=0.7, iou=0.8, stream=True, verbose=False)  # Perform inference
            result = list(result)   # Convert generator to list

            # Select detection closest to the center
            box_center_list =[]
            for i in range(len(result[0].boxes.cls)):
                x1, y1, x2, y2 = map(int, result[0].boxes.xyxy[i])
                box_center_x = (x1+x2)/2  # x center coordinate
                box_center_y = (y1+y2)/2  # y center coordinate
                box_width = x2 - x1  # Box width
                dist = (box_center_x - w_frame/2)**2 + (box_center_y - h_frame/2)**2  # Calculate distance to the center
                box_center_list.append((dist, i, box_width))  # Add tuple to the list
            
            # Get the index of the closest detection
            calibration_distance = 100  # Calibration distance in cm
            if box_center_list:
                i = min(box_center_list)[1]
                f = box_center_list[i][2] * calibration_distance / avg_mask_size  # Calculate the focal length
                stats_dict["f camera in pxl"] = f  # Update the focal length
                break   # Break the loop
        
        # Show the frame
        cv2.imshow("Calibration", frame)
    
    cv2.destroyAllWindows() # Close all OpenCV windows