import cv2

def blur_face(frame, result, last_face_cords=None):
    """
        Blur faces in the frame. Returns the last frame coordinates, in case, nothing is detected in the next frame.
        :param frame: frame to blur faces
        :param result: result from the model

        returns: None
    """
    if last_face_cords:
        result = last_face_cords

    for i in range(len(result[0].boxes.cls)):
        x1, y1, x2, y2 = map(int, result[0].boxes.xyxy[i])
        # Get the face region
        face = frame[y1:y2, x1:x2]
        # Blur the face
        face = cv2.GaussianBlur(face, (21, 21), 30)
        # Put the blurred face back in the frame
        frame[y1:y2, x1:x2] = face
        # Return the current frame coordinates
    return result

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
    navigation_text = f'Controls: IOU: "i" = +, "u" = -], Confidence: ["c" = +, "d" = -], Pred. Framerate: ["p" = +, "o" = -], Model: ["1" = M1, "2" = M2], Dist. test: ["t" = on, "z" = off]'
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

def put_distance_line(frame, result, stats_dict: dict, distance_threshold=150, avg_mask_size=25):
    """
        Add distance line between bounding boxes
        :param frame: frame to add the distance line
        :param result: result from the model
        :param stats_dict: dictionary containing the stats
        :param distance_threshold: threshold for the distance
        :param avg_mask_box_size: average mask box size

        returns: None
    """
    
    boxes = []   # Set to store bounding box tuples (center, scale)
    for i in range(len(result[0].boxes.cls)):
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, result[0].boxes.xyxy[i])

        # Calculate the number of pixels per cm and the center of the bounding box
        box_width = (x2-x1) # Box width
        box_center_x = (x2-x1)/2 + x1  # x center coordinate
        box_center_y = (y2-y1)/2 + y1 # y center coordinate
        box_center_z = (stats_dict["f"] / box_width) * avg_mask_size  # z center coordinate 
        boxes.append((box_width, box_center_x, box_center_y, box_center_z))  # Add tuple to the list

    # Sort the boxes by scale    
    boxes_sort_scale = sorted(boxes, key=lambda x: x[0])    # Sort the boxes by scale
    for i in range(len(boxes_sort_scale)-1):

        # Calculate the distance between the bounding boxes
        pxl_distance_x = int(boxes_sort_scale[i+1][1]-boxes_sort_scale[i][1])  # X distance in pxl
        pxl_distance_y = int(boxes_sort_scale[i+1][2]-boxes_sort_scale[i][2])  # Y distance in pxl
        cm_distance_z = int(boxes_sort_scale[i+1][3]-boxes_sort_scale[i][3])  # Z distance in pxl
        pxl_per_cm_xy = int(boxes_sort_scale[i][0]) / avg_mask_size # Calculate the number of pixels per cm

        # Calculate euclidean distance
        distance_xy = int((pxl_distance_x**2 + pxl_distance_y**2)**0.5 / pxl_per_cm_xy)  # Calculate distance in cm
        distance_xyz = int((distance_xy**2 + (cm_distance_z)**2)**0.5)    # Calculate distance in cm


        # Get line start and end points
        pt1= (int(boxes_sort_scale[i][1]), int(boxes_sort_scale[i][2]))     # Get the first point
        pt2= (int(boxes_sort_scale[i+1][1]), int(boxes_sort_scale[i+1][2])) # Get the second point
        pt_middle = (int((pt1[0]+pt2[0])/2), int((pt1[1]+pt2[1])/2))    # Get the middle point

        # Check distance threshold
        colorcode = (0, 165, 0)
        if distance_xyz < distance_threshold:
            colorcode = (0, 0, 255)
             
        # Add line and text to the frame
        cv2.line(frame, pt1, pt2, colorcode, 2)
        cv2.putText(frame, f"XY: {distance_xy}cm", pt_middle, cv2.FONT_HERSHEY_SIMPLEX, 1, colorcode, 2)
        pt_middle = (pt_middle[0], pt_middle[1]+50)    # Get the middle point   
        cv2.putText(frame, f"XYZ: {distance_xyz}cm", (pt_middle), cv2.FONT_HERSHEY_SIMPLEX, 1, colorcode, 2)

def test_distance_line(frame, result, stats_dict: dict, distance_threshold=150, avg_mask_size=25):
    """
        Add distance line between bounding boxes
        :param frame: frame to add the distance line
        :param result: result from the model
        :param stats_dict: dictionary containing the stats
        :param distance_threshold: threshold for the distance
        :param avg_mask_box_size: average mask box size

        returns: None
    """
    
    boxes = []   # Set to store bounding box tuples (center, scale)
    for i in range(len(result[0].boxes.cls)):
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, result[0].boxes.xyxy[i])

        # Calculate the number of pixels per cm and the center of the bounding box
        box_width = (x2-x1) # Box width
        box_center_x = (x2-x1)/2 + x1  # x center coordinate
        box_center_y = (y2-y1)/2 + y1 # y center coordinate
        box_center_z = (stats_dict["f"] / box_width) * avg_mask_size  # z center coordinate 
        boxes.append((box_width, box_center_x, box_center_y, box_center_z))  # Add tuple to the list

    # Sort the boxes by scale    
    boxes_sort_scale = sorted(boxes, key=lambda x: x[0])    # Sort the boxes by scale
    for i in range(len(boxes_sort_scale)-1):

        # Calculate the distance between the bounding boxes
        pxl_distance_x = int(boxes_sort_scale[i+1][1]-boxes_sort_scale[i][1])  # X distance in pxl
        pxl_distance_y = int(boxes_sort_scale[i+1][2]-boxes_sort_scale[i][2])  # Y distance in pxl
        cm_distance_z = int(boxes_sort_scale[i+1][3]-boxes_sort_scale[i][3])  # Z distance in pxl
        pxl_per_cm_xy = int(boxes_sort_scale[i][0]) / avg_mask_size # Calculate the number of pixels per cm

        # Calculate euclidean distance
        distance_xy = int((pxl_distance_x**2 + pxl_distance_y**2)**0.5 / pxl_per_cm_xy)  # Calculate distance in cm
        distance_xyz = int((distance_xy**2 + (cm_distance_z)**2)**0.5)    # Calculate distance in cm      

    # Put red pint in middle of frame
    h_frame, w_frame, _ = frame.shape  # Get frame dimensions
    cv2.circle(frame, (int(w_frame/2), int(h_frame/2)), 5, (0, 0, 255), -1)  # Red circle

    try:
        # Put red point in the middle of the bounding boxes
        x, y = int(boxes_sort_scale[0][1]), int(boxes_sort_scale[0][2])
        pxl_per_cm_xy = int(boxes_sort_scale[0][0]) / avg_mask_size # Calculate the number of pixels per cm
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red circle

        # Calculate distance to the center
        dist = int(((x - w_frame/2)**2 + (y - h_frame/2)**2)**0.5 / pxl_per_cm_xy)  # Calculate distance to the center

        # Calculate z distance
        z = int(boxes_sort_scale[0][3])  # Get z distance

        # Calculate euclidean distance xyz
        distance_xyz = int((dist**2 + (z)**2)**0.5)    # Calculate distance in cm

        # Check distance threshold
        colorcode = (0, 165, 0)
        if distance_xyz < distance_threshold:
            colorcode = (0, 0, 255)

        # Draw line to the center
        cv2.line(frame, (x, y), (int(w_frame/2), int(h_frame/2)), colorcode, 2)  # Red line

        # Add text to the frame
        line_middle = (int((x+w_frame/2)/2+150), int((y+h_frame/2)/2))    # Get the middle point
        cv2.putText(frame, f"XY: {dist}cm", line_middle, cv2.FONT_HERSHEY_SIMPLEX, 1, colorcode, 2)
        line_middle = (int((x+w_frame/2)/2+150), int((y+h_frame/2)/2)+50)    # Get the middle point
        try:
            cv2.putText(frame, f"Z: {z}cm", line_middle, cv2.FONT_HERSHEY_SIMPLEX, 1, colorcode, 2)
            line_middle = (int((x+w_frame/2)/2+150), int((y+h_frame/2)/2)+100)    # Get the middle point
            cv2.putText(frame, f"XYZ: {distance_xyz}cm", line_middle, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except Exception as e:
            print(f"Exception: {e}")
            pass

    except Exception as e:
        print(f"Exception: {e}")
        pass

def wait_for_key(stats_dict: dict, model_name_dict: dict):
    """
        Wait for key press and adjust the stats accordingly. 
        :param stats_dict: dictionary containing the stats
        :param moderrorel_name_dict: dictionary containing the model names

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
        ord('p'): ("Inf. FPS", 1),
        ord('o'): ("Inf. FPS", -1),
        ord('t'): ("Dist. test", True),
        ord('z'): ("Dist. test", False),
        ord('b'): ("Blur", True),
        ord('n'): ("Blur", False),
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
        elif var == "Inf. FPS":
            pred_framerate = max(min_pred_framerate, min(max_pred_framerate, stats_dict["Inf. FPS"] + delta))
            stats_dict[var] = pred_framerate
        elif var == "Dist. test":
            stats_dict[var] = delta
        elif var == "Blur":
            stats_dict[var] = delta
        elif var == "Model":
            if delta == 1:
                stats_dict["Model"] = "YOLOv8n"
            elif delta == 2:
                stats_dict["Model"] = "YOLOv11n"
    return True