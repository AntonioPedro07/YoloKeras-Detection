#!/usr/bin/python3

"""This sample shows how to use openCV on the depthdata we get back from either a camera or an rrf file.
The Camera's lens parameters are optionally used to remove the lens distortion and then the image is displayed using openCV windows.
Press 'd' on the keyboard to toggle the distortion while a window is selected. Press esc to exit.

Additionally this sample implements the YOLO v4 network for object detection. We convert the image to rgb and then feed this image
into the network. Then we draw bounding boxes around the found object.
"""

import argparse
import queue
import sys
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2
import logging

# insert the path to your Royale installation here:
# note that you need to use \\ or / instead of \ on Windows
ROYALE_DIR = "C:/Program Files/royale/5.4.0.2112/python"
sys.path.append(ROYALE_DIR)

import roypy
from roypy_sample_utils import CameraOpener, add_camera_opener_options
from roypy_platform_utils import PlatformHelper
from keras.models import load_model

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')

# Load the YOLO network classes and colors
CLASSES = None
with open("coco.names", 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load YOLO models
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load keras network classes
CLASSES_KERAS = None
with open("labels.txt", "r") as f:
    CLASSES_KERAS = [line.strip() for line in f.readlines()]

# Load Keras models
keras_model = load_model("keras_model.h5", compile = False)
class_name_id = {name: idx for idx, name in enumerate(CLASSES_KERAS)}

# Function to retrieve the names of the output layers
def get_output_layers(net):

    # Get the names of the output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()] 

    # Return the names of the output layers
    return output_layers

# Function to get the distance to the object
def draw_prediction(img, class_id, confidence, x1, y1, x2, y2, distance, source_model):

    # Convert coordenates to whole
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    if source_model == 'keras': 
        color = COLORS[class_id]
        label = f"{CLASSES_KERAS[class_id]}: {confidence:.2f} - Distance: {distance:.2f}m"
    elif source_model == 'yolo':
        color = COLORS[class_id]
        label = f"{CLASSES[class_id]}: {confidence:.2f}% - Distance: {distance:.2f} m"

    # Draw a bounding box.
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Function to emit Alert of the object to the terminal
def emitAlert(distance, object_name):

    """
    Emits an alert based on the distance of the detected object.
    :param distance: The distance to the object in meters.
    :param object_name: The name/class of the detected object.
    """
    INFO_DISTANCE = 1.0  # Distance to inform about the object's presence
    DANGER_DISTANCE_CM = 50.0  # Distance considered as dangerous
    TOLERANCE_CM = 5.0 # Tolerance for considering the distance as 1 meter

    # Convert distance to centimeters to check Danger_distance_cm and Tolerance_cm
    distance_cm = distance * 100

    if distance_cm <= DANGER_DISTANCE_CM:
        print(f"Immediate danger! Object '{object_name}' too close: {distance_cm:.2f}cm")  # Print the distance cm
        return True
    elif distance <= INFO_DISTANCE:
        print(f"Object detected: '{object_name}', at {distance:.2f}m distance.") # Print the distance m
        return True
    elif (INFO_DISTANCE * 100 - TOLERANCE_CM) <= distance_cm <= (INFO_DISTANCE * 100 + TOLERANCE_CM):
        print(f"Object detected within tolerance: '{object_name}', at {distance_cm:.2f}cm distance.") # Print the distance cm with tolerance
        return True
    return False

# Function to process depth image
def processDepthImage(depth_image):

    # Normalize the depth image to fall between the range 0 - 1
    depthImgNormalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the normalized depth image to a 8-bit integer
    depthImg_uint8 = np.uint8(depthImgNormalized)

    # Convert the depth image to a 3-channel image
    depthImgGray = cv2.cvtColor(depthImg_uint8, cv2.COLOR_GRAY2BGR)

    return depthImgGray

# Function to nms
def nonMaxSuppression(boxes, scores, overlapThresh):

    # If there are no boxes, return an empty list
    if len(boxes) == 0:
        return [], []

    # If the bounding boxes are integers, convert them to floats -- this
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = [] # List of picked indexes

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # Compute the area of the bounding boxes and sort the bounding
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # Keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]] # Compute the ratio of overlap

        # Delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int"), scores[pick]

# Function dynamic adjust the selected classes
def combined_approach(predicted_confidences, class_labels, base_threshold = 0.6, dominant_index = 13, increment_factor = 0.1, dominance_margin = 0.1, adaptive_increment = True, strict_margin = True, proximity_alerts = None, max_threshold = 0.8):
    
    """
        Combined approach to create a more robust filter, 
        where it adjusts whether the threshold is based on 
        the confidence difference and also dynamically increases 
        the threshold if the monitor remains the predominant class.
    """

    # If there are no proximity alerts, return None
    if proximity_alerts is None:
        proximity_alerts = {}

    # If there are no confidences, return None
    if len(predicted_confidences) == 0:
        print("No confidences provided.") # Debug
        return None

    # Sort the confidences
    sorted_indices = np.argsort(predicted_confidences)[::-1]
    top_indices = sorted_indices[:5]

    std_dev = np.std(predicted_confidences) # Sort the confidences
    mean_conf = np.mean(predicted_confidences) # Mean confidence

    # If the top confidence is the dominant class, increase the threshold
    if top_indices[0] == dominant_index:  
        if adaptive_increment:
            # Adjusting increment to be less drastic using a logarithmic approach
            average_conf_diff = np.mean(predicted_confidences[top_indices[0]] - predicted_confidences[top_indices[1]])
            increment = np.log1p(average_conf_diff) # Incrementing by the average confidence difference
        else:
            increment = increment_factor
        base_threshold += increment
        print(f"Threshold increased to {base_threshold:.2f} due to dominance of class {class_labels[dominant_index]}.") # Debug
    
    # If strict_margin is True, adjust the dominance margin based on some codition
    if strict_margin:
        # If the top cofidence is very high, be more strict
        if predicted_confidences[top_indices[0]] > 0.9:
            dominance_margin *= 0.3 # Reducing the margin to make it more strict
        elif predicted_confidences[top_indices[0]] > 0.8:
            dominance_margin *= 0.5 # Reducing the margin to make it more strict
        print(f"Dominance margin adjust to {dominance_margin:.2f} based on top confidence {predicted_confidences[top_indices[0]]:.2f}.") # Debug

    # If the standard deviation is low, decrease the threshold
    if std_dev < 0.05:
        base_threshold = max(base_threshold - 0.05, 0.5)
    elif mean_conf < 0.5:
        base_threshold = max(base_threshold - 0.05, 0.5)
    elif mean_conf > 0.75:
        base_threshold = min(base_threshold + 0.05, max_threshold)
    
    base_threshold = min(base_threshold, max_threshold) # Clamp the threshold

    # If the difference between the top two confidences is less than the dominance margin, return None
    if (predicted_confidences[top_indices[0]] - predicted_confidences[top_indices[1]]) < dominance_margin:
        print("Confidence difference below adjusted dominance margin, no classification made.") # Debug
        return None
    
    # Loop through the top indices
    for index in top_indices:
        # If the confidence is above the threshold, return the class
        if predicted_confidences[index] > base_threshold:
            print(f"Class '{class_labels[index].strip()}' selected with confidence {predicted_confidences[index]:.2f} above threshold {base_threshold:.2f}") # Debug
            object_type = class_labels[index].strip()
            proximity_limit = proximity_alerts.get(object_type, 30) # Defaul proximity limit
            print(f"Check proximity for '{object_type}'. Danger if closer than {proximity_limit}cm.")
            return class_labels[index].strip()
        
    #  If no class confidence exceeds the threshold, return None
    print("No class confidence exceeds the threshold after adjustments.")   
    return None

# Function to the sensity ## Debug
def test_sensitivity():

    # Initialize variables
    class_labels = ["class_" + str(i) for i in range(28)]

    # Test with closely matched confidences
    print("\nTest with closely matched confidences: ")
    confidences_close = np.array([0.61, 0.59, 0.60, 0.62, 0.61] + [0.5] * 10)
    combined_approach(confidences_close, class_labels)

    # Test with varied confidences
    print("\nTest with varied confidences:")
    confidences_varied = np.array([0.92, 0.10, 0.15, 0.20, 0.25] + [0.1] * 10)
    combined_approach(confidences_varied, class_labels)

    # Test with frequent changes in dominant class
    print("\nTest with frequent changes in dominant class:")
    for dominant_index in range(5):
        print(f"Testing with dominant class: class_{dominant_index}")
        confidences = np.random.rand(15)
        confidences[dominant_index] = max(confidences) + 0.1 # Set the dominant class to a high confidence
        combined_approach(confidences, class_labels, dominant_index = dominant_index)
    
    # Test with proximity alerts
    print("\nTest with proximity alerts:")
    proximity_alerts = {'class_0': 20, 'class_1': 15, 'class_2': 10}
    confidences_alerts = np.array([0.75, 0.65, 0.85] + [0.3] * 12)
    combined_approach(confidences_alerts, class_labels, proximity_alerts = proximity_alerts)

# Function preprocess images
def preprocess_image(roi):

    # Resize the ROI to 224x224
    roi_resized = cv2.resize(roi, (224, 224))

    # Normalize the ROI
    roi_normalized = (roi_resized / 127.5) - 1

    return np.expand_dims(roi_normalized, axis = 0)

# Function to detect objects using Keras
def detectObjectsKeras(roi, model, classes_keras, class_name_id, distance, x, y, w, h, img):
    
    # Initialize variables
    detections = []
    scores = []
    boxes = []

    # Process ROI to keras
    roi_array = preprocess_image(roi)

    alert_issued = False

    # Predict the class of the object
    predicted_confidences = model.predict(roi_array) # Get the complete array of confidence scores
    # print("Predicted Confidences:", predicted_confidences) ## Debug

    # Get the index of the highest confidence score
    class_name = combined_approach(predicted_confidences.flatten(), classes_keras)
    # print("Chosen Class Name:", class_name) ## Debug

    # Check if the index is not None
    if class_name is not None:
        class_id = class_name_id[class_name] # Get the class id
        chosen_index = classes_keras.index(class_name)
        confidence_score = predicted_confidences[0][chosen_index] # Get the confidence score
        confidence_threshold = 0.6

         # Check if the confidence score is above the threshold
        if confidence_score > confidence_threshold:
            boxes.append([x, y, x + w, y + h]) # Append the bounding box
            scores.append(confidence_score) # Append the confidence score
            detections.append((class_name, class_id)) # Append the class name and class id
            print(f"Detected Object: {class_name}, Confidence: {confidence_score:.2f}")
        else:
            print("No object meets the combined threshold criteria")
    else:
        print("No suitable index chosen based on confidence scores")

    # Apply non-max suppression to remove overlapping bounding boxes
    if len(boxes) > 0:
        nms_boxes, nms_scores = nonMaxSuppression(np.array(boxes), np.array(scores), 0.6)  # Adjust overlap threshold
        
        # Calculate distance to object
        for i in range(len(nms_boxes)):
            box = nms_boxes[i] # Get the bounding box
            score = nms_scores[i] # Get the confidence score
            x1, y1, x2, y2 = box # Get the coordinates of the bounding box
            class_name, class_id = detections[i] # Get the class of the object

            # Draw prediction and print details
            draw_prediction(img, class_id, score * 100, x1, y1, x2, y2, distance, 'keras')

            # Check if the object is in the list of objects to be alerted
            if emitAlert(distance, class_name):
                alert_issued = True
            # Print details
            if not alert_issued:
                print(f"Object Detected: {class_name} {score * 100:.2f}%, Distance: {distance:.2f} m") 

    # Return the image with the bounding boxes and the detections
    return img, detections    

# Function to check if the detection is plausible
def isDetectionPlausible(class_id, bounding_box, depth_map, size_thresholds = None):

    # Get the bounding box dimensions
    x1, y1, w, h = bounding_box
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    w, h = max(1, int(w)), max(1, int(h)) # Make sure the width and height are at least 1
    x2, y2 = min(depth_map.shape[1], x1 + w), min(depth_map.shape[0], y1 + h) # Get the bottom right corner of the bounding box
 
    if x1 >= x2 or y1 >= y2:
        logging.error(f'Bounding box {bounding_box} is invalid or out of image bounds.') # Log the error
        return False, None
    
    area = (x2 - x1) * (y2 - y1)

    # Define plausible sizes for each class
    if size_thresholds is None:
        size_thresholds = {
        1 : (20, 4000), # bicycle
        2 : (100, 10000), # car
        3 : (300, 2000), # motorbike
        4 : (500, 3000), # aeroplane
        5 : (300, 2500), # bus
        6 : (200, 3500), # train
        7 : (200, 3500), # truck
        8 : (100, 10000), # boat
        9 : (20, 2000), # traffic light
        10 : (20, 1000), # fire hydrant
        11 : (20, 1000), # stop sign
        12 : (20, 1000), # parking meter
        13 : (30, 1500), # bench
        14 : (500, 3000), # bird
        15 : (500, 3000), # cat
        16 : (500, 3000), # dog
        17 : (500, 3000), # horse
        18 : (500, 3000), # sheep
        19 : (500, 3000), # cow
        20 : (600, 4000), # elephant
        21 : (600, 4000), # bear
        22 : (600, 4000), # zebra
        23 : (600, 4000), # giraffe
        25 : (50, 3000), # umbrella
        27 : (60, 2000), # handbag
        28 : (60, 2000), # tie
        29 : (60, 2000), # suitcase
        30 : (100, 3000), # frisbee
        31 : (100, 3000), # skis
        32 : (100, 3000), # snowboard
        33 : (100, 3000), # sports ball
        34 : (100, 3000), # kite
        35 : (100, 3000), # baseball bat
        36 : (100, 3000), # baseball glove
        37 : (100, 3000), # skateboard
        38 : (100, 3000), # surfboard
        40 : (100, 3000), # tennis racket
        41 : (40, 2000), # wine glass
        42 : (40, 2000), # cup
        43 : (10, 1000), # fork
        44 : (10, 1000), # knife
        45 : (10, 1000), # spoon
        46 : (10, 1000), # bowl
        47 : (20, 1500), # banana
        48 : (20, 1500), # apple
        49 : (20, 1500), # sandwich
        50 : (20, 1500), # orange
        51 : (20, 1500), # broccoli
        52 : (20, 1500), # carrot
        53 : (20, 1500), # hot dog
        54 : (20, 1500), # pizza
        55 : (20, 1500), # donut
        56 : (20, 1500), # cake
        57 : (80, 4000), # chair
        58 : (80, 4000), # sofa
        59 : (80, 4000), # pottedplant
        60 : (80, 4000), # bed
        61 : (80, 4000), # diningtable
        62 : (50, 2000), # toilet
        63 : (50, 3000), # tvmonitor
        65 : (30, 1000), # remote
        68 : (40, 1500), # microwave
        69 : (50, 2000), # oven
        70 : (50, 2000), # toaster
        71 : (50, 2000), # sink
        72 : (50, 2000), # refrigerator
        73 : (10, 1000), # vase
        75 : (10, 500), # scissors
        76 : (10, 500), # teddy bear
        78 : (10, 500), # hair drier
        79 : (10, 500), # toothbrush
    }

    # Check if the object is in the list of objects to be alerted
    if class_id in size_thresholds:
        min_size, max_size = size_thresholds[class_id]
        if area < min_size or area > max_size:
            logging.warning(f'Implausible detection for class_id {class_id}: size {area} out of bounds {min_size} - {max_size}') # Implausible detection
            return False, None
    
    # Get the average depth of the bounding box
    depth_region = depth_map[y1 : y2, x1 : x2]
    positive_depth_values = depth_region[depth_region > 0] # Get the positive depth values

    # Check if the average depth is plausible
    if positive_depth_values.size > 0:
        average_distance = np.mean(positive_depth_values)
    else:
        logging.warning(f'No positive depth values for class_id {class_id} in region {bounding_box}') # No positive depth values
        average_distance = float('inf')
    
    if average_distance > 50:
        logging.info(f'Average distance {average_distance} for class_id {class_id} considered too high') # Average distance too high
        return False, average_distance

    return True, calculate_distance(x1, y1, w, h, depth_map)

# Calculate the distance to the object
def calculate_distance(x1, y1, w, h, depth_map):

    # Convert the bounding box coordinates to integers
    x1, y1, w, h = int(x1), int(y1), int(w), int(h)

    # Get the depth values within the bounding box
    depth_values = depth_map[y1:y1+h, x1:x1+w]
    valid_depths = depth_values[depth_values > 0]

    if valid_depths.size == 0:
        logging.warning(f"No valid depth values in the region x:{x1}, y:{y1}, w:{w}, h:{h}")
        return None, None  # or return a suitable default or error state

    average_depth_mm = np.mean(valid_depths)
    median_depth_mm = np.median(valid_depths)

    conversion_factor = 0.001  # Convert from mm to meters
    average_distance = average_depth_mm * conversion_factor
    median_distance = median_depth_mm * conversion_factor

    return median_distance, average_distance

# Function to check if the detection is plausible
def reevaluateDetection(class_id, bounding_box, depth_map, retry_count = 1):
    
    x, y, w, h = bounding_box # Get the bounding box coordinates

    plausible, distance = isDetectionPlausible(class_id, bounding_box, depth_map) # Check if the detection is plausible

    # If the detection is not plausible, try to increase the size of the bounding box
    if not plausible and distance == float('inf') and retry_count > 0:
        # Increase the size of the bounding box
        logging.info(f'Retrying detection for class_id {class_id} with bounding_box {bounding_box}')

        # Increase the size of the bounding box
        new_w = w * 1.1
        new_h = h * 1.1
        new_x = max(0, x - (new_w - w) / 2)
        new_y = max(0, y - (new_h - h) / 2)
        new_bounding_box = (new_x, new_y, new_w, new_h)
        
        # Re-evaluate the detection
        return reevaluateDetection(class_id, new_bounding_box, depth_map, retry_count - 1)
    return plausible, distance

# Function to process the detections
def process_detections(img, depth_map, detections):
    
    distances = []

    # Filter detections
    for det in detections:
        class_id, bbox = det['class_id'], det['bbox']
        plausible, dist_tuple = isDetectionPlausible(class_id, bbox, depth_map)

        # If the detection is not plausible, try to increase the size of the bounding box
        if not plausible:
            plausible, dist_tuple = reevaluateDetection(class_id, bbox, depth_map, retry_count=1)

        # If the detection is plausible, add it to the list
        if plausible:
            # Assuming dist_tuple is (median, average), and you need average
            average_distance = dist_tuple[1]
            distances.append((class_id, average_distance, bbox))

    # Sort detections by distance
    if distances:
        distances.sort(key=lambda x: x[1])
        closest_distance = distances[0][1]

    # Draw detections
    for class_id, dist, bbox in distances:
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h

        # Check if the difference between this detection's distance and the closest distance is less than 5
        if abs(dist - closest_distance) < 5:
            draw_prediction(img, class_id, 1.0, x1, y1, x2, y2, dist, 'yolo')

# Function to detect objects using Yolo
def detectObjects(img, depth_map):
    #  Get the image dimensions 
    Width = img.shape[1]
    Height = img.shape[0]
    scale = 1/255
    
    # Convert image to blob
    blob = cv2.dnn.blobFromImage(img, scale, (416,416), (0,0,0), False, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net)) # Forward pass throught the network

    # Initialization for late use
    class_ids = []
    confidences = []
    boxes = []
    detections = []
    conf_threshold = 0.6 #  Confidence threshold
    nms_threshold = 0.5 #  Non-maximum suppression threshold



    # Iterate over each detection
    for out in outs:
        for detection in out:
            scores = detection[5:] # Get scores of all scores 
            class_id = np.argmax(scores) # Class with the  highest score
            confidence = scores[class_id] # Confidence of the prediction
            if confidence > 0.5: # Filter out weak detections

                # Calculate bounding box coordinates
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = detection[2] * Width
                h = detection[3] * Height
                x = int (center_x - w / 2)
                y = int (center_y - h / 2)

                # Append to lists
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

                # Append to detections list for later processing
                detections.append({'class_id': class_id, 'bbox': (x, y, w, h), 'confidence': confidence})
    
    # Apply Non-max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    print("aqui. Detections:",detections)
    # Process the detections
    process_detections(img, depth_map, detections)
    print("ALI. Detections:",detections)

    if len(indices) == 0:
        # If no object is detected, use the Keras model to detect objects
        roi = img
        average_distance = get_average_distance_for_whole_image(depth_map)
        #detectObjectsKeras(roi, keras_model, CLASSES_KERAS, class_name_id, average_distance, 0, 0, Width, Height, img)
    
    alert_issued = False

    # Draw the bounding boxes and labels of the detections
    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        # Calculate the average distance of the object
        depth_region = depth_map[int(y): int(y + h), int(x): int(x + w)]
        positiveDepthValues = depth_region[depth_region > 0]

        if positiveDepthValues.size > 0:
            average_distance = np.mean(positiveDepthValues) # Ignore zero values, if applicable
        else:
            average_distance = float('inf')
        distance_meters = average_distance # Adjust as necessary for the correct unit

        object_name = CLASSES[class_ids[i]] # Get the name of the detected object

        if emitAlert(distance_meters, object_name):  # Calls the alert function with the distance and detected object
            alert_issued = True

        if not alert_issued:
            #Print the detection in the console
            print(f"Object detected: {CLASSES[class_ids[i]]}, Distance: {distance_meters:.2f} meters!")

    return img

def get_average_distance_for_whole_image(depth_map):
    positiveDepthValues = depth_map[depth_map > 0]
    return np.mean(positiveDepthValues) if positiveDepthValues.size > 0 else float('inf')

# OPENCV SAMPLE + INTEGRATED OBJECT DETECTION WITH YOLO
class MyListener(roypy.IDepthDataListener):
    def __init__(self, q):
        super(MyListener, self).__init__()
        self.frame = 0
        self.done = False
        self.undistortImage = True
        self.lock = threading.Lock()
        self.once = False
        self.queue = q

    def onNewData(self, data):
        p = data.npoints()
        self.queue.put(p)

    def paint (self, data):
        """Called in the main thread, with data containing one of the items that was added to the
        queue in onNewData.
        """

        # mutex to lock out changes to the distortion while drawing
        self.lock.acquire()

        depth = data[:, :, 2]
        gray = data[:, :, 3]
        confidence = data[:, :, 4]

        zImage = np.zeros(depth.shape, np.float32)
        grayImage = np.zeros(depth.shape, np.float32)
        depthImgRaw = depth
        depthImgProcess = processDepthImage(depthImgRaw)  

        # iterate over matrix, set zImage values to z values of data
        # also set grayImage adjusted gray values
        xVal = 0
        yVal = 0
        for x in zImage:        
            for y in x:
                if confidence[xVal][yVal]> 0:
                  grayImage[xVal,yVal] = self.adjustGrayValue(gray[xVal][yVal])
                yVal=yVal+1
            yVal = 0
            xVal = xVal+1

        grayImage8 = np.uint8(grayImage)

        # apply undistortion
        if self.undistortImage: 
            grayImage8 = cv2.undistort(grayImage8,self.cameraMatrix,self.distortionCoefficients)

        # convert the image to rgb first, because YOLO needs 3 channels, and then detect the objects
        yoloResultImageGray = detectObjects(cv2.cvtColor(grayImage8, cv2.COLOR_GRAY2RGB), depth)
        yoloResultDepthImage = detectObjects(depthImgProcess, depthImgRaw)
        
        # finally show the images
        cv2.imshow("Gray Image", yoloResultImageGray)
        cv2.imshow("Depth Image", yoloResultDepthImage)

        self.lock.release()
        self.done = True

    def setLensParameters(self, lensParameters):
        # Construct the camera matrix
        # (fx   0    cx)
        # (0    fy   cy)
        # (0    0    1 )
        self.cameraMatrix = np.zeros((3,3),np.float32)
        self.cameraMatrix[0,0] = lensParameters['fx']
        self.cameraMatrix[0,2] = lensParameters['cx']
        self.cameraMatrix[1,1] = lensParameters['fy']
        self.cameraMatrix[1,2] = lensParameters['cy']
        self.cameraMatrix[2,2] = 1

        # Construct the distortion coefficients
        # k1 k2 p1 p2 k3
        self.distortionCoefficients = np.zeros((1,5),np.float32)
        self.distortionCoefficients[0,0] = lensParameters['k1']
        self.distortionCoefficients[0,1] = lensParameters['k2']
        self.distortionCoefficients[0,2] = lensParameters['p1']
        self.distortionCoefficients[0,3] = lensParameters['p2']
        self.distortionCoefficients[0,4] = lensParameters['k3']

    def toggleUndistort(self):
        self.lock.acquire()
        self.undistortImage = not self.undistortImage
        self.lock.release()

    # Map the gray values from the camera to 0..255
    def adjustGrayValue(self,grayValue):
        clampedVal = min(400,grayValue) # try different values, to find the one that fits your environment best
        newGrayValue = clampedVal / 400 * 255
        return newGrayValue

def main ():
    # Set the available arguments
    platformhelper = PlatformHelper()
    parser = argparse.ArgumentParser (usage = __doc__)
    add_camera_opener_options (parser)
    options = parser.parse_args()
   
    opener = CameraOpener (options)

    try:
        cam = opener.open_camera ()
    except:
        print("could not open Camera Interface")
        sys.exit(1)

    try:
        # retrieve the interface that is available for recordings
        replay = cam.asReplay()
        print ("Using a recording")
        print ("Framecount : ", replay.frameCount())
        print ("File version : ", replay.getFileVersion())
    except SystemError:
        print ("Using a live camera")

    q = queue.Queue()
    l = MyListener(q)
    cam.registerDataListener(l)
    cam.startCapture()

    lensP = cam.getLensParameters()
    l.setLensParameters(lensP)

    process_event_queue (q, l)

    cam.stopCapture()
    print("Done")

def process_event_queue (q, painter):

    while True:
        try:

            # try to retrieve an item from the queue.
            # this will block until an item can be retrieved
            # or the timeout of 1 second is hit
            if len(q.queue) == 0:
                item = q.get(True, 1)
            else:
                for i in range (0, len (q.queue)):
                    item = q.get(True, 1)
        except queue.Empty:
            # this will be thrown when the timeout is hit
            break
        else:
            painter.paint(item) 
            # waitKey is required to use imshow, we wait for 1 millisecond
            currentKey = cv2.waitKey(1)
            """print(f"Current  key pressed: {currentKey}")""" #for debuging purposes
            if currentKey == ord('d'):
                painter.toggleUndistort()
            # close if escape key pressed
            if currentKey == 27: 
                break

if (__name__ == "__main__"):
    main()
    test_sensitivity()
