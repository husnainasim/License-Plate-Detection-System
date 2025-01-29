# anpr-->(Automatic Number Plate Recognition)

# Task: Detect license plates in a video using YOLOv10 model and OpenCV.
# Steps:
# 1. Create a video capture object to read the video file.
# 2. Initialize the YOLOv10 model with pre-trained weights.
# 3. Initialize a frame counter.
# 4. Define the class names for the detected objects.
# 5. Process each frame of the video in a loop.
# 6. Predict objects in the frame using the YOLOv10 model.
# 7. Draw bounding boxes and labels for detected objects.
# 8. Display the processed frame.
# 9. Break the loop if the '1' key is pressed or if there are no more frames.
# 10. Release the video capture object and close all OpenCV windows.

# Importing the necessary Libraries
import json

import cv2
from ultralytics import YOLOv10
import numpy as np
import math
import re
import os
import sqlite3
from datetime import datetime
from paddleocr import PaddleOCR


os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Step 1: Create a video capture object to read the video file
cap = cv2.VideoCapture('yolov10/Resources/carLicence4.mp4')

# Step 2: Initialize the YOLOv10 model with pre-trained weights
model = YOLOv10("yolov10/weights/best.pt")

# Step 3: Initialize a frame counter
count = 0

# Step 4: Define the class names for the detected objects
className = ["license"]
# Initialize the Paddle PaddleOCR
ocr= PaddleOCR(use_angle_cls=True, use_gpu=False)

def paddle_ocr(frame,x1,y1,x2,y2):
    frame=frame[y1:y2, x1:x2]
    result=ocr.ocr(frame, det=False, rec=True, cls=False)
    text=""
    for r in result:
        scores = r[0][1]
        if np.isnan(scores):
            scores = 0
        else:
            scores = int(scores*100)
        if scores> 60:
            text=r[0][0]
    
    pattern=re.compile('[\W]')
    text = pattern.sub('', text)
    text = text.replace('???', "")
    text = text.replace('O', "0")
    text = text.replace("粤", "")
    return str(text)
    
def save_json(license_plates,startTime,endTime):
    # Generate Individual JSON files for each 20 seconds interval
    interval_data={
        "Start Time":startTime.isoformat(),
        "End Time":endTime.isoformat(),
        "License Plate":list(license_plates),
    }
    interval_file_path= "yolov10/json/output_" + datetime.now().strftime("%Y%m%d%H%M%S")+ ".json"
    with open(interval_file_path, 'w') as f:
        json.dump(interval_data, f, indent=2)
        
    # Cummulative JSON File
    
    cummulative_file_path="yolov10/json/LicensePlateData.json"
    if os.path.exists(cummulative_file_path):
        with open(cummulative_file_path,'r') as f:
            existing_data=json.load(f)
    else:
        existing_data=[]
    #Adding new interval data to cummulative data
    existing_data.append(interval_data)
        
    with open(cummulative_file_path,'w') as f:
            json.dump(existing_data,f,indent=2)

    #Save data to the database
    save_to_database(license_plates,startTime,endTime)
    
def save_to_database(license_plates,start_time,end_time):
    conn=sqlite3.connect("yolov10/licensePlatesDatabase.db")
    cursor=conn.cursor()
    for plate in license_plates:
        cursor.execute(
            '''
            INSERT INTO LicensePlates(start_time,end_time,license_plate)
            VALUES(?,?,?)
            ''',
            (start_time.isoformat(),end_time.isoformat(),plate)
        )
    conn.commit()
    conn.close()
       
            
startTime=datetime.now()    
license_plates=set()        

# Step 5: Process each frame of the video in a loop
while True:
    # Read the next frame from the video capture object
    ret, frame = cap.read()
    
    # Check if the frame was successfully read
    if ret:
        currentTime=datetime.now()
        # Increment the frame counter
        count += 1
        print(f"Frame Number: {count}")
        
        # Step 6: Predict objects in the frame using the YOLOv10 model
        results = model.predict(frame, conf=0.45)
        
        # Iterate over each result in the prediction results
        for result in results:
            # Extract the bounding boxes from the result
            boxes = result.boxes
            
            # Iterate over each bounding box
            for box in boxes:
                # Extract the coordinates of the bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                # Convert the coordinates to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Step 7: Draw bounding boxes and labels for detected objects
                # Draw a rectangle around the detected object
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Get the class index and name for the detected object
                classNameInt = int(box.cls[0])
                clsName = className[classNameInt]
                
                # Round the confidence score to two decimal places
                conf = math.ceil(box.conf[0] * 100) / 100
                
                # Create a label string with the class name and confidence score
                # label = f'{clsName}:{conf}'
                label=paddle_ocr(frame,x1,y1,x2,y2)
                if label:
                    license_plates.add(label)
                
                # Calculate the size of the label text
                textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                
                # Calculate the bottom-right corner of the text background rectangle
                c2 = x1 + textSize[0], y1 - textSize[1] - 3
                
                # Draw a filled rectangle behind the text for better visibility
                cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                
                # Overlay the label text on the frame
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        if (currentTime-startTime).seconds>=20:
            endTime=currentTime
            save_json(license_plates,startTime,endTime)
            startTime=currentTime
            license_plates.clear()
        # Step 8: Display the processed frame
        cv2.imshow('Video', frame)
        
        # Step 9: Break the loop if the '1' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        # Break the loop if there are no more frames to read
        break

# Step 10: Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

