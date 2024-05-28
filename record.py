import cv2 as cv
import os
import csv
import argparse
import shutil
 
 
def record():
    # argument parser for terminal prompts
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["record"])
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()
 
    # check if objects folder exists, if not create
    folder_path = os.path.join("objects", args.folder)
    if os.path.exists(folder_path):
        # delete existing folder
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
 
    # activate webcam, if not end skript
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("error opening camera")
        exit()
 
    # using face cascade to detect faces
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
 
    # count every detected face with counter
    face_counter = 0
 
    # counter to skip frames after saving an image
    skip_counter = 0
 
    # loop for opening the cam til user interrupts
    while True:
        # reads cam frame-by-frame
        # cam returns a tuple (ret, frame) ret = boolean whether the frame was read successfully
        # ret is not necesarry in the skript thats why its replaces to "_"
        _, frame = cam.read()
 
        # single frame convert to grayscale ( necesarry for detecting faces with haarcascades )
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
        # function returns a LIST of rectangle-coordinates in which faces are recognized.
        # IF len(faces) > 0 ==> face detected
        # ((Parameters are scaling factors to find nearby rectangles. They are specified in the read me.))
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
 
        # faces = list of coordinates where a face was detected
        if len(faces) > 0 and skip_counter == 0:
            #save face picture with path and name
            pic_name = os.path.join(folder_path, f"face_{face_counter}.png")
            cv.imwrite(pic_name, frame)
 
            # save face coordinates to CSV with same name
            csv_name = os.path.join(folder_path, f"face_{face_counter}.csv")
            with open(csv_name, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["x", "y", "width", "height"])  # Write column as headers
                for (x, y, w, h) in faces:
                    writer.writerow([x, y, w, h])
 
            # iterate counter for detecting next face
            face_counter += 1
 
            #set skip counter to 30
            skip_counter = 30
       
        # decrease skip counter if greater than 0
        if skip_counter > 0:
            skip_counter -= 1
 
        # blue rectangles in the detected faces to veryfy the detection
        # should only be on the live cam video an not on the saved picture
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
 
        # open window for live cam video
        cv.imshow('Face Detection', frame)
 
        # if the if condition sets to "TRUE" (when ´q´is pressed),
        # the "break" stoops only the while TRUE loop from line 36
        if cv.waitKey(30) & 0xFF == ord('q'):  
            break
 
    # camera is released to the system again
    cam.release()
 
    # close cam window
    cv.destroyAllWindows()
 
 
if __name__ == "__main__":
    record()