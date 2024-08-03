import csv
import os
import cv2
import numpy as np
import pandas as pd
import datetime
import time

def TakeImage(l1, l2, haarcasecade_path, trainimage_path, message, err_screen, text_to_speech):
    if l1 == "" and l2 == "":
        t = 'Please enter your Enrollment Number and Name.'
        text_to_speech(t)
    elif l1 == '':
        t = 'Please enter your Enrollment Number.'
        text_to_speech(t)
    elif l2 == "":
        t = 'Please enter your Name.'
        text_to_speech(t)
    else:
        try:
            # Open camera
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                raise Exception("Could not open camera.")
            
            # Load the Haar Cascade classifier
            detector = cv2.CascadeClassifier(haarcasecade_path)
            if detector.empty():
                raise Exception("Haar Cascade classifier file not found or invalid.")

            # Create directory for saving images
            Enrollment = l1
            Name = l2
            sampleNum = 0
            directory = Enrollment + "_" + Name
            path = os.path.join(trainimage_path, directory)
            os.makedirs(path, exist_ok=True)

            while True:
                ret, img = cam.read()
                if not ret:
                    raise Exception("Failed to capture image from camera.")

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    sampleNum += 1
                    cv2.imwrite(
                        os.path.join(path, f"{Name}_{Enrollment}_{sampleNum}.jpg"),
                        gray[y:y + h, x:x + w]
                    )
                    cv2.imshow("Frame", img)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                elif sampleNum > 50:
                    break

            cam.release()
            cv2.destroyAllWindows()

            # Save student details to CSV
            row = [Enrollment, Name]
            csv_path = r"C:\\Users\\USER\\Desktop\\Attendance-Management-system-using-face-recognition-master\\StudentDetails\\studentdetails.csv"
            with open(csv_path, "a+", newline='') as csvFile:
                writer = csv.writer(csvFile, delimiter=",")
                writer.writerow(row)

            # Provide feedback to user
            res = f"Images saved for ER No: {Enrollment} Name: {Name}"
            message.configure(text=res)
            text_to_speech(res)

        except FileExistsError:
            text_to_speech("Student data already exists.")
        except Exception as e:
            err_msg = str(e)
            text_to_speech(err_msg)
            print(err_msg)
