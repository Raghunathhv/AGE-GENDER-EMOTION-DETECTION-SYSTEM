import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
from deepface import DeepFace

# ---------------- Excel Setup ----------------

EXCEL_FILE = "attendance.xlsx"

if not os.path.exists(EXCEL_FILE):
    df = pd.DataFrame(columns=["Name","Age","Gender","Emotion","Date","Time"])
    df.to_excel(EXCEL_FILE,index=False)

# ---------------- Webcam ----------------

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

mode = "NORMAL"
age = "-"
gender = "-"
emotion = "-"
attendance_status = "-"

student_name = input("\nEnter Student First Name: ").strip()

# ---------------- Helper ----------------

def draw_panel(img):
    panel = img.copy()

    cv2.rectangle(panel,(0,0),(640,170),(30,30,30),-1)

    cv2.putText(panel,"SMART AI ATTENDANCE SYSTEM",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)

    cv2.putText(panel,f"MODE: {mode}",(10,60),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    cv2.putText(panel,f"NAME: {student_name}",(10,90),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    cv2.putText(panel,f"AGE: {age}",(10,120),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    cv2.putText(panel,f"GENDER: {gender}",(180,120),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,255),2)

    cv2.putText(panel,f"EMOTION: {emotion}",(360,120),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,255),2)

    cv2.putText(panel,f"ATTENDANCE: {attendance_status}",(10,150),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    return panel

def mark_attendance():
    global attendance_status

    df = pd.read_excel(EXCEL_FILE)
    now = datetime.now()

    new_row = {
        "Name":student_name,
        "Age":age,
        "Gender":gender,
        "Emotion":emotion,
        "Date":now.date(),
        "Time":now.strftime("%H:%M:%S")
    }

    df = pd.concat([df,pd.DataFrame([new_row])],ignore_index=True)
    df.to_excel(EXCEL_FILE,index=False)

    attendance_status="Saved"

# ---------------- Controls ----------------

print("\n1 → Age + Gender")
print("2 → Emotion")
print("3 → Full AI Mode")
print("4 → Attendance")
print("Q → Exit\n")

# ---------------- Main Loop ----------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame,(640,480))

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        face = frame[y:y+h,x:x+w]

        if mode in ["AGE_GENDER","FULL"]:
            result = DeepFace.analyze(face, actions=['age','gender'], enforce_detection=False)
            age = str(int(result[0]['age']))
            gender = result[0]['dominant_gender']

        if mode in ["EMOTION","FULL"]:
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']

        if mode=="ATTENDANCE":
            mark_attendance()

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    panel = draw_panel(frame)
    cv2.imshow("SMART AI ENTERPRISE SYSTEM",panel)

    key = cv2.waitKey(1) & 0xFF

    if key==ord('1'):
        mode="AGE_GENDER"
        emotion="-"
        attendance_status="-"

    elif key==ord('2'):
        mode="EMOTION"
        age="-"
        gender="-"
        attendance_status="-"

    elif key==ord('3'):
        mode="FULL"
        attendance_status="-"

    elif key==ord('4'):
        mode="ATTENDANCE"

    elif key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()