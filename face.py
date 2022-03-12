import cv2
import numpy as np 
  #pyzbar helps in detection and decoding of the qrcode
import pickle,time
import pyttsx3   #offline lib for tts
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
import os
import pymongo
from pymongo import MongoClient
from PIL import Image
import requests
from gtts import gTTS
from playsound import playsound



BASEURL = 'http://127.0.0.1:5000/'
""" import requests

url = 'https://www.w3schools.com/python/demopage.php'
myobj = {'somekey': 'somevalue'}

x = requests.post(url, data = myobj)

print(x.text) """

with open('QR.txt') as f:
    authUser = f.read().splitlines()

#Initiallize speech engine
def speech_to_text(text):
    mytext = "Welcome,{} unlocking door.The door will remain open for the next 5 seconds".format(text)
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
    myobj.save("sound/welcome.mp3")
    

flag=True  # to switch between face recognition and qr code decoding
maskFlag = True
spoofFlag = True
MAX_TRY= 3
tries=0  #for invalid face recognition
flag_face_recognised=False   #to keep track if the user face is recognized
flag_face_not_recognised=False

no_of_adjacent_prediction=0
no_face_detected=0  #to track the number of times the face is detected
prev_predicted_name=''   #to keep track of the previously predicted face(w.r.t frame)
count_frames = total_no_face_detected = 0

time_out_no_of_frames_after_qrcode=0

font=cv2.FONT_HERSHEY_SIMPLEX
clr=(255,255,255)

cap=cv2.VideoCapture(0)
qr = cv2.QRCodeDetector()


response = requests.get(f'{BASEURL}/unlock')
data = response.json()
print(data)

while(True):
    ret,frame = cap.read()
    text, bbox, _ = qr.detectAndDecode(frame)
    if(flag):
        if(text in authUser):   #Check private key
            flag=False
            tries=0
            speech_to_text(text)
            playsound('sound/welcome.mp3')
            #speak("Valid qr code, proceed to face recognition")
            time_out_no_of_frames_after_qrcode=0
            print("valid")
            print(text)
            requests.post(f'{BASEURL}/alert', data = {'name': text, "status": "accepted"})
            requests.post(f'{BASEURL}/history', data = {'name': text, "status": "accepted"})
            vid = cv2.VideoCapture("door.mp4")
            while True:
                _ret, _frame = vid.read()
                if _ret == True:
                    cv2.imshow('frame', _frame)
                    time.sleep(5)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                else:
                    break
        else:
            
            print("INVALID QR CODE")  

    cv2.imshow('Face Recognition Cam',frame)
    ch=cv2.waitKey(20) #delay of 1ms    
    if(ch==113):
        break

