from cgitb import grey
import cv2
import numpy as np 
import face_recognition
  #pyzbar helps in detection and decoding of the qrcode
import pickle,time
  #offline lib for tts
from datetime import datetime
import os
from PIL import Image
import requests
from gtts import gTTS
from playsound import playsound
import socket
from exponent_server_sdk import (
    DeviceNotRegisteredError,
    PushClient,
    PushMessage,
    PushServerError,
    PushTicketError,
)
from requests.exceptions import ConnectionError, HTTPError
import rollbar
import base64


serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
# Bind and listen
serverSocket.bind(("127.0.0.1",5001))
serverSocket.listen(1)


""" CONNECTION_STRING = "mongodb+srv://Maze:Maze@cluster0.bjjtz.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
client = MongoClient(CONNECTION_STRING)
db = client.get_database('myFirstDatabase')
collection = db.get_collection('users') """

BASEURL = 'http://127.0.0.1:5000/'

now = datetime.now()
dt_string = now.strftime("%B %d, %Y  %H:%M:%S")
#Specify the recognizer
root_dir = os.getcwd()
# Load Face Detection Model
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
# Load Anti-Spoofing Model graph
#json_file = open('antispoofing_models/antispoofing_model.json','r')
#loaded_model_json = json_file.read()
#json_file.close()
#model = model_from_json(loaded_model_json)
# load antispoofing model weights 
#model.load_weights('antispoofing_models/antispoofing_model.h5')
print("Model loaded from disk")


#load mask detection model
print("[INFO] loading mask detector model")
#mask_model = load_model('maskDetectionModel.h5')

#load face encoding model
print("[INFO] loading encodings + face detector...")
known_faces, known_names= pickle.loads(open('face_encodings.pickle', "rb").read())
print('Processing...')
print('Success')
print('[INFO] Opening Webcam')
with open('QR.txt') as f:
    authUser = f.read().splitlines()

#Initiallize speech engine
#engine = pyttsx3.init() 

""" def speak(text):  #fn to convert text to speech
    engine.say(text)
    engine.runAndWait() """
def speech_to_text(text):
    mytext = "Welcome,{} unlocking door.The door will remain open for the next 5 seconds".format(text)
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
    myobj.save("sound/welcome.mp3")

def send_push_message(token, message, extra=None):
    try:
        response = PushClient().publish(
            PushMessage(to=token,
                        body=message,
                        data=extra))
    except PushServerError as exc:
        # Encountered some likely formatting/validation error.
        rollbar.report_exc_info(
            extra_data={
                'token': token,
                'message': message,
                'extra': extra,
                'errors': exc.errors,
                'response_data': exc.response_data,
            })
        raise
    except (ConnectionError, HTTPError) as exc:
        # Encountered some Connection or HTTP error - retry a few times in
        # case it is transient.
        rollbar.report_exc_info(
            extra_data={'token': token, 'message': message, 'extra': extra})
    try:
        response.validate_response()
    except PushTicketError as exc:
        # Encountered some other per-notification error.
        rollbar.report_exc_info(
            extra_data={
                'token': token,
                'message': message,
                'extra': extra,
                'push_response': exc.push_response._asdict(),
            })
token = "ExponentPushToken[44fq11JDMwpbR95NufmFzb]"            
flag=True  # to switch between face recognition and qr code decoding
maskFlag = True
spoofFlag = True
MAX_TRY= 3
tries=0  #for invalid face recognition
flag_face_recognised=False   #to keep track if the user face is recognized
flag_face_not_recognised=False
match=""
no_of_adjacent_prediction=0
no_face_detected=0  #to track the number of times the face is detected
prev_predicted_name=''   #to keep track of the previously predicted face(w.r.t frame)
count_frames = total_no_face_detected = 0

time_out_no_of_frames_after_qrcode=0

font=cv2.FONT_HERSHEY_SIMPLEX
clr=(255,255,255)

cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30)
qr = cv2.QRCodeDetector()
start_frame_number = 50
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
def door_open():
    vid = cv2.VideoCapture("door.mp4")
    while True:
        _ret, _frame = vid.read()
        if _ret == True:
            cv2.imshow('frame', _frame)
            if cv2.waitKey(1) & 0xFF == ord('w'):
                break
        else:
            break

ptime = 0
from imutils.video import FPS
from imutils.video import VideoStream
import imutils
stream = VideoStream(src=0).start()
fps = FPS().start()

while(True):
    frame = stream.read()
    frame = imutils.resize(frame, width=400)
    #ret,frame = cap.read()
    text, bbox, _ = qr.detectAndDecode(frame)
    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    #cv2.putText(frame,f'FPS {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2 )

    if(flag):
        if(text in authUser):   #Check private key
            flag=False
            tries=0
            playsound('sound/qr.mp3')
            time_out_no_of_frames_after_qrcode=0
            print("valid")
            print("Valid QR code: ", text)
            cv2.putText(frame,"Valid QR code", (20,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2 ) 
        else:
            cv2.putText(frame,"Show QR code", (20,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2 ) 
        
    else:
        count_frames+=1
        time_out_no_of_frames_after_qrcode+=1
        # print(ret,frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5)

        # print("FACES : ",faces)
        # print(type(frame))   #The frames are automatically converted into numpy arrays of pixels.
        for (x,y,w,h) in faces:
            total_no_face_detected+=1
            no_face_detected+=1
            # print(x,y,w,h)
            roi_gray=gray[y:y+h,x:x+w]  
            facesCurFrame = face_recognition.face_locations(frame, model='cnn')
            encodingsCurFrame = face_recognition.face_encodings(frame,facesCurFrame)
            for encodeFace, faceLoc in zip(encodingsCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(known_faces, encodeFace,tolerance=0.5)
                faceDis = face_recognition.face_distance(known_faces, encodeFace)
                matchIndex = np.argmin(faceDis)
                if matches[matchIndex]:
                    match = known_names[matches.index(True)]
                    prev_predicted_name = match
                    flag_face_recognised=True
                else:
                    print("unknown face")
                    flag_face_not_recognised=True

        if(flag_face_recognised):    #if face is recognized then open the door
            #speak("Welcome "+matches.replace('_',' ')+", unlocking door. The door will remain open for the next 5 seconds")
            
            requests.post(f'{BASEURL}/history', data = {'name': match,"date":dt_string , "status": "accepted"})
            print("DOOR is OPEN")
            print(match)
            playsound('sound/welcome.mp3')
            #speak("Closing door")
            door_open()
            print("DOOR is CLOSED")
            flag_face_recognised=False
            flag=True         #to start from qrcode

        if(flag_face_not_recognised):
            #speak("Face not recognised. The door will remain closed")
            speech_to_text("Face not recognised. The door will remain closed")
            playsound('sound/unauthorize.mp3')               
            time.sleep(2)
            flag_face_not_recognised=False
            tries+=1
            if(tries>=MAX_TRY):
                cv2.imwrite("unknown.png", frame)
                img = 'unknown.png'
                with open(img, "rb") as imageFile:
                    face = base64.b64encode(imageFile.read())
                requests.post(f'{BASEURL}/alert', data = {'name': "unknown", "date":dt_string ,"image": face,"status": "rejected"})
                #speak("User authentication failed as face is not recognised")
                send_push_message(token, "User authentication Detected")
                playsound('sound/sending.mp3')
                serverSocket.settimeout(20)
                try:
                    (clientConnected, clientAddress) = serverSocket.accept()
                    print("Accepted a connection request from %s:%s"%(clientAddress[0], clientAddress[1]))
                    dataFromClient = clientConnected.recv(1024)
                    if(dataFromClient):
                        requests.post(f'{BASEURL}/history', data = {'name': match,"date":dt_string , "status": "rejected"})
                        print("Opening the door")
                        door_open()
                    print(dataFromClient.decode('utf-8'))
                except:
                    requests.post(f'{BASEURL}/history', data = {'name': "unknown","date":dt_string , "status": "rejected"})
                    print("Timeout")
                    playsound("sound/denied.mp3")

                flag=True       #to start from qrcode
                tries=0

        if(time_out_no_of_frames_after_qrcode>=400):
            #speak("User authentication failed due to time out")
            flag=True     #to start from qrcode



    # cv2.imshow('TRIAL',frame)
    cv2.imshow('Face Recognition Cam',frame)
    ch=cv2.waitKey(1) #delay of 1ms    
    if(ch==113):
        break


print("No. of frames : ",count_frames," |   No. of times face detected : ",total_no_face_detected)
cap.release()
stream.stop()
cv2.destroyAllWindows()