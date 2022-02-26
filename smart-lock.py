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

CONNECTION_STRING = "mongodb+srv://Maze:Maze@cluster0.bjjtz.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
client = MongoClient(CONNECTION_STRING)
db = client.get_database('myFirstDatabase')
collection = db.get_collection('users')

#Specify the recognizer
root_dir = os.getcwd()
# Load Face Detection Model
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
# Load Anti-Spoofing Model graph
json_file = open('antispoofing_models/antispoofing_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load antispoofing model weights 
model.load_weights('antispoofing_models/antispoofing_model.h5')
print("Model loaded from disk")

# known_faces, known_names= pickle.loads(open('face_encodings.pickle', "rb").read())

#load mask detection model
print("[INFO] loading mask detector model")
mask_model = load_model('maskDetectionModel.h5')

#load face encoding model
print("[INFO] loading encodings + face detector...")
known_faces, known_names= pickle.loads(open('face_encodings.pickle', "rb").read())
print('Processing...')


with open('QR.txt') as f:
    authUser = f.read().splitlines()

#Initiallize speech engine
engine = pyttsx3.init() 

def speak(text):  #fn to convert text to speech
    engine.say(text)
    engine.runAndWait()
    

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
while(True):
    ret,frame = cap.read()
    text, bbox, _ = qr.detectAndDecode(frame)
    if(flag):
        if(text in authUser):   #Check private key
            flag=False
            tries=0
            #speak("Valid qr code, proceed to face recognition")
            time_out_no_of_frames_after_qrcode=0
            print("valid")
                
        else:
            print("INVALID QR CODE")  
        
    elif (spoofFlag):
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:  
            face = frame[y-5:y+h+5,x-5:x+w+5]
            resized_face = cv2.resize(face,(160,160))
            resized_face = resized_face.astype("float") / 255.0
            # resized_face = img_to_array(resized_face)
            resized_face = np.expand_dims(resized_face, axis=0)
            # pass the face ROI through the trained liveness detector
            # model to determine if the face is "real" or "fake"
            preds = model.predict(resized_face)[0]
            if preds< 0.5:
                label = 'real'
                spoofFlag=False
                #speak("Live face, Proceding to face recognition")
                cv2.putText(frame, label, (x,y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                cv2.rectangle(frame, (x, y), (x+w,y+h),
                        (0, 0, 255), 2)
            else:
                #speak("Spoof face")
                label = 'real'
    elif (maskFlag):
        scale_factor = 1.05
        min_neighbour = 6
        faces = face_cascade.detectMultiScale(frame, scale_factor, min_neighbour, minSize=(100, 100),
                                 flags=cv2.CASCADE_SCALE_IMAGE)
    
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            cropped_face = frame[y:y+h, x:x+w]  
            face = cv2.resize(cropped_face, (224, 224))
            im = Image.fromarray(face, 'RGB')
            img_array = np.array(im)
            img_array = np.expand_dims(img_array, axis=0)            
            # Use the model to predict
            pred = mask_model.predict(img_array)    
            #Check the threshold
            if(pred[:,1] > 0.001):
                name='no mask found'
                maskFlag = False
            else:
                name='mask found'
            cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

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

            facesCurFrame = face_recognition.face_locations(frame, model='hog')
            encodingsCurFrame = face_recognition.face_encodings(frame,facesCurFrame)
            for encodeFace, faceLoc in zip(encodingsCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(known_faces, encodeFace)
                faceDis = face_recognition.face_distance(known_faces, encodeFace)
                matchIndex = np.argmin(faceDis)
                if True in matches:
                    match = known_names[matches.index(True)]
                    if (prev_predicted_name == match):
                        print("same face")
                    prev_predicted_name = match
                    flag_face_recognised=True
                else:
                    print("unknown face")
                    flag_face_not_recognised=True

        if(flag_face_recognised):    #if face is recognized then open the door
            #speak("Welcome "+matches.replace('_',' ')+", unlocking door. The door will remain open for the next 5 seconds")
            print("DOOR is OPEN")
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
            time.sleep(5)
            speak("Closing door")
            print("DOOR is CLOSED")
            flag_face_recognised=False
            flag=True         #to start from qrcode

        if(flag_face_not_recognised):
            speak("Face not recognised. The door will remain closed")    
            time.sleep(2)
            flag_face_not_recognised=False
            tries+=1
            if(tries>=MAX_TRY):
                speak("User authentication failed as face is not recognised")
                flag=True       #to start from qrcode
                tries=0

        if(time_out_no_of_frames_after_qrcode>=400):
            speak("User authentication failed due to time out")
            flag=True     #to start from qrcode



    # cv2.imshow('TRIAL',frame)
    cv2.imshow('Face Recognition Cam',frame)
    ch=cv2.waitKey(20) #delay of 1ms    
    if(ch==113):
        break


print("No. of frames : ",count_frames," |   No. of times face detected : ",total_no_face_detected)
cap.release()
cv2.destroyAllWindows()