import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
from tensorflow.keras.models import model_from_json
import face_recognition

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
# video.open("http://192.168.1.101:8080/video")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)

path = 'images'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    print(classNames)

def findEncodings(images):
    encodeList= []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        return encodeList
encodeListKnown = findEncodings(images)
print('Encoding Complete')

with open('QR.txt') as f:
    authUser = f.read().splitlines()

video = cv2.VideoCapture(1)
qr = cv2.QRCodeDetector()
while video.isOpened():
    try:
        ret,frame = video.read()
        text, bbox, _ = qr.detectAndDecode(frame)
        if text in authUser:
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
                if preds> 0.5:
                    label = 'spoof'
                    cv2.putText(frame, label, (x,y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    cv2.rectangle(frame, (x, y), (x+w,y+h),
                        (0, 0, 255), 2)
                else:
                    label = 'real'
                    facesCurFrame = face_recognition.face_locations(frame)
                    encodingsCurFrame = face_recognition.face_encodings(frame,facesCurFrame)
                    for encodeFace, faceLoc in zip(encodingsCurFrame, facesCurFrame):
                        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                        # print(faceDis)
                        matchIndex = np.argmin(faceDis)
                        # print(matchIndex)
                        if matches[matchIndex]:
                            vid = cv2.VideoCapture("door.mp4")
                            while True:
                                _ret, _frame = vid.read()
                                if _ret == True:
                                    cv2.imshow('frame', _frame)
                                    if cv2.waitKey(1) & 0xFF == ord('q'):
                                        break
                                else:
                                    break
                            name = classNames[matchIndex]
                            print(name)
                            y1,x2,y2,x1 = faceLoc
                            y1, x2, y2, x1 = y1-5, x2+5, y2+5, x1-5
                            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                            cv2.putText(frame,name,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            label = 'Show QR'
            cv2.putText(frame, label, (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)       
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    except Exception as e:
        pass
video.release()        
cv2.destroyAllWindows()