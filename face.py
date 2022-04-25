import face_recognition
import cv2
import pickle
import numpy as np
# Get the face encodings for the known images
cap=cv2.VideoCapture(0)
known_faces, known_names= pickle.loads(open('face_encodings.pickle', "rb").read())


while True:
    # Grab a single frame of video
    ret, frame =  cap.read()


    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
# See how far apart the test image is from the known faces
    for encodeFace, faceLoc in zip(face_encodings, face_locations):
        face_distances = face_recognition.face_distance(known_faces, encodeFace)
        matches = face_recognition.compare_faces(known_faces, encodeFace, tolerance=0.5)
        matchIndex = np.argmin(face_distances)
        print(matchIndex)
        print(matches[matchIndex])
        if matches[matchIndex]:
            print("known face")
        else:
            print("unknown face")

        

    cv2.imshow('Face Recognition Cam',frame)
    ch=cv2.waitKey(1) #delay of 1ms    
    if(ch==113):
        break


cap.release()
cv2.destroyAllWindows()