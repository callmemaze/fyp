import face_recognition
import imutils
import pickle
import time
import cv2
import os
import base64
import pymongo
from pymongo import MongoClient

KNOWN_FACES_DIR = 'dataset'

CONNECTION_STRING = "mongodb+srv://Maze:Maze@cluster0.bjjtz.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
client = MongoClient(CONNECTION_STRING)
db = client.get_database('myFirstDatabase')
collection = db.get_collection('faces')


print('Loading known faces...')
known_faces = []
known_names = []

# oranize known faces as subfolders of KNOWN_FACES_DIR
# Each subfolder's name becomes our label (name)
for name in os.listdir(KNOWN_FACES_DIR):

    # Next we load every file of faces of known person
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        print(f'Processing image of: {name}')
        # Load an image
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')

        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
        try:
            encoding = face_recognition.face_encodings(image)[0]
        except IndexError:
            continue
        # Append encodings and name
        img = f'{KNOWN_FACES_DIR}/{name}/{filename}'
        with open(img, "rb") as imageFile:
            face = base64.b64encode(imageFile.read())

        collection.insert_one({'name': name, 'face': face})
        known_faces.append(encoding)
        known_names.append(name)



print("[INFO] serializing encodings...")
data =[known_faces, known_names]
f = open('face_encodings.pickle', "wb")
f.write(pickle.dumps(data))
f.close()


