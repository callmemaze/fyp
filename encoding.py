import os,pickle,numpy as np
from PIL import Image   
import cv2

face_cascade=cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# print(os.getcwd())
BASE_DIR=os.path.dirname(os.path.abspath(__file__)) 
image_dir=os.path.join(BASE_DIR,'dataset')  
# print(image_dir)

image_id=0
label_id={}
y_train_label=[]
x_train_data=[]

#To see the images in images dir
for root,dirs,files in os.walk(image_dir):
    print(root,dirs,files)
    for file in files:
        print("FILES : ",file)
        if(file.endswith('png') or file.endswith('jpg')):
            path=os.path.join(root,file)
            label=os.path.basename(root).replace(' ','_').lower()
           
            if(label not in label_id):  #Assinging numbers to labels(LabelEncoding)
                label_id[label]=image_id
                image_id+=1



            pil_image=Image.open(path).convert('L')  #Coverting into grayscale.
            #Resizing the image for better accuracy.
            size=(400,533)
            final_image=pil_image.resize(size,Image.ANTIALIAS)

            image_array=np.array(final_image)  #Convert the gray_scale image into an array of pixels.
            # print(image_array)

            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.2, minNeighbors=5)
            for (x,y,w,h) in faces:
              roi_gray = image_array[y:y + h, x:x + w]    #Pixel values of the face region.
               
            y_train_label.append(label_id[label])         #Appending the label number
            x_train_data.append(roi_gray)               #Appending the pixel values of the label face


print(y_train_label)
# print(x_train_data)
print(label_id)

#Using Pickle to save label_id
with open('label_ids.pickle','wb') as fw:
    pickle.dump(label_id,fw)


#Training the OpenCV recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(x_train_data,np.array(y_train_label))
recognizer.save('encoding.yml')