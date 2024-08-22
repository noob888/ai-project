# AI model to detect if the online face is real or fake
# Import required packages
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import os
import glob

# Image Paths to test
# If video, then first transform video in to a list of images)
img1_path = ('{ADD_PATH_TO_FACE}')
img2_path = ('{ADD_PATH_TO_FACE_WITH_OPEN_EYES}')
img3_path = ('{ADD_PATH_TO_FACE_WITH_CLOSED_EYES}')

# Confirming and displaying images
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
img3 = cv2.imread(img3_path)

plt.imshow(img1[:, :, ::-1 ]) #saturation
plt.show()
plt.imshow(img2[:, :, ::-1 ]) #saturation
plt.show()
plt.imshow(img3[:, :, ::-1 ]) #saturation
plt.show()

# Verify if deepface can detect face
result = DeepFace.verify(img1_path,img2_path)
result

# Download HAAR cascade face and eye eye wights online and add path below (it's just a google search away)
haar_face_path = '{ADD_YOUR_FILE_PATH}/haarcascade_frontalface_default.xml'
haar_eye_path = '{ADD_YOUR_FILE_PATH}/haarcascade_eye.xml'

# Set face and eye cascades to draw bounding boxes on face and the eye
face_cascade = cv2.CascadeClassifier(haar_face_path)
eye_cascade = cv2.CascadeClassifier(haar_eye_path)

# Store the previous image 1,2 and 3 file paths from a folder (can do it manually too)
downloaded_file_path = '{ADD_YOUR_FOLDER_PATH_HERE}/*.jpg'
image_paths = glob.glob(downloaded_file_path)
image_paths

# Draw bounding boxes on face, eyes and print result
for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Unable to read image at path: {path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray is None:
        print(f"Error: Unable to convert image to grayscale: {path}")
        continue

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(img, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

    # cv2.imshow('Detection', img) deprecated
    cv2.imshow(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# From the images
# detect if the face is real or fake and print result
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for path in image_paths:
  img = cv2.imread(path)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  for (x,y,w,h) in faces:
          cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
          roi_gray = gray[y:y+h, x:x+w]
          roi_color = img[y:y+h, x:x+w]

          eyes = eye_cascade.detectMultiScale(roi_gray)
          for (ex,ey,ew,eh) in eyes:
              cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
  status = []
  if (len(eyes)>2):
    cv2.putText(img,"Eyes are open!", (140,140),cv2.FONT_HERSHEY_PLAIN, 5,(255,255,255),2)
    status +=['open']
  elif (len(eyes)==2):
    cv2.putText(img, "Eyes are closed", (140,140),cv2.FONT_HERSHEY_PLAIN, 5,(255,255,255),2)
    status += ['closed']
  else:
    cv2.putText(img, "No eyes detected", (140,140),cv2.FONT_HERSHEY_PLAIN, 5,(255,255,255),2)

if 'open' and 'closed' in status:
  print("The face is real")
else:
  print('The face is fake')