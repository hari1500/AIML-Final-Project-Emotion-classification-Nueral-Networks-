from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import sys

from utils.datasets import get_labels
from utils.inference import detect_faces, draw_text, draw_bounding_box, load_detection_model

# parameters for loading data and images
detection_model_path = '../models/haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = '../models/model.85-0.65.hdf5'
source_image_path = sys.argv[1]

# hyper-parameters for bounding boxes shape
# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = get_labels()

# reading the frame
frame = cv2.imread(source_image_path)
frame = imutils.resize(frame,width=400)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
faces = detect_faces(face_detection, gray)
    
if len(faces) > 0:
    faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (fX, fY, fW, fH) = faces
    # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
    # the ROI for classification via the CNN
    roi = gray[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]

    if label == 'angry': color = emotion_probability * np.asarray((255, 0, 0))
    elif label == 'sad': color = emotion_probability * np.asarray((0, 0, 255))
    elif label == 'happy': color = emotion_probability * np.asarray((255, 255, 0))
    elif label == 'surprise': color = emotion_probability * np.asarray((0, 255, 255))
    else: color = emotion_probability * np.asarray((0, 255, 0))

    color = color.astype(int)
    color = color.tolist()

    draw_bounding_box(faces, rgb, color)
    draw_text(faces, rgb, label, color, 0, -5, 1, 1)

bgr_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
cv2.imshow('face', bgr_image)
cv2.imwrite('../output/'+source_image_path.split('/')[-1],bgr_image)
if cv2.waitKey() & 0xFF == ord('q'): sys.exit()
