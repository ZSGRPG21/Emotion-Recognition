# Main file NOT WORKING///
import numpy as np
import cv2
from keras import models, layers
import warnings
warnings.filterwarnings('ignore')
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])
# load trained model
model.load_weights(r'Emotion-detection-master\src\model.h5')
# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)
# dictionary mapping class labels with corresponding emotions
# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emotion_dict = {0: "😠 Angry", 1: "😖 Disgusted", 2: "😨 Fearful", 3: "😄 Happy", 4: "😐 Neutral", 5: "😢 Sad", 6: "😲 Surprised"}
# start the webcam feed
cap = cv2.VideoCapture(0)
# To find haar cascade to draw bounding box around face
facecasc = cv2.CascadeClassifier(r'Emotion-detection-master\src\haarcascade_frontalface_default.xml')
while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        break;
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)
    cv2.imshow('Video', cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC))
    # press 'q' to quit webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
