#training File WORKING::: EXE TIME 1HOUR 30MIN ~~

import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from keras import models, layers, optimizers
import warnings
warnings.filterwarnings('ignore')


def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1),
                      len(model_history.history['accuracy']) / 10)
    axs[0].legend(['train', 'val'], loc='best')

    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()
df = pd.read_csv(r'C:\Users\lenovo\OneDrive\Desktop\Emotion-detection-master\Emotion-detection-master\src\fer2013.csv', delimiter=',')
df.head()

training = df[df['Usage'] == 'Training']
testing = df[df['Usage'] != 'Training']

data = testing['pixels'].apply(lambda x: x.split(' ')).to_numpy()
testing_data = np.zeros((len(data), len(data[0])))
for i in range(len(data)):
    testing_data[i] = np.array(data[i])

testing_data = testing_data.astype('float')
testing_data.shape

data = training['pixels'].apply(lambda x: x.split(' ')).to_numpy()
training_data = np.zeros((len(data), len(data[0])))
for i in range(len(data)):
    training_data[i] = np.array(data[i])

training_data = training_data.astype('float')
training_data.shape

training_data.shape = (len(training_data), 48, 48, 1)
training_labels = training['emotion'].values.astype('float')

testing_data.shape = (len(testing_data), 48, 48, 1)
testing_labels = testing['emotion'].values.astype('float')

model = models.Sequential()

model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
num_epoch = 50
training_labels = keras.utils.to_categorical(training_labels, 7)
testing_labels = keras.utils.to_categorical(testing_labels, 7)
model_info = model.fit(
        training_data, training_labels,
        epochs=num_epoch,
        validation_data=(testing_data, testing_labels))
model.save_weights(r'C:\Users\lenovo\OneDrive\Desktop\Emotion-detection-master\Emotion-detection-master\src\model.h5')
# load trained model
model.load_weights(r'C:\Users\lenovo\OneDrive\Desktop\Emotion-detection-master\Emotion-detection-master\src\model.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary mapping class labels with corresponding emotions
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# start the webcam feed
cap = cv2.VideoCapture(0)

# To find haar cascade to draw bounding box around face
facecasc = cv2.CascadeClassifier(r'C:\Users\lenovo\OneDrive\Desktop\Emotion-detection-master\Emotion-detection-master\src\haarcascade_frontalface_default.xml')

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
