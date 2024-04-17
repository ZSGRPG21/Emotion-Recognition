# Modified + optimised code
# Train/test time 58 minutes
# After test accuracy: 80.3% (highest achieved)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import models, layers, optimizers
import warnings
from sklearn.model_selection import train_test_split

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


df = pd.read_csv(r'fer2013.csv', delimiter=',')
df.head()

# Extract pixel values and convert to numpy array
data = np.array(df['pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ')).tolist(), dtype=np.float32)

# Reshape data to (num_samples, 48, 48, 1)
data = data.reshape((len(data), 48, 48, 1))

# Extract labels
labels = df['emotion'].values.astype('int')

# Split the data
training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Model architecture
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
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

# Model compilation
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001, amsgrad=True),
              metrics=['accuracy'])
num_epoch = 50

# Model training
model_info = model.fit(
    training_data, training_labels,
    epochs=num_epoch,
    validation_data=(testing_data, testing_labels)
)

# Save trained model
model.save_weights(r'model.h5')
