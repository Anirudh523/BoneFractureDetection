import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

img_height, img_width = 150, 150
num_classes = 6  # Number of fracture types

# Load your dataset
# Replace 'path_to_images' and 'labels.npy' with your actual paths
images = np.load(r'C:\Users\Anirudh\Documents\GitHub\BoneFractureDetection\BoneFractureData2\bone fracture detection.v4-v4.yolov8\test\images')  # Shape should be (num_samples, img_height, img_width, num_channels)
labels = np.load(r'C:\Users\Anirudh\Documents\GitHub\BoneFractureDetection\BoneFractureData2\bone fracture detection.v4-v4.yolov8\train\labels')  # Shape should be (num_samples,) with integer labels

# Preprocess images: normalize pixel values to [0, 1]
images = images / 255.0

# Split data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the model using Functional API
inputs = tf.keras.Input(shape=(img_height, img_width, 3))

x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])  # Use generic accuracy metric

# Print the model summary
model.summary()

# Train the model
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=10,
                    batch_size=32)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(x_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")