import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
import os
import cv2

img_height, img_width = 150, 150
num_classes = 6  # Number of fracture types

# Function to load and preprocess images
def load_images_from_folder(folder, exclude_indices=None):
    images = []
    filenames = []
    for idx, filename in enumerate(os.listdir(folder)):
        if exclude_indices and idx in exclude_indices:
            continue
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            resized_img = cv2.resize(img, (img_height, img_width))
            images.append(resized_img)
            filenames.append(filename)
        else:
            print(f"Failed to load image: {img_path}")
    return np.array(images), filenames

# Function to load labels from folder
def load_labels_from_folder(folder, filenames):
    labels = []
    exclude_indices = []
    for idx, filename in enumerate(filenames):
        label_path = os.path.join(folder, filename.replace('.jpg', '.txt'))  # Assuming label files are .txt
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                label_data = file.readline().strip().split()
                if label_data:  # Check if label_data is not empty
                    label = np.array(label_data, dtype=np.float32)
                    if label[0] < num_classes:
                        labels.append(label[0])  # Assuming the first value is the class label
                    else:
                        print(f"Invalid label {label[0]} for image {filename}")
                        exclude_indices.append(idx)
                else:
                    print(f"Empty label file for image {filename}")
                    exclude_indices.append(idx)
        else:
            print(f"Label not found for image: {filename}")
            exclude_indices.append(idx)
    return np.array(labels), exclude_indices

# Paths to your dataset
train_img_folder = r'C:\Users\Anirudh\Documents\GitHub\BoneFractureDetection\BoneFractureData2\BoneFractureYolo8\train\images'
train_labels_folder = r'C:\Users\Anirudh\Documents\GitHub\BoneFractureDetection\BoneFractureData2\BoneFractureYolo8\train\labels'

val_img_folder = r'C:\Users\Anirudh\Documents\GitHub\BoneFractureDetection\BoneFractureData2\BoneFractureYolo8\valid\images'
val_labels_folder = r'C:\Users\Anirudh\Documents\GitHub\BoneFractureDetection\BoneFractureData2\BoneFractureYolo8\valid\labels'

test_img_folder = r'C:\Users\Anirudh\Documents\GitHub\BoneFractureDetection\BoneFractureData2\BoneFractureYolo8\test\images'
test_labels_folder = r'C:\Users\Anirudh\Documents\GitHub\BoneFractureDetection\BoneFractureData2\BoneFractureYolo8\test\labels'

# Load training data
train_filenames = os.listdir(train_img_folder)
y_train, train_exclude_indices = load_labels_from_folder(train_labels_folder, train_filenames)
x_train, train_filenames = load_images_from_folder(train_img_folder, train_exclude_indices)

# Load validation data
val_filenames = os.listdir(val_img_folder)
y_val, val_exclude_indices = load_labels_from_folder(val_labels_folder, val_filenames)
x_val, val_filenames = load_images_from_folder(val_img_folder, val_exclude_indices)

# Load test data
test_filenames = os.listdir(test_img_folder)
y_test, test_exclude_indices = load_labels_from_folder(test_labels_folder, test_filenames)
x_test, test_filenames = load_images_from_folder(test_img_folder, test_exclude_indices)

# Preprocess images: normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

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

# Test the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")