import tensorflow as tf
from tensorflow.keras import layers, models
import os
import json
from pathlib import Path

# Assuming you have the path to the main directory containing subdirectories for each class
data_dir = './dataset'

# Define image parameters
batch_size = 64
img_height = 480
img_width = 480
num_classes = len(os.listdir(data_dir))  # Number of classes based on subdirectories

# Create the image dataset from the directory
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

with open('classnames-v2.json', 'w') as fp:
    json.dump(train_ds.class_names, fp)


# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Define your CNN model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model with the appropriate loss function and optimizer
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define the EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',  # Metric to monitor for early stopping
    patience=5,              # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restores the weights of the best epoch when training is stopped
)

# Train the model using the dataset
history = model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[early_stopping])

# Evaluate the model on the test set or validation set
test_loss, test_accuracy = model.evaluate(val_ds)
print(f'Test accuracy: {test_accuracy}')

model_path = Path('models')
os.makedirs(model_path)

model.save(str(model_path.joinpath('v2')))
