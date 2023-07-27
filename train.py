import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
import json
import os
import pickle
import click


def _train(width, height, channels, variation):
    variation_dir = Path(f'variations/{variation}')

    with open(os.path.join(variation_dir, 'label_ref.json')) as fp:
        label_ref = json.load(fp)
    
    num_classes = len(label_ref)

    image_paths = np.array([
        str(x.absolute()) for x in sorted(variation_dir.joinpath('data').iterdir())
    ])
    label_paths = np.array([
        str(x.absolute()) for x in sorted(variation_dir.joinpath('labels').iterdir())
    ])
    # Define the batch size you want to use during training
    batch_size = 32

    # Function to load pickled images
    def load_pickled_images(image_paths):
        for image_path in image_paths:
            with open(image_path, 'rb') as file:
                image = pickle.load(file)
                # Normalize pixel values to [0, 1]
                image = image / 255.0
                yield image

    # Function to load pickled labels
    def load_pickled_labels(label_paths):
        for label_path in label_paths:
            with open(label_path, 'rb') as file:
                label = pickle.load(file)
                yield label

    # Load the file paths of pickled image and label data and split into training and validation sets
    train_image_paths, val_image_paths = train_test_split(image_paths, test_size=0.2, random_state=42)
    train_label_paths, val_label_paths = train_test_split(label_paths, test_size=0.2, random_state=42)

    # Create data generators for training and validation image data
    train_image_generator = load_pickled_images(train_image_paths)
    val_image_generator = load_pickled_images(val_image_paths)

    # Create data generators for training and validation label data
    train_label_generator = load_pickled_labels(train_label_paths)
    val_label_generator = load_pickled_labels(val_label_paths)

    # # Determine the input shape (height, width, channels) based on the loaded images
    input_shape = next(train_image_generator).shape[1:]

    # Define your CNN model architecture
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
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

    # Train the model using the data generators
    history = model.fit_generator(zip(train_image_generator, train_label_generator),
                        steps_per_epoch=len(train_image_paths) // batch_size,
                        epochs=100,
                        validation_data=(val_image_generator, val_label_generator),
                        validation_steps=len(val_image_paths) // batch_size)

    # Evaluate the model on the test set or validation set
    test_loss, test_accuracy = model.evaluate_generator(zip(val_image_generator, val_label_generator),
                                            steps=len(val_image_paths) // batch_size)
    print(f'Test accuracy: {test_accuracy}')

    return model

def save(variation, model):
    model.save(
        os.path.join('variations',variation, 'model')
    )

@click.command()
@click.option('--width', default=320)
@click.option('--height', default=320)
@click.option('--channels', default=3)
def train(width, height, channels):
    variation = f'{width}x{height}x{channels}'
    model = _train(width, height, channels, variation)
    save(variation, model)

if __name__ == '__main__':
    train()

