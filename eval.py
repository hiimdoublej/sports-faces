import tensorflow as tf
import numpy as np
from pathlib import Path
import os
import click
import cv2
import json
import mtcnn

face_detector = mtcnn.MTCNN()


@click.command()
@click.argument('input')
@click.option('--variation', default='320x320x3')
def predict(input, variation):
    width, height, _ = variation.split('x', 2)

    # Read the image in BGR format (opencv default)
    image = cv2.imread(input)
    
    # Convert the image to RGB format (if needed)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces using MTCNN
    try:
        face = face_detector.detect_faces(image)[0]
    except Exception:
        click.echo('Cannot detect face in input, do you have face ?', err=True)

    x, y, w, h = face['box']
    image = image[y:y+h, x:x+w]

    # Resize the image to the desired dimensions
    image = cv2.resize(image, (height, width))
    
    # Normalize pixel values to [0, 1]
    image = image / 255.0

    variation_dir = Path(f'variations/{variation}')
    model = tf.keras.models.load_model(os.path.join(variation_dir, 'model'))
    # Assuming 'new_images' are the new, unseen images for which you want predictions
    predictions = model.predict(np.array([image]))

    # 'predictions' will contain the probabilities for each class. You can get the predicted class as follows:
    with open(os.path.join(variation_dir, 'label_ref.json'), 'rb') as fp:
        translate_table = json.load(fp)
    
    translate_table = {
        v: k for k, v in translate_table.items()
    }

    results = {
        translate_table[idx]: v for idx, v in enumerate(predictions[0])
    }
    click.echo(predictions[0])
    click.echo(results)

if __name__ == '__main__':
    predict()

