import cv2
from pathlib import Path
import json
import os
import pickle
import click
import mtcnn

# Load the pre-trained MTCNN face detection model
face_detector = mtcnn.MTCNN()

def load():
    # Load and preprocess your labeled image dataset
    image_file_paths = []  # List of file paths to your images
    labels = []  # List of corresponding labels for each image
    label_ref = {}

    for idx, path in enumerate(sorted(Path('./outputs/').iterdir())):
        label_ref[path.name] = idx
        files = [
            p for p in path.iterdir() if p.is_file()
        ]
        image_file_paths.extend([
            str(f.absolute()) for f in files
        ])
        labels.extend([idx] * len(files))
    
    return image_file_paths, labels, label_ref


def _pre_process(image_file_paths, labels, width, height):
    for image_path, label in zip(image_file_paths, labels):
        # Read the image in BGR format (opencv default)
        image = cv2.imread(image_path)
        
        # Convert the image to RGB format (if needed)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces using MTCNN
        # Only use images with exactly 1 face
        faces = face_detector.detect_faces(image)
        if len(faces) != 1:
            continue

        face = faces[0]

        x, y, w, h = face['box']
        image = image[y:y+h, x:x+w]

        # Resize the image to the desired dimensions
        image = cv2.resize(image, (height, width))
        
        # Normalize pixel values to [0, 1]
        image = image / 255.0

        yield image, label
    

def dump_data(data_iter, variation):
    data_dir = Path(f'variations/{variation}/data')
    label_dir = Path(f'variations/{variation}/labels')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for idx, (d, label) in enumerate(data_iter):
        with open(os.path.join(data_dir, str(idx)), 'wb') as fp:
            pickle.dump(d, fp)
        
        with open(os.path.join(label_dir, str(idx)), 'wb') as fp:
            pickle.dump(label, fp)


def dump(label_ref, variation):
    variation_dir = Path(f'variations/{variation}')
    variation_dir.mkdir(exist_ok=True)

    with open(os.path.join(variation_dir, 'label_ref.json'), 'w') as fp:
        json.dump(label_ref, fp)
    

@click.command()
@click.option('--width', default=320)
@click.option('--height', default=320)
@click.option('--channels', default=3)
def pre_process(width, height, channels):
    variation = f'{width}x{height}x{channels}'
    image_file_paths, labels, label_ref = load()
    dump_data(_pre_process(image_file_paths, labels, width, height), variation=variation)
    dump(label_ref, variation)


if __name__ == '__main__':
    pre_process()

