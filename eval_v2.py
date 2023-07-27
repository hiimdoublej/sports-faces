import cv2
import tensorflow as tf
from pathlib import Path
import click
import mtcnn
import json

face_detector = mtcnn.MTCNN()


@click.command()
@click.argument('inputs', nargs=-1)
@click.option('--model', default='v2')
def predict(inputs, model):
    # Load model
    model_path = Path('models').joinpath(model)
    model = tf.keras.models.load_model(str(model_path.absolute()))

    def _results():
        for input in sorted(inputs):
            # Read the image in BGR format (opencv default)
            image = cv2.imread(input)
            
            # Convert the image to RGB format (if needed)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect faces using MTCNN
            try:
                face = face_detector.detect_faces(image)[0]
            except Exception:
                click.echo(f'Cannot detect face in file: {input}', err=True)
                continue

            x, y, w, h = face['box']
            cropped = image[y:y+h, x:x+w]
            cropped = cv2.resize(cropped, (480, 480))

            img_array = tf.expand_dims(cropped, 0) # Create a batch

            with open('classnames-v2.json') as fp:
                classnames = json.load(fp)

            predictions = model.predict(img_array)
            prediction = dict(zip(
                classnames,
                map(lambda x: x * 100, predictions[0]),
            ))

            yield input, prediction

    results = list(_results())
    click.echo('=========================')
    for input, prediction in results:
        click.echo(f'Predictions for {input}:')
        click.echo(json.dumps(prediction, indent=2))
        click.echo('=========================')

if __name__ == '__main__':
    predict()

