import cv2
from mtcnn import MTCNN
import os
from pathlib import Path


for label, sports in {
    'baseball': ['mlb', 'npb'],
    'racing': ['formula1', 'motogp'],
    'basketball': ['nba'],
    'esports': ['lolesports'],
}.items():
    output_folder = Path('dataset').joinpath(f'class_{label}')
    os.makedirs(output_folder, exist_ok=True)

    for sport in sports:
        input_folder = Path('outputs').joinpath(sport)

        # Initialize MTCNN face detector
        detector = MTCNN()

        for image_file in input_folder.iterdir():
            if not image_file.is_file():
                continue

            image = cv2.imread(str(image_file.absolute()))
            
            # Detect faces using MTCNN
            results = detector.detect_faces(image)
            if len(results) != 1:
                continue

            x, y, w, h = results[0]['box']
            cropped = image[y:y+h, x:x+w]

            cropped = cv2.resize(cropped, (480, 480))

            output_path = str(output_folder.joinpath(f'{label}_{sport}_{image_file.name}').absolute())
            cv2.imwrite(output_path, cropped)
