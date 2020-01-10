import requests
import cv2
import numpy as np

img: np.ndarray = cv2.imread(
    r'F:\PycharmProjects\Hello-Object-Detection\data\PlantVillage-Dataset-master\raw\color\Apple___Cedar_apple_rust\0cd24b0c-0a9d-483f-8734-5c08988e029f___FREC_C.Rust 3762.JPG')[
                  ..., ::-1]


with requests.post(f'http://127.0.0.1:6007/plant', json=img.tolist()) as url:
    print(url.text)