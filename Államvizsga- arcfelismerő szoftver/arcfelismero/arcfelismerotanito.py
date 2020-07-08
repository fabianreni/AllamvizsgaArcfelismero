import cv2
import os
import numpy as np
from PIL import Image
import pickle

def tanito():
    # konyvtár elérés
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "images")
    # arc detektáló beolvasás
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    # felismerő létrehozás
    recognizer = cv2.face.LBPHFaceRecognizer_create()   
    #tombok és azonositó létrehozás 
    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []
    # kép mapparendszer bejárás
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                # kép kiolvasás
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                # azonosító beálítás
                id_ = label_ids[label]
                pil_image = Image.open(path).convert("L")  # szürke árnyalat
                image_array = np.array(pil_image, "uint8")
                # arc detektáló inicializálás
                faces = face_cascade.detectMultiScale(
                    image_array, scaleFactor=1.3, minNeighbors=5)
                for (x, y, w, h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)
    # azonoítok kimentése
    with open("labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)
    # felismerő tanítása
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainner.yml")
    print("Kesz!")
# tanito()
