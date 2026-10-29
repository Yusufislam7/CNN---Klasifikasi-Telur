import os
import cv2
import numpy as np

IMG_SIZE = 224

CLASSES = ["arab", "horn", "kampung"]

def load_data(split):
    X, y = [], []

    base_dir = split  # train / valid / test

    for idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(base_dir, class_name)

        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(class_dir, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            X.append(img)
            y.append(idx)

    return np.array(X), np.array(y)
