import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from prepare_dataset import load_data

CLASSES = [
    "telur ayam arab",
    "telur ayam kampung",
    "telur horn"
]

X_test, y_test = load_data("test")
model = load_model("cnn_telur.h5")

y_pred = np.argmax(model.predict(X_test), axis=1)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
