from tensorflow.keras.models import load_model
from prepare_dataset import load_data

X_test, y_test = load_data("test")
model = load_model("cnn_telur_mobilenet.h5")

loss, acc = model.evaluate(X_test, y_test)

print("Test Loss :", loss)
print("Test Accuracy :", acc)