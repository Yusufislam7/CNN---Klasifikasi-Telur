import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from prepare_dataset import load_data

X_train, y_train = load_data("train")
X_val, y_val = load_data("valid")

base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,      
    weights="imagenet"
)

base_model.trainable = False  # transfer learning

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(3, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=16
)

model.save("cnn_telur_mobilenet.h5")
print("MODEL MOBILENET DISIMPAN")