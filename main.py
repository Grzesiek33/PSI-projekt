import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

# Ustawienia
BATCH_SIZE = 128
EPOCHS = 10
MODEL_PATH = "emnist_model2.h5"

# Ładowanie EMNIST (byclass)
print("📥 Ładowanie zbioru EMNIST...")
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/byclass',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

NUM_CLASSES = ds_info.features['label'].num_classes

# Przetwarzanie danych
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0     # Normalizacja [0,1]
    image = tf.expand_dims(image, -1)              # Kanał: (28,28,1)
    return image, label

ds_train = ds_train.map(preprocess).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Tworzenie modelu CNN
def create_emnist_model(input_shape=(28, 28, 1), num_classes=NUM_CLASSES):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_better_emnist_model(input_shape=(28, 28, 1), num_classes=NUM_CLASSES):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = create_better_emnist_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Trenowanie
print("🚀 Rozpoczynanie trenowania...")
model.fit(ds_train, epochs=EPOCHS, validation_data=ds_test)

# Ewaluacja
print("📊 Ewaluacja na zbiorze testowym:")
test_loss, test_acc = model.evaluate(ds_test)
print(f"✅ Dokładność testowa: {test_acc * 100:.2f}%")

# Zapis modelu
model.save(MODEL_PATH)
print(f"💾 Model zapisany jako {MODEL_PATH}")
