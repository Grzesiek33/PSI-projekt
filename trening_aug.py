import tensorflow as tf
import tensorflow_datasets as tfds

# Parametry
BATCH_SIZE = 128
EPOCHS = 10
MODEL_PATH = "emnist_model_aug_updated2.h5"               # <-- Model wejściowy
OUTPUT_MODEL_PATH = "emnist_model_aug_updated3.h5"  # <-- Model po kontynuacji

# Wczytanie modelu
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model załadowany.")

# Preprocessing danych
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)
    return image, label

# Ładowanie danych
(ds_train, ds_test), info = tfds.load(
    'emnist/byclass',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomZoom(0.1, 0.1),
])

def augment(image, label):
    image = data_augmentation(image)
    return image, label

# Oryginalny zbiór treningowy
ds_train_orig = ds_train.map(preprocess)

# Zaugmentowany zbiór treningowy
ds_train_aug = ds_train_orig.map(augment)

# Połączenie obu zbiorów
ds_train = ds_train_orig.concatenate(ds_train_aug)
ds_train = ds_train.shuffle(20000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Kompilacja modelu (jeśli nie był wcześniej skompilowany)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Kontynuacja treningu
print("🔁 Kontynuacja trenowania...")
model.fit(ds_train, epochs=EPOCHS, validation_data=ds_test)

# Zapis modelu
model.save(OUTPUT_MODEL_PATH)
print(f"💾 Zapisano zaktualizowany model jako {OUTPUT_MODEL_PATH}")