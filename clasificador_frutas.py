import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- 1. CONFIGURACIÓN ---
# Parámetros de la imagen y el entrenamiento
IMG_WIDTH, IMG_HEIGHT = 128, 128 
BATCH_SIZE = 32
EPOCHS = 10 
DATA_DIR = 'dataset' # ¡Asegúrate que esta carpeta exista y contenga las subcarpetas de frutas!

print(f"TensorFlow Version: {tf.__version__}")

# --- 2. PREPARACIÓN Y AUMENTO DE DATOS ---
# ImageDataGenerator ayuda a cargar imágenes, normalizar píxeles (rescale=1./255) y crear variaciones.
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,       
    horizontal_flip=True,    
    validation_split=0.2     # 20% de las imágenes se usan para validación
)

print("\n--- Generando flujos de datos ---")

# Flujo de datos de ENTRENAMIENTO
train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical', # Clasificación multi-clase
    subset='training'
)

# Flujo de datos de VALIDACIÓN
validation_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

CLASS_NAMES = list(train_generator.class_indices.keys())
NUM_CLASSES = len(CLASS_NAMES)
print(f"Clases detectadas: {CLASS_NAMES}")

# --- 3. CONSTRUCCIÓN DEL MODELO CNN ---
model = Sequential([
    # Bloque 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D((2, 2)),

    # Bloque 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Bloque 3
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Capas Densa (Clasificación)
    Flatten(),
    Dense(512, activation='relu'),
    
    # Capa de Salida
    Dense(NUM_CLASSES, activation='softmax')
])

# --- 4. COMPILACIÓN Y ENTRENAMIENTO ---
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("\n--- COMENZANDO EL ENTRENAMIENTO ---")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# --- 5. GUARDAR EL MODELO ---
MODEL_FILE = 'modelo_clasificador_frutas.h5'
model.save(MODEL_FILE)
print(f"\nModelo entrenado y guardado como: {MODEL_FILE}")