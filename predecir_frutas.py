import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# --- CONFIGURACIÓN (Debe coincidir con el entrenamiento) ---
MODEL_PATH = 'modelo_clasificador_frutas.h5'
IMG_WIDTH, IMG_HEIGHT = 128, 128 

# ¡AJUSTA ESTO A TUS CLASES REALES y ASEGÚRATE DE QUE EL ORDEN SEA EL MISMO QUE EL DETECTADO EN EL ENTRENAMIENTO!
CLASS_NAMES = ['manzana', 'naranja', 'platano'] 

# --- 1. CARGAR EL MODELO ENTRENADO ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Modelo cargado correctamente desde {MODEL_PATH}")
except Exception as e:
    print(f"ERROR: No se pudo cargar el modelo. Asegúrate de que el archivo {MODEL_PATH} existe.")
    print(e)
    exit()

# --- 2. FUNCIÓN DE PREDICCIÓN ---
def predecir_imagen(img_path):
    # Cargar y preprocesar la imagen
    img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    
    # Expandir la dimensión para que el modelo la acepte (1 imagen, 128x128, 3 canales)
    img_array = np.expand_dims(img_array, axis=0) 
    
    # Normalizar (dividir por 255)
    img_array /= 255.0  

    # Hacer la predicción
    predictions = model.predict(img_array)

    # Obtener el índice de la clase con la mayor probabilidad
    predicted_class_index = np.argmax(predictions[0])
    
    # Obtener el nombre de la clase y la probabilidad
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100

    return predicted_class, confidence

# --- 3. PRUEBA ---
IMAGEN_DE_PRUEBA = 'ruta/a/tu/imagen_prueba.jpg' # <<<<<< ¡CAMBIA ESTA RUTA POR UNA IMAGEN REAL!
print(f"\nPrediciendo: {IMAGEN_DE_PRUEBA}")

# Ejecutar la predicción
clase, confianza = predecir_imagen(IMAGEN_DE_PRUEBA)

print(f"Resultado: La imagen es una **{clase}**.")
print(f"Confianza: {confianza:.2f} %")