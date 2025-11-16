import cv2
from deepface import DeepFace
import numpy as np
import os
import sys

# --- Configuración ---
DB_PATH = "Fotos_referencia"
UMBRAL_DISTANCIA = 0.50 # Usado por DeepFace para la comparación

def iniciar_reconocimiento():
    """Inicia la cámara y realiza el reconocimiento facial en tiempo real."""

    if not os.path.exists(DB_PATH):
        print(f"ERROR: La carpeta '{DB_PATH}' no existe.")
        sys.exit()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: No se puede acceder a la cámara.")
        sys.exit()

    print("Cámara iniciada. Presiona 'q' para salir.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Estado por defecto
        estado_global = "Esperando rostro..."
        rostros_para_dibujar = []

        try:
            # 1. Intentar detectar rostros y verificar la identidad en un solo paso
            # DeepFace.find es la función más completa para esto.
            resultados_df = DeepFace.find(
                img_path=frame, 
                db_path=DB_PATH, 
                model_name='VGG-Face', 
                enforce_detection=False, # No falla si no encuentra cara
                threshold=UMBRAL_DISTANCIA
            )
            
            # Recorrer los resultados (un DataFrame por cada rostro detectado)
            if resultados_df and isinstance(resultados_df, list):
                for df in resultados_df:
                    if df.empty:
                        # Si DeepFace detectó un rostro pero no encontró coincidencias por umbral,
                        # no podemos extraer coordenadas del DataFrame de 'find', lo ignoramos.
                        continue
                    
                    # Extraer coordenadas del rostro detectado (DeepFace las devuelve)
                    x = df.iloc[0]['source_x']
                    y = df.iloc[0]['source_y']
                    w = df.iloc[0]['source_w']
                    h = df.iloc[0]['source_h']
                    
                    nombre = "Desconocido"
                    color_rec = (0, 0, 255) # Rojo: Detectado pero no reconocido

                    # 2. Verificar si el rostro fue reconocido (hay coincidencias)
                    if not df.empty and 'is_verified' in df.columns:
                         # DeepFace 0.0.95 no siempre devuelve 'is_verified' en find()

                        # Simplificación: Si el DataFrame tiene coincidencias, tomamos la mejor
                        if not df.empty:
                            identidad_completa = df.iloc[0]['identity']
                            nombre = os.path.basename(os.path.dirname(identidad_completa))
                            color_rec = (0, 255, 0) # Verde: Reconocido

                    elif not df.empty and 'identity' in df.columns:
                        # Si hay una identidad en la primera fila, significa que hubo coincidencia.
                        identidad_completa = df.iloc[0]['identity']
                        nombre = os.path.basename(os.path.dirname(identidad_completa))
                        color_rec = (0, 255, 0) # Verde: Reconocido
                    
                    # Guardar la información para dibujar
                    rostros_para_dibujar.append({
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'nombre': nombre, 
                        'color': color_rec
                    })

        except Exception as e:
            # Si hay un error de procesamiento, simplemente pasamos al siguiente frame
            pass
        
        # 3. Dibujar el cuadro y el texto para CADA rostro detectado
        if rostros_para_dibujar:
            estado_global = "" # Limpiamos el mensaje si hay caras para dibujar
            for rostro in rostros_para_dibujar:
                x, y, w, h = rostro['x'], rostro['y'], rostro['w'], rostro['h']
                nombre = rostro['nombre']
                color_rec = rostro['color']
                
                # Dibujar el rectángulo alrededor de la cara
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_rec, 2)
                
                # Dibujar el nombre sobre el cuadro
                cv2.putText(frame, nombre, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_rec, 2)
        else:
             # Si no hay rostros para dibujar, mostramos el mensaje de espera
             cv2.putText(frame, estado_global, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        # Mostrar el fotograma con las anotaciones
        cv2.imshow('Reconocimiento Facial con DeepFace', frame)

        # Romper el bucle al presionar 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    # Liberar la cámara y cerrar todas las ventanas
    cap.release()
    cv2.destroyAllWindows()
    print("Programa finalizado.")

if __name__ == "__main__":
    iniciar_reconocimiento()