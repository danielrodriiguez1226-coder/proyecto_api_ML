# 🥤 Proyecto YOLO: Detección de Botellas, Latas y Marcadores

¡Bienvenido! Este proyecto utiliza **YOLOv8** para detectar botellas, latas y marcadores en imágenes. Ideal para entrenamientos rápidos y pruebas de detección. 🚀

---

## 📂 Contenido del Repositorio

- `config/` → Configuración del dataset  
- `data/` → Imágenes y etiquetas para entrenamiento y validación  
- `scripts/` → Scripts para generar datasets y entrenar el modelo  
- `runs/` → Resultados de entrenamientos e inferencias  
- `yolov8n.pt` → Pesos base del modelo YOLOv8 (pre-entrenado)

---

## 🎯 Entrenamiento

El modelo se entrenó con **10 épocas** para pruebas iniciales.  
Puedes entrenar tu modelo así:

```bash
python scripts/train.py --data config/data.yaml --weights yolov8n.pt --epochs 10
