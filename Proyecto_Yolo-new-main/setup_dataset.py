from PIL import Image
import os

# Carpetas
train_img_dir = r'datasets/train/images'
train_lbl_dir = r'datasets/train/labels'
val_img_dir = r'datasets/val/images'
val_lbl_dir = r'datasets/val/labels'

# Crear carpetas si no existen
for folder in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
    os.makedirs(folder, exist_ok=True)

# Crear 2 imágenes de ejemplo para train
for i in range(1, 3):
    img = Image.new('RGB', (640, 640), color=(0, 0, 0))
    img.save(os.path.join(train_img_dir, f'img{i:03d}.jpg'))
    with open(os.path.join(train_lbl_dir, f'img{i:03d}.txt'), 'w') as f:
        f.write('0 0.5 0.5 0.5 0.5\n')  # clase 0, centro 0.5,0.5, ancho y alto 0.5

# Crear 1 imagen de ejemplo para val
img = Image.new('RGB', (640, 640), color=(0, 0, 0))
img.save(os.path.join(val_img_dir, 'img001.jpg'))
with open(os.path.join(val_lbl_dir, 'img001.txt'), 'w') as f:
    f.write('0 0.5 0.5 0.5 0.5\n')
