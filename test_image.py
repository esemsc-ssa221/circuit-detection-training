from PIL import Image
import os

image_folder = '/home/idacy/development/circuit-detection-training/images'
for filename in os.listdir(image_folder):
    try:
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path)
        img.verify()  # Verify the image integrity
        print(f"{filename} is valid.")
    except Exception as e:
        print(f"{filename} could not be opened: {e}")
