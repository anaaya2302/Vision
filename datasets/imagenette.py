from fastai.data.external import untar_data, URLs
import os

# To download into the folder you are currently in:
path = untar_data(URLs.IMAGENETTE, dest='.') 

# Or to a specific custom path:
# path = untar_data(URLs.IMAGENETTE, dest='./my_datasets')

print(f"Dataset is at: {path}")
print(f"Training folder: {path}/train")

print(f"Validation folder: {path}/val")# List the 10 classes (folders)

classes = os.listdir(os.path.join(path, 'train'))

print(f"Classes: {classes}")