import uvicorn
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from PIL import Image
import io
import numpy as np
import os
# Charger le modèle TensorFlow pré-entraîné
script_dir = os.path.dirname(__file__)
model = tf.keras.models.load_model(os.path.join (script_dir,
                                                 'models',
                                                 'model_ubnet_MobileNetV3Large.h5'))

# Créer l'instance de l'application FastAPI
app = FastAPI()

# Indice des classes
idx_class = ['Covid', 'Lung Opacity', 'Normal', 'Viral Pneumonia']

# Définir le point de terminaison pour l'API
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Lire l'image à partir du fichier envoyé
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Prétraitement de l'image
    # Assurez-vous que l'image est redimensionnée à la taille attendue par le modèle (256x256x3)
    image = image.resize((256, 256))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)  # Ajouter une dimension de lot (batch dimension)
    
    # Faire une prédiction avec le modèle TensorFlow
    prediction = model.predict(image_array)[0]

    idx_max_pred = prediction.argmax()
    print(prediction,idx_max_pred)
    # Convertir la prédiction en chaîne de caractères (à adapter selon votre modèle)
    return {'prediction': f'{idx_class[idx_max_pred]}',"proba": f"{prediction[idx_max_pred]}"}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
