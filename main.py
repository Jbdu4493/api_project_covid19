import uvicorn
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from PIL import Image
import io
import numpy as np
import os
from io import BytesIO 
from grad_cam import grad_cam_fonction
import base64
import logging
from datetime import datetime
import pytz

# Charger le modèle TensorFlow pré-entraîné


script_dir = os.path.dirname(__file__)
model = tf.keras.models.load_model(os.path.join (script_dir,
                                                 'models',
                                                 'model_ubnet_MobileNetV3Large.h5'))

logging.basicConfig(level=logging.DEBUG)


desired_timezone = pytz.timezone('Europe/Paris')

# Créer un objet de date et d'heure au fuseau horaire actuel
current_time = datetime.now()

# Convertir la date et l'heure au fuseau horaire désiré
localized_time = current_time.astimezone(desired_timezone)

# Créer un objet Formatter avec le format souhaité
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Configurer le Formatter avec l'objet de date et d'heure localisée
formatter.converter = lambda *args: localized_time.timetuple()

# Créer un formateur de messages
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

timestamp = datetime.now().strftime('%Y-%m-%d%H%M%S')

# Créer un gestionnaire de sortie vers un fichier .log
file_handler = logging.FileHandler(os.path.join(script_dir,f'app_{timestamp}.log'))
file_handler.setFormatter(formatter)

# Ajouter le gestionnaire à la racine du logger
logger = logging.getLogger('')
logger.addHandler(file_handler)

# Créer l'instance de l'application FastAPI
app = FastAPI()

# Indice des classes
idx_class = ['Covid', 'Lung Opacity', 'Normal', 'Viral Pneumonia']


# Définir le point de terminaison pour l'API
@app.post('/covid19_detect/predict')
async def predict(file: UploadFile = File(...)):
    # Lire l'image à partir du fichier envoyé
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Prétraitement de l'image
    # Assurez-vous que l'image est redimensionnée à la taille attendue par le modèle (256x256x3)
    image = image.resize((256, 256))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array_ready_pred = tf.expand_dims(image_array, 0)  # Ajouter une dimension de lot (batch dimension)
    
    # Faire une prédiction avec le modèle TensorFlow
    prediction = model.predict(image_array_ready_pred)[0]

    idx_max_pred = prediction.argmax()

    # Calcul du grad-Cam
    grad_cam_image = grad_cam_fonction(image_array,model,2, "multiply_101",224)
    buffer = io.BytesIO()
    grad_cam_image.save(buffer, 'png')
    buffer.seek(0)
    
    data = buffer.read()
    data = base64.b64encode(data).decode()
    
    data_return = {'prediction': f'{idx_class[idx_max_pred]}',"proba": f"{prediction[idx_max_pred]}","grad_cam":data}
    logger.debug(f'---{data_return }')

    return data_return

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level="trace")
