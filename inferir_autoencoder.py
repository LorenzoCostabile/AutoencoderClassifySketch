import tensorflow as tf
import numpy as np
import cv2
import os
from train import get_data  # Reutilizamos la función de carga de datos

# Cargar el modelo entrenado
autoencoder = tf.keras.models.load_model('autoencoder.keras')

# Parámetros
dim0 = 320
dim1 = 320
path_base_datos = "imagenes"

# Cargar lista de archivos de test
lista_archivos_test = []
with open("splits/test.txt", "r") as f:
    for linea in f:
        lista_archivos_test.append(linea.strip())

# Obtener datos de test
data_test, _ = get_data(path_base_datos, lista_archivos_test, [], dim0, dim1)

# Evaluar el modelo en el conjunto de test
test_loss = autoencoder.evaluate(data_test, data_test)
print(f"Loss en el conjunto de test: {test_loss}")

# Generar algunas reconstrucciones para visualización
reconstrucciones = autoencoder.predict(data_test)

# Guardar algunas imágenes originales y sus reconstrucciones
if not os.path.exists('resultados'):
    os.makedirs('resultados')

for i in range(min(5, len(data_test))):  # Guardar las primeras 5 imágenes
    # Imagen original
    original = (data_test[i] * 255).reshape(dim0, dim1).astype(np.uint8)
    cv2.imwrite(f'resultados/test_{i}_original.png', original)
    
    # Imagen reconstruida
    reconstruida = (reconstrucciones[i] * 255).reshape(dim0, dim1).astype(np.uint8)
    cv2.imwrite(f'resultados/test_{i}_reconstruida.png', reconstruida)
