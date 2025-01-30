import tensorflow as tf
import numpy as np
import cv2
import os
from train import get_data
import json

# Cargar el modelo entrenado
classifier = tf.keras.models.load_model('classifier.keras')

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
data_test = []
true_labels = []

for archivo in lista_archivos_test:
    path_archivo_test = os.path.join(path_base_datos, archivo)
    imagen = cv2.imread(path_archivo_test, cv2.IMREAD_GRAYSCALE)
    imagen = cv2.resize(imagen, (dim0, dim1))
    data_test.append(imagen)
    # Obtener la clase del nombre del archivo (primer directorio)
    clase = os.path.basename(os.path.dirname(path_archivo_test))
    true_labels.append(clase)

# Convertir las listas a arrays de numpy y normalizar
data_test = np.array(data_test).astype('float32') / 255.0
data_test = data_test.reshape((-1, dim0, dim1, 1))

# Realizar predicciones
predicciones = classifier.predict(data_test)
clases_predichas = np.argmax(predicciones, axis=1)

# Obtener nombres de clases ordenados (mismo orden que en el entrenamiento)
clases = sorted(os.listdir(path_base_datos))

# Convertir índices a nombres de clases
predicciones_clases = [clases[idx] for idx in clases_predichas]

# Calcular métricas
correct = sum(1 for true, pred in zip(true_labels, predicciones_clases) if true == pred)
accuracy = correct / len(true_labels)

# Crear matriz de confusión
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(true_labels, predicciones_clases, labels=clases)

# Guardar resultados
resultados = {
    'accuracy': accuracy,
    'confusion_matrix': cm.tolist(),
    'classification_report': classification_report(true_labels, predicciones_clases, target_names=clases, output_dict=True)
}

# Crear directorio de resultados si no existe
if not os.path.exists('resultados'):
    os.makedirs('resultados')

# Guardar resultados en formato JSON
with open('resultados/metricas_clasificacion.json', 'w') as f:
    json.dump(resultados, f, indent=4)

# Imprimir resultados principales
print(f"\nPrecisión en el conjunto de test: {accuracy:.4f}")
print("\nInforme de clasificación:")
print(classification_report(true_labels, predicciones_clases, target_names=clases))

# Guardar algunas predicciones de ejemplo
for i in range(min(10, len(data_test))):  # Guardar las primeras 10 imágenes
    imagen = (data_test[i] * 255).reshape(dim0, dim1).astype(np.uint8)
    # Agregar texto con la predicción y la etiqueta verdadera
    imagen_con_texto = cv2.putText(
        cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR),
        f"Pred: {predicciones_clases[i]}", 
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )
    imagen_con_texto = cv2.putText(
        imagen_con_texto,
        f"True: {true_labels[i]}", 
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1
    )
    cv2.imwrite(f'resultados/test_{i}_clasificacion.png', imagen_con_texto)
