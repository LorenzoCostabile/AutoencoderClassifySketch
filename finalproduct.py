import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import json

def load_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    files, labels = zip(*[line.strip().split() for line in lines])
    return list(files), list(labels)

def main():
    test_file = "splits/test.txt"
    model_path = "model.h5"
    output_labels = "etiquetas_estimadas.txt"
    output_matrix = "matriz_confusion.txt"

    test_files, test_labels = load_data(test_file)
    model = tf.keras.models.load_model(model_path)

    y_true = []
    y_pred = []
    label_map = {v: k for k, v in model.class_indices.items()}

    with open(output_labels, "w") as f:
        for file, label in zip(test_files, test_labels):
            image = load_img(file, target_size=(128, 128))
            image = img_to_array(image) / 255.0
            image = np.expand_dims(image, axis=0)
            prediction = model.predict(image)
            predicted_label = label_map[np.argmax(prediction)]
            f.write(f"{file} {predicted_label}\\n")
            y_true.append(label)
            y_pred.append(predicted_label)

    cm = confusion_matrix(y_true, y_pred, labels=list(label_map.values()))
    with open(output_matrix, "w") as f:
        f.write(json.dumps(cm.tolist()))

if __name__ == "__main__":
    main()