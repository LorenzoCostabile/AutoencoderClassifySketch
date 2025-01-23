import os
import random
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt

def write_to_file(file_path, data):
    with open(file_path, 'w') as file:
        for item in data:
            file.write(item + '\n')

def plot_porcentaje_clases(train_files, val_files, test_files, clases, output_dir):
    porcentaje_clases_train = {}
    porcentaje_clases_val = {}
    porcentaje_clases_test = {}
    for clase in clases:
        porcentaje_clases_train[clase] = len([x for x in train_files if x[1] == clase]) / len(train_files)
        porcentaje_clases_val[clase] = len([x for x in val_files if x[1] == clase]) / len(val_files)
        porcentaje_clases_test[clase] = len([x for x in test_files if x[1] == clase]) / len(test_files)

    plt.figure(figsize=(10, 6))
    x = range(len(clases))
    width = 0.25
    
    plt.bar([i - width for i in x], [porcentaje_clases_train[clase] for clase in clases], width, label='Train')
    plt.bar(x, [porcentaje_clases_val[clase] for clase in clases], width, label='Validation')
    plt.bar([i + width for i in x], [porcentaje_clases_test[clase] for clase in clases], width, label='Test')
    
    plt.xlabel('Clases')
    plt.ylabel('Porcentaje')
    plt.title('Porcentaje de clases en cada conjunto')
    plt.xticks(x, clases, rotation=90)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(output_dir, "porcentaje_clases.png"))
    plt.close()

def crear_splits(path_dataset, train_percentage, validation_percentage):

    test_percentage = 1 - train_percentage - validation_percentage
    if test_percentage < 0:
        raise ValueError("La suma de los porcentajes de entrenamiento y validación no puede ser mayor que 1")

    clases = os.listdir(path_dataset)
    all_files_con_clase = []
    for clase in clases:
        files = os.listdir(os.path.join(path_dataset, clase))
        all_files_con_clase.extend([(os.path.join(clase, f), clase) for f in files])

    percentage_val_test = test_percentage + validation_percentage

    percentage_test_referent_val = test_percentage / (test_percentage + validation_percentage)

    train_files, test_val_files = train_test_split(all_files_con_clase, test_size=percentage_val_test, stratify=[x[1] for x in all_files_con_clase], random_state=42)
    test_files, val_files = train_test_split(test_val_files, test_size=percentage_test_referent_val, stratify=[x[1] for x in test_val_files], random_state=42)

    return train_files, val_files, test_files, clases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into train, validation, and test sets")
    parser.add_argument("--path_dataset", type=str, default="imagenes", help="Ruta del dataset, la carpeta con las clases")
    parser.add_argument("--path_salida", type=str, default="splits", help="Ruta donde se crearan los archivos train, validation y test")
    parser.add_argument("--train_percentage", type=float, default=0.8, help="Porcentaje de datos para el conjunto de entrenamiento")
    parser.add_argument("--validation_percentage", type=float, default=0.1, help="Porcentaje de datos para el conjunto de validacion")
    parser.add_argument("--visualizar", type=bool, default=False, help="Si se desea visualizar los porcentajes de clases en cada conjunto")
    args = parser.parse_args()

    # Creamos la carpeta de salida si no existe
    os.makedirs(args.path_salida, exist_ok=True)

    if args.train_percentage + args.validation_percentage > 1:
        raise ValueError("La suma de los porcentajes de entrenamiento y validación no puede ser mayor que 1")
    
    train_files, val_files, test_files, clases = crear_splits(args.path_dataset, args.train_percentage, args.validation_percentage)
    if args.visualizar:
        plot_porcentaje_clases(train_files, val_files, test_files, clases, args.path_salida)

    write_to_file(os.path.join(args.path_salida, "train.txt"), [x[0] for x in train_files])
    write_to_file(os.path.join(args.path_salida, "val.txt"), [x[0] for x in val_files])
    write_to_file(os.path.join(args.path_salida, "test.txt"), [x[0] for x in test_files])
