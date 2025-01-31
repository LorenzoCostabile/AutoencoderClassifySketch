import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
def create_classifier_head(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    
    # Normalización por lotes
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def get_encoder_from_autoencoder(autoencoder):
    input_layer = autoencoder.input
    code_layer = autoencoder.layers[8].output
    encoder = Model(inputs=input_layer, outputs=code_layer, name='encoder_model')
    
    # Congelar todas las capas del encoder
    for layer in encoder.layers:
        layer.trainable = False

        
    return encoder

def create_classifier(autoencoder, num_classes):
    encoder = get_encoder_from_autoencoder(autoencoder)
    classifier_head = create_classifier_head(encoder.output_shape[1:], num_classes)
    inputs = Input(shape=autoencoder.input_shape[1:])
    x = encoder(inputs)
    outputs = classifier_head(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def get_data(path_base_datos, list_archivos_entrenamiento, list_archivos_validacion, num_classes, dim0=None, dim1=None):
    data_train = []
    data_validacion = []
    labels_train = []
    labels_validacion = []
    dim0 = dim0
    dim1 = dim1

    for archivo in list_archivos_entrenamiento:
        path_archivo_entrenamiento = os.path.join(path_base_datos, archivo)
        imagen = cv2.imread(path_archivo_entrenamiento, cv2.IMREAD_GRAYSCALE)
        if dim0 is None:
            dim0, dim1 = imagen.shape[0], imagen.shape[1]
        imagen = cv2.resize(imagen, (dim0, dim1))
        data_train.append(imagen)
        # Obtener la clase del nombre del archivo (primer directorio)
        clase = os.path.basename(os.path.dirname(path_archivo_entrenamiento))
        # Convertir nombre de clase a índice numérico (asumiendo orden alfabético)
        label = sorted(os.listdir(path_base_datos)).index(clase)
        labels_train.append(label)
    
    for archivo in list_archivos_validacion:
        path_archivo_validacion = os.path.join(path_base_datos, archivo)
        imagen = cv2.imread(path_archivo_validacion, cv2.IMREAD_GRAYSCALE)
        imagen = cv2.resize(imagen, (dim0, dim1))
        data_validacion.append(imagen)
        # Obtener la clase del nombre del archivo
        clase = os.path.basename(os.path.dirname(path_archivo_validacion))
        # Convertir nombre de clase a índice numérico
        label = sorted(os.listdir(path_base_datos)).index(clase)
        labels_validacion.append(label)
    
    # Convertir las listas a arrays de numpy y normalizar
    data_train = np.array(data_train).astype('float32') / 255.0
    data_validacion = np.array(data_validacion).astype('float32') / 255.0
    
    # Reshape para añadir el canal
    data_train = data_train.reshape((-1, dim0, dim1, 1))
    data_validacion = data_validacion.reshape((-1, dim0, dim1, 1))

    # Convertir etiquetas a one-hot
    labels_train = tf.keras.utils.to_categorical(labels_train, num_classes)
    labels_validacion = tf.keras.utils.to_categorical(labels_validacion, num_classes)

    # Añadir verificaciones
    print(f"Forma de datos de entrenamiento: {data_train.shape}")
    print(f"Forma de etiquetas de entrenamiento: {labels_train.shape}")
    print(f"Rango de valores de datos: [{data_train.min()}, {data_train.max()}]")
    print(f"Distribución de clases en entrenamiento:")
    for i in range(num_classes):
        count = np.sum(labels_train[:, i])
        print(f"Clase {i}: {count} muestras")
    
    return data_train, labels_train, data_validacion, labels_validacion

if __name__ == "__main__":
    num_classes = 26
    dim0 = 640
    dim1 = 640
    

    autoencoder = load_model("autoencoder_smol_Conv2DTrasnpose.keras")
    classifier = create_classifier(autoencoder, num_classes)

    
    # Mostrar el resumen con mayor detalle
    classifier.summary()
    

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=5,
        min_lr=1e-6
    )
    
    classifier.compile(optimizer=optimizer, 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)])

    path_base_datos = "imagenes"
    lista_archivos_entrenamiento = []
    lista_archivos_validacion = []
    with open("splits/train.txt", "r") as f:
        for linea in f:
            lista_archivos_entrenamiento.append(linea.strip())
    with open("splits/val.txt", "r") as f:
        for linea in f:
            lista_archivos_validacion.append(linea.strip())

    x_train, y_train, x_val, y_val = get_data(path_base_datos, lista_archivos_entrenamiento, lista_archivos_validacion, num_classes, dim0, dim1)
    
    # Guardar el historial del entrenamiento
    history = classifier.fit(
        x_train, y_train,
        epochs=100,
        validation_data=(x_val, y_val),
        batch_size=32,
        callbacks=[early_stopping, reduce_lr]
    )

    # Plotear las métricas
    # Gráfica de pérdida (loss)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Gráfica de precisión (accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

    classifier.save("classifiersimple.keras")



