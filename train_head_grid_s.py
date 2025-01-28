import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
import cv2
import os
import numpy as np
from tensorflow.keras import backend as K

def create_classifier_head(input_shape, num_classes, dense_layers, dropout_rate, l2_reg):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    
    # Normalización por lotes inicial
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Capas densas dinámicas
    for units in dense_layers:
        x = Dense(units, activation='relu', 
                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def load_autoencoder(path):
    return load_model(path)

def get_encoder_from_autoencoder(autoencoder):
    # Obtener la entrada del autoencoder
    input_layer = autoencoder.input
    
    # Encontrar la capa del código (última capa del encoder)
    code_layer = autoencoder.layers[8].output
    
    # Crear el modelo del encoder
    encoder = Model(inputs=input_layer, outputs=code_layer, name='encoder_model')
    
    # Congelar todas las capas del encoder
    for layer in encoder.layers:
        layer.trainable = False
        
    return encoder

def create_classifier(autoencoder, num_classes, dense_layers, dropout_rate, l2_reg):
    encoder = get_encoder_from_autoencoder(autoencoder)
    classifier_head = create_classifier_head(encoder.output_shape[1:], num_classes, 
                                          dense_layers, dropout_rate, l2_reg)
    
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
    # Parámetros para grid search
    param_grid = {
        'dense_layers': [
            [256, 128],           # 2 capas (baseline)
            [256, 128, 64],       # 3 capas
            [256, 128, 64, 32],   # 4 capas
        ],
        'dropout_rates': [
            0.25,   # Un poco más fuerte
            0.3     # Más fuerte para redes más profundas
        ],
        'l2_regs': [
            0.003,  # Regularización suave
            0.005,  # Moderada
            0.008   # Más fuerte para redes profundas
        ],
        'learning_rates': [
            0.0001,   # Conservador (mejor para redes profundas)
            0.0002,   # Moderado
            0.0003    # Más agresivo
        ],
        'batch_sizes': [
            16,   # Pequeño
            32,   # Medio
        ]
    }

    num_classes = 26
    dim0 = 320
    dim1 = 320
    
    autoencoder = load_autoencoder("autoencoder.keras")
    
    path_base_datos = "imagenes"
    lista_archivos_entrenamiento = []
    lista_archivos_validacion = []
    with open("splits/train.txt", "r") as f:
        lista_archivos_entrenamiento = f.read().splitlines()
    with open("splits/val.txt", "r") as f:
        lista_archivos_validacion = f.read().splitlines()

    x_train, y_train, x_val, y_val = get_data(path_base_datos, 
                                             lista_archivos_entrenamiento, 
                                             lista_archivos_validacion, 
                                             num_classes, dim0, dim1)
    
     # Grid Search
    best_val_acc = 0
    best_params = None
    results = []
    
    for dense_layer in param_grid['dense_layers']:
        for dropout_rate in param_grid['dropout_rates']:
            for l2_reg in param_grid['l2_regs']:
                for lr in param_grid['learning_rates']:
                    for batch_size in param_grid['batch_sizes']:
                        print(f"\nProbando configuración:")
                        print(f"Dense layers: {dense_layer}")
                        print(f"Dropout rate: {dropout_rate}")
                        print(f"L2 reg: {l2_reg}")
                        print(f"Learning rate: {lr}")
                        print(f"Batch size: {batch_size}")
                        
                        # Crear y compilar modelo
                        classifier = create_classifier(autoencoder, num_classes, 
                                                    dense_layer, dropout_rate, l2_reg)
                        
                        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                        classifier.compile(optimizer=optimizer,
                                        loss='categorical_crossentropy',
                                        metrics=['accuracy', 
                                                tf.keras.metrics.TopKCategoricalAccuracy(k=3),
                                                tf.keras.metrics.AUC()])
                        
                        # Callbacks
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
                        
                        # Entrenar
                        history = classifier.fit(
                            x_train,
                            y_train,
                            epochs=100,
                            validation_data=(x_val, y_val),
                            callbacks=[early_stopping, reduce_lr],
                            batch_size=batch_size,
                            shuffle=True
                        )
                        
                        # Guardar resultados de manera más robusta
                        val_acc = max(history.history['val_accuracy'])
                        metrics_dict = {
                            'val_accuracy': val_acc,
                            'val_top_k': history.history['val_top_k_categorical_accuracy'][-1],
                            'train_accuracy': history.history['accuracy'][-1],
                            'val_loss': history.history['val_loss'][-1]
                        }
                        
                        # Intentar obtener AUC con diferentes nombres posibles
                        try:
                            metrics_dict['val_auc'] = history.history['val_auc_1'][-1]
                        except KeyError:
                            try:
                                metrics_dict['val_auc'] = history.history['val_auc'][-1]
                            except KeyError:
                                metrics_dict['val_auc'] = None
                        
                        results.append({
                            'params': {
                                'dense_layers': dense_layer,
                                'dropout_rate': dropout_rate,
                                'l2_reg': l2_reg,
                                'learning_rate': lr,
                                'batch_size': batch_size
                            },
                            'metrics': metrics_dict
                        })
                        
                        # Actualizar mejor modelo
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_params = results[-1]['params']
                            classifier.save("best_classifier.keras")
                            
                        # Guardar resultados parciales
                        import json
                        with open('grid_search_results.json', 'w') as f:
                            json.dump(results, f, indent=4)

    print("\nMejores parámetros encontrados:")
    print(json.dumps(best_params, indent=4))
    print(f"Mejor precisión de validación: {best_val_acc}")