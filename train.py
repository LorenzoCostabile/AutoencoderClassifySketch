import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.models import Model
import cv2
import os
import numpy as np

def encoder(x, filter_list, activation="relu"):
    h1 = Conv2D(filter_list[0], (3, 3), activation=activation, padding="same")(x)
    p1 = MaxPooling2D((2,2))(h1)
    h2 = Conv2D(filter_list[1], (3, 3), activation=activation, padding="same")(p1)
    p2 = MaxPooling2D((2,2))(h2)
    h3 = Conv2D(filter_list[2], (3, 3), activation=activation, padding="same")(p2)
    p3 = MaxPooling2D((2,2))(h3)
    h4 = Conv2D(filter_list[3], (3, 3), activation=activation, padding="same")(p3)
    code = MaxPooling2D((2,2))(h4)
    return code

def decoder(x, filter_list, activation="relu"):
    h1 = Conv2D(filter_list[0], (3, 3), activation=activation, padding="same")(x)
    u1 = UpSampling2D((2,2))(h1)
    h2 = Conv2D(filter_list[1], (3, 3), activation=activation, padding="same")(u1)
    u2 = UpSampling2D((2,2))(h2)
    h3 = Conv2D(filter_list[2], (3, 3), activation=activation, padding="same")(u2)
    u3 = UpSampling2D((2,2))(h3)
    h4 = Conv2D(filter_list[3], (3, 3), activation=activation, padding="same")(u3)
    u4 = UpSampling2D((2,2))(h4)
    y  = Conv2D(1, (4, 4), activation='linear', padding="same")(u4)
    return y

def create_autoencoder(dim0, dim1, filter_list, activation="relu"):
    x = Input(shape=(dim0, dim1, 1))
    code = encoder(x, filter_list, activation)
    decoder_filter_list = filter_list[::-1]
    x_pred = decoder(code, decoder_filter_list, activation)
    ae = Model(x, x_pred)
    ae.compile(optimizer='adam', loss='mse')
    return ae

def get_data(path_base_datos, list_archivos_entrenamiento, list_archivos_validacion, dim0=None, dim1=None):
    data_train = []
    data_validacion = []
    dim0 = dim0
    dim1 = dim1

    for archivo in list_archivos_entrenamiento:
        path_archivo_entrenamiento = os.path.join(path_base_datos, archivo)
        imagen = cv2.imread(path_archivo_entrenamiento, cv2.IMREAD_GRAYSCALE)
        if dim0 is None:
            dim0, dim1 = imagen.shape[0], imagen.shape[1]
        imagen = cv2.resize(imagen, (dim0, dim1))
        data_train.append(imagen)
    
    for archivo in list_archivos_validacion:
        path_archivo_validacion = os.path.join(path_base_datos, archivo)
        imagen = cv2.imread(path_archivo_validacion, cv2.IMREAD_GRAYSCALE)
        imagen = cv2.resize(imagen, (dim0, dim1))
        data_validacion.append(imagen)
    
    # Convertir las listas a arrays de numpy y normalizar
    data_train = np.array(data_train).astype('float32') / 255.0
    data_validacion = np.array(data_validacion).astype('float32') / 255.0
    
    # Reshape para añadir el canal
    data_train = data_train.reshape((-1, dim0, dim1, 1))
    data_validacion = data_validacion.reshape((-1, dim0, dim1, 1))
    
    return data_train, data_validacion


if __name__ == "__main__":
    # Reducimos las dimensiones a 320x320 en lugar de 640x640
    dim0 = 320
    dim1 = 320
    
    # Reducimos el número de filtros
    autoencoder = create_autoencoder(dim0=dim0, dim1=dim1, filter_list=[16, 32, 64, 128])
    autoencoder.summary()
    path_base_datos = "imagenes"
    lista_archivos_entrenamiento = []
    lista_archivos_validacion = []
    with open("splits/train.txt", "r") as f:
        for linea in f:
            lista_archivos_entrenamiento.append(linea.strip())
    with open("splits/val.txt", "r") as f:
        for linea in f:
            lista_archivos_validacion.append(linea.strip())

    # Obtener los datos de entrenamiento y validacion
    data_train, data_validacion = get_data(path_base_datos, lista_archivos_entrenamiento, lista_archivos_validacion, dim0, dim1)

    # Entrenar el autoencoder
    autoencoder.fit(data_train, data_train, epochs=100, batch_size=8, validation_data=(data_validacion, data_validacion))

    # Guardar el modelo entrenado
    autoencoder.save('autoencoder.keras')
