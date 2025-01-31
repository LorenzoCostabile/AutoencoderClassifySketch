import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPooling2D, Input, UpSampling2D
from tensorflow.keras.models import Model
import cv2
import os
import numpy as np
import tensorflow.image as tfimg

def ssim_loss(y_true, y_pred):
    return 1 - tfimg.ssim(y_true, y_pred, max_val=1.0)

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

def decoder(x, filter_list):
    # First upsampling block
    h1 = UpSampling2D((2, 2))(x)
    h1 = Conv2D(filter_list[0], (3, 3), padding="same")(h1)


    # Second upsampling block
    h2 = UpSampling2D((2, 2))(h1)
    h2 = Conv2D(filter_list[1], (3, 3), padding="same")(h2)

    # Third upsampling block
    h3 = UpSampling2D((2, 2))(h2)
    h3 = Conv2D(filter_list[2], (3, 3), padding="same")(h3)

    # Fourth upsampling block
    h4 = UpSampling2D((2, 2))(h3)
    h4 = Conv2D(filter_list[3], (3, 3), padding="same")(h4)

    # Final convolution
    y = Conv2D(1, (3, 3), activation='sigmoid', padding="same")(h4)
    
    return y


"""
def decoder(x, filter_list, activation="relu"):
    h1 = Conv2DTranspose(filter_list[0], (3, 3), strides=(2, 2), padding="same")(x)
    h1 = BatchNormalization()(h1)
    h1 = Activation(activation)(h1)
    
    h2 = Conv2DTranspose(filter_list[1], (3, 3), strides=(2, 2), padding="same")(h1)
    h2 = BatchNormalization()(h2)
    h2 = Activation(activation)(h2)
    
    h3 = Conv2DTranspose(filter_list[2], (3, 3), strides=(2, 2), padding="same")(h2)
    h3 = BatchNormalization()(h3)
    h3 = Activation(activation)(h3)
    
    h4 = Conv2DTranspose(filter_list[3], (3, 3), strides=(2, 2), padding="same")(h3)
    h4 = BatchNormalization()(h4)
    h4 = Activation(activation)(h4)
    
    y  = Conv2D(1, (3, 3), activation='sigmoid', padding="same")(h4)
    
    return y
"""



def create_autoencoder(dim0, dim1, filter_list):
    x = Input(shape=(dim0, dim1, 1))
    code = encoder(x, filter_list, activation="relu")
    decoder_filter_list = filter_list[::-1]
    x_pred = decoder(code, decoder_filter_list)
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
    dim0 = 640
    dim1 = 640
    
    # Reducimos el número de filtros
    autoencoder = create_autoencoder(dim0=dim0, dim1=dim1, filter_list=[4, 7, 15, 27])
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
    autoencoder.save('autoencoder_smol.keras')
