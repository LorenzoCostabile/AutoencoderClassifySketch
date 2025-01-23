import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.models import Model

def encoder(x, filter_list, activation="relu"):
    h1 = Conv2D(filter_list[0], (3, 3), activation=activation, padding="same")(x)
    p1 = MaxPooling2D((2,2))(h1)
    h2 = Conv2D(filter_list[1], (3, 3), activation=activation, padding="same")(p1)
    p2 = MaxPooling2D((2,2))(h2)
    h3 = Conv2D(filter_list[2], (3, 3), activation=activation, padding="same")(p2)
    code = MaxPooling2D((2,2), padding="same")(h3)
    return code

def decoder(x, filter_list, activation="relu"):
    h1 = Conv2D(filter_list[0], (3, 3), activation=activation, padding='same')(x)
    h2 = UpSampling2D((2, 2))(h1)
    h3 = Conv2D(filter_list[1], (3, 3), activation=activation, padding='same')(h2)
    h4 = UpSampling2D((2, 2))(h3)
    h5 = Conv2D(filter_list[2], (3, 3), activation=activation, padding='valid')(h4)
    h6 = UpSampling2D((2, 2))(h5)
    y  = Conv2D(1,(3,3),  activation='linear', padding='same')(h6)
    return y

def create_autoencoder(dim0, dim1, filter_list, activation="relu"):
    x = Input(shape=(dim0, dim1, 1))
    code = encoder(x, filter_list, activation)
    filter_list = [filter_list[2], filter_list[1], filter_list[0]]
    x_pred = decoder(code, filter_list, activation)
    ae = Model(x, x_pred)
    ae.compile(optimizer='adam', loss='mse')
    return ae

if __name__ == "__main__":
    dim0 = 256
    dim1 = 256
    filter_list = [5, 15, 25]
    activation = "relu"
    ae = create_autoencoder(dim0, dim1, filter_list, activation)
    ae.summary()

