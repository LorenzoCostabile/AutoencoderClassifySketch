import tensorflow as tf
import os
import sys

print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)

# Verificar variables de entorno
print("\nVariables de entorno:")
print("CUDA_PATH:", os.environ.get('CUDA_PATH'))
print("PATH contiene CUDA:", 'cuda' in os.environ.get('PATH').lower())

# Verificar dispositivos
print("\nDispositivos físicos:")
print(tf.config.list_physical_devices())
print("\nDispositivos GPU:")
print(tf.config.list_physical_devices('GPU'))

# Verificar CUDA
print("\nCUDA built:", tf.test.is_built_with_cuda())
print("CUDA disponible:", tf.test.is_gpu_available())

# Información adicional de CUDA
from tensorflow.python.client import device_lib
print("\nInformación detallada de dispositivos:")
print(device_lib.list_local_devices())