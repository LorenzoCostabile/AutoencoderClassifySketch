FROM tensorflow/tensorflow:2.7.0-gpu

WORKDIR /app

# Instalar dependencias adicionales
RUN apt-get update && apt-get install -y \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Copiar los archivos del proyecto
COPY . .

# Comando por defecto
CMD ["python", "train.py"]