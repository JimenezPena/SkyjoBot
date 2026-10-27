# 1. Usar una imagen oficial de Python ligera
FROM python:3.11-slim

# 2. Definir el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiar primero el archivo de dependencias (optimiza la caché de Docker)
COPY requirements.txt .

# 4. Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el resto del código de tu proyecto local a la imagen
COPY . .

# 6. Exponer el puerto donde corre el juego
EXPOSE 5000

# 7. Ejecutar directamente tu aplicación
CMD ["python", "online_app.py"]
