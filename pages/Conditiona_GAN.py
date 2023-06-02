import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Загрузка модели
model = keras.models.load_model('cgan_model.h5')

# Задание размерностей входных данных модели
latent_dim = 128
num_classes = 10

# Функция для генерации изображения
def generate_image(number):
    # Преобразование входных данных
    random_latent_vector = tf.random.normal(shape=(1, latent_dim))
    one_hot_label = tf.one_hot([number], num_classes)
    input_data = tf.concat([random_latent_vector, one_hot_label], axis=1)

    # Генерация изображения
    generated_image = model.predict(input_data)
    generated_image = generated_image.reshape(28, 28)

    return generated_image

# Веб-приложение с использованием Streamlit
st.title('Генерация изображений с Conditional GAN')
number = st.slider('Выберите число:', 0, 9, step=1)

# Генерация и отображение изображения
generated_image = generate_image(number)
fig, ax = plt.subplots()
ax.scatter([1, 2], [1, 2], color='black')
plt.imshow(generated_image, cmap='gray')
plt.axis('off')
fig.set_size_inches(8, 8)
st.pyplot(fig)
