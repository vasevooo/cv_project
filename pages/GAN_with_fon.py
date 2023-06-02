import streamlit as st
import base64
import streamlit as st
import plotly.express as px

df = px.data.iris()

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


#img = get_img_as_base64("https://catherineasquithgallery.com/uploads/posts/2021-02/1612739741_65-p-goluboi-fon-tsifri-110.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://catherineasquithgallery.com/uploads/posts/2021-02/1612739741_65-p-goluboi-fon-tsifri-110.jpg");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("https://catherineasquithgallery.com/uploads/posts/2021-02/1612739741_65-p-goluboi-fon-tsifri-110.jpg");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

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
    random_latent_vector = tf.random.normal(shape=(1, latent_dim))
    one_hot_label = tf.one_hot([number], num_classes)
    input_data = tf.concat([random_latent_vector, one_hot_label], axis=1)

    generated_image = model.predict(input_data)
    generated_image = generated_image.reshape(28, 28)
    generated_image = tf.image.resize(generated_image[None, ...], (28, 28))[0]  # Добавлено [None, ...] для добавления измерения
    return generated_image

# Веб-приложение с использованием Streamlit
st.title('Генерация изображений с Conditional GAN')
number = st.slider('Выберите число:', 0, 9, step=1)

col1, col2, col3 = st.columns([2,4,2])
#col1, col2 = st.columns(2)

### Гистограмма total_bill
with col2:
 #col1.subheader("Гистограмма total_bill:")

    # Генерация и отображение изображения
    generated_image = generate_image(number)
    generated_image_np = generated_image.numpy()  # Преобразование в массив NumPy
    fig, ax = plt.subplots()
    ax.scatter([1, 2], [1, 2], color='black')
    plt.imshow(generated_image_np, cmap='gray')
    plt.axis('off')
    fig.set_size_inches(3, 3)
    st.pyplot(fig)


