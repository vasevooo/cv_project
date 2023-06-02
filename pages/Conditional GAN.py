import streamlit as st
# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

import base64
import streamlit as st
import plotly.express as px

df = px.data.iris()

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

css-1n76uvr e1tzin5v0
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://avatars.mds.yandex.net/i?id=0af4b340256b74ab0c94fc3f60639c26e8b58124-9093415-images-thumbs&n=13");
background-size: 130%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("https://klike.net/uploads/posts/2022-11/1668930175_8-3.jpg");
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

div.css-1n76uvr.e1tzin5v0 {{
background-color: rgba(238, 238, 238, 0.5);
border: 10px solid #EEEEEE;
padding: 5% 5% 5% 10%;
border-radius: 5px;
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

col1, col2, col3 = st.columns([1,8,1])
#col1, col2 = st.columns(2)

### Гистограмма total_bill
with col2:
# Веб-приложение с использованием Streamlit
    st.title('Генерация изображений с Conditional GAN')
col1, col2, col3 = st.columns([2,5,2])
#col1, col2 = st.columns(2)

### Гистограмма total_bill
with col2:
# Веб-приложение с использованием Streamlit
    
    number = st.slider('Выберите число:', 0, 9, step=1)


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


