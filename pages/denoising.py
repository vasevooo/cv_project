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

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://romandecoratingproducts.com/wp-content/uploads/2021/09/Wrinkled-Paper-scaled.jpg");
background-size: 130%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("https://i.ibb.co/n75r0q1/angryimg.png");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(1,1,1,1);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}

div.css-1sdqqxz.esravye1 {{
background-color: rgba(238, 238, 238, 0.5);
border: 10px solid #EEEEEE;
padding: 5% 5% 5% 10%;
border-radius: 5px;
}}



</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import io
# Функция для обработки изображения с помощью модели

col1, col2, col3 = st.columns([1,8,1])
#col1, col2 = st.columns(2)

### Гистограмма total_bill
with col2:
    class ConvAutoencoder(nn.Module):
        def __init__(self):
            super(ConvAutoencoder, self).__init__()
            # encoder 
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.SELU()
                )

            self.conv2 = nn.Sequential(
                nn.Conv2d(16, 64, kernel_size=2, bias=False),
                nn.BatchNorm2d(64),
                nn.SELU()
                )
            self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=2, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU()
                )

            self.pool = nn.MaxPool2d(2, 2, return_indices=True, ceil_mode=True) 

            self.unpool = nn.MaxUnpool2d(2, 2)

            self.conv1_t = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, bias=False),
                nn.BatchNorm2d(64),
                nn.SELU()
                )

            self.conv2_t = nn.Sequential(
                nn.ConvTranspose2d(64, 16, kernel_size=2, bias=False),
                nn.BatchNorm2d(16),
                nn.SELU()
                )
            self.conv3_t = nn.Sequential(
                nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1),
                nn.LazyBatchNorm2d(),
                nn.Sigmoid()
                )     
            self.drop_out = nn.Dropout(0.5) 
        def encode(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
    #         x = self.drop_out(x)
            x, indicies = self.pool(x) # ⟸ bottleneck
            return x, indicies

        def decode(self, x, indicies):
            x = self.unpool(x, indicies)
            x = self.conv1_t(x)
            x = self.conv2_t(x)
    #         x = self.drop_out(x)
            x = self.conv3_t(x)
            return x

        def forward(self, x):
            latent, indicies = self.encode(x)
            out = self.decode(latent, indicies)      
            return out


    model = ConvAutoencoder()
    def preproces_img(image):
        # Подгонка размера
        #image = transforms.Resize(256, 256)(image)
        # Преобразование изображения в серый (одноканальный)
        image = transforms.Grayscale(1)(image)
        # Преобразование изображения в тензор
        tensor = transforms.ToTensor()(image).unsqueeze(0)
        return tensor

    def clean_image(image, model):
        # Обработка изображения с помощью модели
        cleaned_tensor = model(image)
        # Преобразование тензора обратно в изображение
        cleaned_image = transforms.ToPILImage()(cleaned_tensor.squeeze(0))
        return cleaned_image

    # Определение главной функции приложения
    def main():
        # Загрузка модели из файла
        model = torch.load("model_denoising.pth", map_location=torch.device('cpu'))
        model.eval()
        # Загрузка изображения и обработка его с помощью модели
        st.header('Очистка изображения с текстом от шумов')
        uploaded_file = st.file_uploader("Загрузите изображение с текстом", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Исходное изображение")
            image = preproces_img(image)
            cleaned_image = clean_image(image, model)
            st.image(cleaned_image, caption="Очищенное изображение")
            # Кнопка для загрузки очищенного изображения на компьютер
            #if st.button("Сохранить очищенное изображение"):
                #cleaned_image.save("cleaned_image.png")
            buffer = io.BytesIO()
            cleaned_image.save(buffer, format='PNG')
            st.success("Желаете загрузить изображение?")
                # Создание кнопки для скачивания обработанного изображения
            st.download_button(
            label='Download processed image',
            data=buffer.getvalue(),
            file_name='processed_image.png',
            mime='image/png')

    if __name__ == '__main__':
        main()
