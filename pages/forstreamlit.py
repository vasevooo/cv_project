import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog
import io
# Функция для обработки изображения с помощью модели

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

def clean_image(image, model):
    # Преобразование изображения в тензор
    tensor = transforms.ToTensor()(image).unsqueeze(0)
    # Обработка изображения с помощью модели
    cleaned_tensor = model(tensor)
    # Преобразование тензора обратно в изображение
    cleaned_image = transforms.ToPILImage()(cleaned_tensor.squeeze(0))
    return cleaned_image

# Определение главной функции приложения
def main():
    # Загрузка модели из файла
    model = torch.load("https://github.com/vasevooo/cv_project/tree/main/pages/model.pth", map_location=torch.device('cpu'))
    model.eval()
    # Загрузка изображения и обработка его с помощью модели
    uploaded_file = st.file_uploader("Загрузите изображение с текстом", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Исходное изображение")
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
