import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import streamlit as st

# Загрузка сохраненной модели
class GeneratorModel(nn.Module):
    def __init__(self):
        super(GeneratorModel, self).__init__()
        input_dim = 100 + 10
        output_dim = 784
        self.label_embedding = nn.Embedding(10, 10)
        
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2)
        )

        self.hidden_layer2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )

        self.hidden_layer3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        self.hidden_layer4 = nn.Sequential(
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x, labels):
        c = self.label_embedding(labels).view(labels.shape[0], -1)
        x = torch.cat([x, c], 1)
        output = self.hidden_layer1(x)
        output = self.hidden_layer2(output)
        output = self.hidden_layer3(output)
        output = self.hidden_layer4(output)
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

generator = GeneratorModel()
generator.load_state_dict(torch.load("CGAN.pth"))
generator.eval()

# Задать число от 0 до 9 с помощью ползунка
number = st.slider("Select a number", 0, 9)

# Функция денормализации изображений
def denorm(img_tensors):
    return img_tensors * 0.5 + 0.5

# Генерация изображения при нажатии кнопки "Generate"
if st.button("Generate"):
    latent_tensor = torch.randn(1, 100).to(device)
    label_tensor = torch.tensor([number]).to(device)
    fake_image = generator(latent_tensor, label_tensor)
    
    # Сохранение сгенерированного изображения
    save_image(denorm(fake_image), "generated_image.png")
    
    # Отображение сгенерированного изображения
    st.image("generated_image.png", caption="Generated Image")
