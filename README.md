# CV_project

*CV_project* - это многопользовательский проект, в котором участвовало 5 человек: Василий Севостьянов, Мария Козлова, Анна Филина, Ильдар Хасанов и Виктория Красикова. Целью проекта было разработать  multipage-приложение с использованием платформы Streamlit и применить различные методы компьютерного зрения для решения задач.

## Задачи

### Генерация заданной цифры с помощью Conditional GAN на датасете MNIST:
Виктория Красикова и Анна Филина занимались разработкой модели Conditional Generative Adversarial Network (CGAN) для генерации изображений рукописных цифр от 0 до 9 на основе датасета MNIST. Архитектура CGAN состоит из генератора, который получает на вход случайный шум и заданную цифру, а затем генерирует соответствующее изображение, и дискриминатора, который оценивает насколько изображение, предоставленное генератором, соответствует реальным изображениям из датасета. После обучения модель может генерировать новые изображения цифр, учитывая заданную цифру во входном шуме. Были использованы библиотеки глубокого обучения TensorFlow или PyTorch.

### Детекция опухолей мозга по фотографии и детекция моделей машин Formula 1 с помощью YOLOv5:
Василий Севостьянов и Мария Козлова использовали архитектуру YOLOv5 для детекции опухолей мозга на медицинских изображениях и детекции моделей машин Formula 1. Для обучения модели были размечены данные с помощью платформы robolow.com, где объекты (опухоли мозга или модели машин) были помечены на изображениях. YOLOv5 обучалась на этих данных с целью научиться обнаруживать и классифицировать соответствующие объекты на изображениях. Обучение включало в себя процесс оптимизации параметров модели с использованием градиентного спуска и функции потерь.

### Очистка документов от шумов с помощью автоэнкодера на датасете Denoising Dirty Documents:
Ильвир Хасанов разработал модель автоэнкодера для очистки документов от различных типов шумов на датасете Denoising Dirty Documents. Была проведена подготовка данных и создание шумовых версий документов. Затем была построена архитектура автоэнкодера, состоящая из энкодера, декодера и  latent size, который представляет собой сжатое представление исходного документа. Обучение проходило путем минимизации функции потерь, основанной на сравнении восстановленного документа с исходным чистым документом. После завершения обучения модель была протестирована на новых зашумленных документах, и ее эффективность была оценена по метрикам точности восстановления и снижения уровня шума.


# CV_project

CV_project is a multi-user project involving 5 individuals: Vasiliy Sevostyanov, Maria Kozlova, Anna Filina, Ildar Khasanov, and Victoria Krasikova. The goal of the project was to develop a multi-page application using the Streamlit platform and apply various computer vision methods to solve tasks.

## Tasks

### Generating a Specified Digit Using Conditional GAN on the MNIST Dataset:
Victoria Krasikova and Anna Filina worked on developing a Conditional Generative Adversarial Network (CGAN) model to generate handwritten digit images from 0 to 9 based on the MNIST dataset. The CGAN architecture consists of a generator that takes random noise and a specified digit as input and generates the corresponding image. The discriminator evaluates how well the generated image matches the real images from the dataset. The model was trained through iterations of generator and discriminator interactions. After training, the model can generate new digit images, considering the specified digit in the input noise. Deep learning libraries such as TensorFlow or PyTorch were used.

### Brain Tumor Detection from Images and Formula 1 Car Detection using YOLOv5:
Vasiliy Sevostyanov and Maria Kozlova utilized the YOLOv5 architecture for brain tumor detection in medical images and Formula 1 car model detection. The training data was annotated using the robolow.com platform, where objects (brain tumors or car models) were labeled in the images. YOLOv5 was trained on this data to learn how to detect and classify the corresponding objects in the images. The training process involved optimizing the model parameters using gradient descent and loss functions.

### Document Denoising with an Autoencoder on the Denoising Dirty Documents Dataset:
Ilvir Khasanov developed an autoencoder model for cleaning documents from various types of noise using the Denoising Dirty Documents dataset. Data preparation and the creation of noisy document versions were performed. Then, an autoencoder architecture was constructed, consisting of an encoder, a decoder, and a latent size that represents the compressed representation of the original document. Training involved minimizing a loss function based on comparing the reconstructed document with the original clean document. After training, the model was tested on new noisy documents, and its effectiveness was evaluated based on metrics such as reconstruction accuracy and noise reduction.
