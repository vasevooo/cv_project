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

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://phonoteka.org/uploads/posts/2022-02/1643931520_73-phonoteka-org-p-fon-tekhnologii-svetlii-77.jpg");
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

div.css-1n76uvr.e1tzin5v0 {{
background-color: rgba(238, 238, 238, 0.5);
border: 10px solid #EEEEEE;
padding: 5% 5% 5% 10%;
border-radius: 5px;
}}



</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms


st.markdown("## Brain Tumor Detector")
st.sidebar.success("You are currently viewing Brain Tumor Detector Page")


def main():
    
    st.text("Upload an image and the Detector will determine \nif a tumor is positive or negative")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Read the image file
        image = preprocess_image(Image.open(uploaded_image))
        results = model(image)
        
        # Plot the uploaded image and results side by side
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(image, cmap = 'gray')
        axs[0].axis("off")
        axs[0].set_title("Uploaded Image")
        axs[1].imshow(results.render()[0])
        axs[1].axis("off")
        axs[1].set_title("Detection Result")
        st.pyplot(fig)

def preprocess_image(image):
    transforms_pipeline = transforms.Compose([
        transforms.Resize((640, 640))
        
    ])
    return transforms_pipeline(image)


if __name__ == "__main__":
    # Load the pre-trained model
    
    path_to_weights = 'yolo_model/best.pt'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_to_weights, force_reload=True)

    main()
