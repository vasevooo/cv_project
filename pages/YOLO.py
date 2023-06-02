import streamlit as st
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
