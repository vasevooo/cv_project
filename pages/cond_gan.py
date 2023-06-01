import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


# Load the model
generator = keras.models.load_model('generator_model.h5')


@st.cache
def generate_image(seed):
    noise = np.random.normal(0, 1, (1, 100))
    label = np.array([[seed]])
    img = generator.predict([noise, label])
    return img.reshape(28, 28)

def main():
    st.title('Number Image Generator')

    seed = st.slider('Select a number from 0 to 9', 0, 9)
    if st.button('Generate'):
        image = generate_image(seed)

        st.image(image, caption=f'Generated number: {seed}', use_column_width=True)

if __name__ == '__main__':
    main()
