import streamlit as st
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



st.markdown("# Computer Vision Project by FasterRCNN 🎈")

st.text ('Team members: \n 1. Vasily S. \n 2. Anna F. \n 3. Viktoria K. \n 4. Maria K. \n 5. Ilvir Kh.')
