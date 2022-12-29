import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps


hide_streamlit_style = """

            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html = True)

st.write("""
          # 토마토 병해충 판별기
          """
          )
upload_file = st.sidebar.file_uploader("토마토 잎을 올려주세요", type=["jpg","jpeg","png","webP"])
Generate_pred=st.sidebar.button("예측하기")
model=tf.keras.models.load_model('tomatos.h5')
def import_n_pred(image_data, model):
    size = (256,256)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    reshape=img[np.newaxis,...]
    prediction = model.predict(reshape)
    #confidence = round(100 * (np.max(prediction[0])), 2)
    return prediction#,confidence
    
if upload_file is None:
    st.text("토마토 잎을 올려주세요.")
else:
    image=Image.open(upload_file)
    with st.expander('Crop Image', expanded = True):
        st.image(image, use_column_width=True)
    prediction=import_n_pred(image, model)
    class_labels=['Bacterial_spot(반점세균병)',
    'Early_blight(겹무늬병)',
    'Late_blight(잎마름역병)',
    'Leaf_Mold(잎곰팡이병)',
    'Septoria_leaf_spot(흰무늬병)',
    'Spider_mites_Two_spotted_spider_mite(점박이응애)',
    'Target_Spot(갈색무늬병)',
    'YellowLeaf_Curl_Virus(황화잎말림바이러스)',
    'mosaic_virus(모자이크병)',
    'healthy(정상)']
    st.title("{}".format(class_labels[np.argmax(prediction)]))
    if np.argmax(prediction)==0:
        st.header('\nTreatment:\nspraying Bordeaux mixture (4:4:100) once or twice on young bunches prevents the infection')
    elif np.argmax(prediction)==1:
        st.header('\nTreatment:\nProphylactic sprays with Captan (0.2%) and Benomyl or Bavistin(Carbendazim) (0.1%) minimize the development of the fungus during transit and storage.')
    elif np.argmax(prediction)==2:
        st.header('\nTreatment:\nSpraying of the grapevines at 3-4 leaf stage with fungicides like Bordeaux mixture @ 0.8% or Copper Oxychloride @ 0.25% or Carbendazim @ 0.1% are effective against this disease.')
    
    elif np.argmax(prediction)==3:
        st.header('\nPlants are healthy')

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
          
             background-image: url("http://cdn.itdaily.kr/news/photo/202103/202318_202307_3426.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )       

footer = """
<style>
a:link , a:visited{
    color: white;
    background-color: transparent;
    text-decoration: None;
}

a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: None;
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: black;
    text-align: center;
}

</style>

<div class="footer">
<p style = "align:center; color:white">Developed by taeksu</p>
</div>
"""

st.markdown(footer, unsafe_allow_html = True)     