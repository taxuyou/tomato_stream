import streamlit as st
from PIL import Image
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow_hub as hub

hide_streamlit_style = """

            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html = True)

st.title('토마토 병해 예측기')


upload_file = st.sidebar.file_uploader("Upload Crop Leaf Images", type=["jpg",'jpeg','png'])
Generate_pred=st.sidebar.button("Predict")
model=tf.keras.models.load_model('model1.h5')
def import_n_pred(image_data, model):
    size = (256,256)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    reshape=img[np.newaxis,...]
    prediction = model.predict(reshape)
    return prediction
    
if upload_file is None:
    st.text("사진을 선택해주세요.")
else:
    image=Image.open(upload_file)
    with st.expander('Crop Image', expanded = True):
        st.image(image, use_column_width=True)
    prediction=import_n_pred(image, model)
    class_labels= ['Bacterial_spot(반점세균병)','Early_blight(겹무늬병)','Late_blight(잎마름역병)','Leaf_Mold(잎곰팡이병)','Septoria_leaf_spot(흰무늬병)','Spider_mites_Two_spotted_spider_mite(점박이응애)',
'Target_Spot(갈색무늬병)','YellowLeaf_Curl_Virus(황화잎말림바이러스)','mosaic_virus(모자이크병)','healthy(정상)']
    st.title("{}".format(class_labels[np.argmax(prediction)]))
st.caption("Made by taeksu KIM ")



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

if __name__ == '__main__' : main()
