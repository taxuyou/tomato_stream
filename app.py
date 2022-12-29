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
    with st.expander('토마토 잎 이미지', expanded = True):
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
        st.header('\n처방:\n종자 전염이 주요한 제 1차 전염원이기 때문에 건전 종자를 사용하는 것이 아주 중요하다. 종자 생산자는 채종포에서 발병을 철저히 막을 필요가 있다. 특히, 발병한 과실로 부터는 채종을 금한다. 시판종자는 종자소독(유효염소4%, 20배액에 20분간 침적)을 철저히 하여 물로 충분히 씻어낸다. 시설과 노지에서도 발병하지만 시설 내에서 발병을 좌우하는 것은 습도 특히, 물방울의 유무에 있다. 아침과 저녁 저온으로 하우스 천장으로 부터 물방울이 떨어져 이들이 결정적으로 발병을 촉진하는 요인이 되었다. 따라서 발병을 예방하기 위해서는 겨울철 하우스에는 난방과 환기를 충분히 하는 것이 최고의 예방책이 된다. 적용약제로는 동수화제, 쿠퍼수화제, 델란 K 수화제 등을 수확 10일전까지 3∼5회살포하여 방제한다.')
    elif np.argmax(prediction)==1:
        st.header('\n처방:\n재배적인 방법- 신고, 행수, 장십랑, 국수, 운정 등 저항성 품종을 재식한다.\n- 병든 가지는 잘라서 태운다.\n- 적절한 토양습도가 유지되지 않는 과원에서 많이 발생하므로 관·배수 시설을 잘하고, 나무의 수세를 좋게 한다. 봉지를 일찍 씌운다.약제방제 6∼7월의 장마철에 흑성병, 흑반병 방제를 겸하여 가지와 잎에 충분히 묻도록 살포한다. (약제의 종류는 흑성병, 흑반병 참조)')
    elif np.argmax(prediction)==2:
        st.header('\n처방:\n- 환기를 철저히 하여 시설내가 과습하지 않도록 한다.\n- 잦은 물주기를 하지 않고 물빠짐을 좋게 한다.\n- 항상 포장을 청결히 하고 병든 잎이나 줄기는 조기에 제거하여 불에 태우거나 땅속 깊이 묻는다.\n- 등록약제를 이용하여 방제한다.')
    elif np.argmax(prediction)==3:
        st.header('\n처방:\nSpraying of the grapevines at 3-4 leaf stage with fungicides like Bordeaux mixture @ 0.8% or Copper Oxychloride @ 0.25% or Carbendazim @ 0.1% are effective against this disease.')
    elif np.argmax(prediction)==4:
        st.header('\n처방:\nSpraying of the grapevines at 3-4 leaf stage with fungicides like Bordeaux mixture @ 0.8% or Copper Oxychloride @ 0.25% or Carbendazim @ 0.1% are effective against this disease.')
    elif np.argmax(prediction)==5:
        st.header('\n처방:\nSpraying of the grapevines at 3-4 leaf stage with fungicides like Bordeaux mixture @ 0.8% or Copper Oxychloride @ 0.25% or Carbendazim @ 0.1% are effective against this disease.')
    elif np.argmax(prediction)==6:
        st.header('\n처방:\nSpraying of the grapevines at 3-4 leaf stage with fungicides like Bordeaux mixture @ 0.8% or Copper Oxychloride @ 0.25% or Carbendazim @ 0.1% are effective against this disease.')
    elif np.argmax(prediction)==7:
        st.header('\n처방:\nSpraying of the grapevines at 3-4 leaf stage with fungicides like Bordeaux mixture @ 0.8% or Copper Oxychloride @ 0.25% or Carbendazim @ 0.1% are effective against this disease.')
    elif np.argmax(prediction)==8:
        st.header('\n처방:\nSpraying of the grapevines at 3-4 leaf stage with fungicides like Bordeaux mixture @ 0.8% or Copper Oxychloride @ 0.25% or Carbendazim @ 0.1% are effective against this disease.')
    elif np.argmax(prediction)==9:
        st.header('\n정상입니다.')    

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