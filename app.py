import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.markdown('<h1 style="color:white;">토마토 병해충 예측기</h1>', unsafe_allow_html=True)
#st.markdown('<h4 style="color:gray;"> 이 분류 모델은 다음 범주로 분류합니다.:</h2>', unsafe_allow_html=True)
#st.markdown('<h5 style="color:gray;"> 반점세균병(Bacterial spot),겹무늬병(Early blight),잎마름역병(Late blight),잎곰팡이병(Leaf Mold),흰무늬병(Septoria_leaf_spot),점박이응애(Spider mites Two spotted spider mite),갈색무늬병(Target Spot),황화잎말림바이러스(YellowLeaf Curl Virus),모자이크병(mosaic virus),정상healthy(정상)</h3>', unsafe_allow_html=True)

# background image to streamlit

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/free-photo/abstract-background-with-low-poly-design_1048-8478.jpg?w=996&t=st=1672364807~exp=1672365407~hmac=6f2ab4616122c896a7ffc40ec50c72f1ef7b8ad3fda115778c7cae90f7274482");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

upload_file = st.sidebar.file_uploader("토마토 잎의 사진을 올려주세요 !", type=["jpg","jpeg","png","webP"])
model=tf.keras.models.load_model('tomatos.h5')

def import_n_pred(image_data, model):
    size = (256,256)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    reshape=img[np.newaxis,...]
    prediction = model.predict(reshape)
    return prediction
    
if upload_file is None:
    st.text("진단결과.")
else:
    image=Image.open(upload_file)
    with st.expander('토마토 잎 이미지', expanded = True):
         st.image(image, use_column_width=True)
    prediction=import_n_pred(image, model)
    confidence = round(100 * (np.max(prediction[0])), 2)
    class_labels=['반점세균병(Bacterial spot)',
    '겹무늬병(Early blight)',
    '잎마름역병(Late blight)',
    '잎곰팡이병(Leaf Mold)',
    '흰무늬병(Septoria_leaf_spot)',
    '점박이응애(Spider mites Two spotted spider mite)',
    '갈색무늬병(Target Spot)',
    '황화잎말림바이러스(YellowLeaf Curl Virus)',
    '모자이크병(mosaic virus)',
    '정상healthy(정상)']
  
    st.title("{}".format(class_labels[np.argmax(prediction)]))
    st.title("신뢰도: {}%".format(confidence))
    if np.argmax(prediction)==0:
        st.header('<h4 style="color:white;">\n처방:\n종자 전염이 주요한 제 1차 전염원이기 때문에 건전 종자를 사용하는 것이 아주 중요하다. 종자 생산자는 채종포에서 발병을 철저히 막을 필요가 있다. 특히, 발병한 과실로 부터는 채종을 금한다. 시판종자는 종자소독(유효염소4%, 20배액에 20분간 침적)을 철저히 하여 물로 충분히 씻어낸다. 시설과 노지에서도 발병하지만 시설 내에서 발병을 좌우하는 것은 습도 특히, 물방울의 유무에 있다. 아침과 저녁 저온으로 하우스 천장으로 부터 물방울이 떨어져 이들이 결정적으로 발병을 촉진하는 요인이 되었다. 따라서 발병을 예방하기 위해서는 겨울철 하우스에는 난방과 환기를 충분히 하는 것이 최고의 예방책이 된다. 적용약제로는 동수화제, 쿠퍼수화제, 델란 K 수화제 등을 수확 10일전까지 3∼5회살포하여 방제한다.</h2>', unsafe_allow_html=True)
    elif np.argmax(prediction)==1:
        st.header('\n처방:\n재배적인 방법- 신고, 행수, 장십랑, 국수, 운정 등 저항성 품종을 재식한다.\n- 병든 가지는 잘라서 태운다.\n- 적절한 토양습도가 유지되지 않는 과원에서 많이 발생하므로 관·배수 시설을 잘하고, 나무의 수세를 좋게 한다. 봉지를 일찍 씌운다.약제방제 6∼7월의 장마철에 흑성병, 흑반병 방제를 겸하여 가지와 잎에 충분히 묻도록 살포한다. (약제의 종류는 흑성병, 흑반병 참조)')
    elif np.argmax(prediction)==2:
        st.header('\n처방:\n- 환기를 철저히 하여 시설내가 과습하지 않도록 한다.\n- 잦은 물주기를 하지 않고 물빠짐을 좋게 한다.\n- 항상 포장을 청결히 하고 병든 잎이나 줄기는 조기에 제거하여 불에 태우거나 땅속 깊이 묻는다.\n- 등록약제를 이용하여 방제한다.')
    elif np.argmax(prediction)==3:
        st.header('\n처방:\n- 병든 잎을 신속히 제거한다.\n- 90%이상의 상대습도가 유지되지 않도록 한다.\n- 통풍이 잘되게 하고 밀식하지 않는다.\n- 건전한 종자를 사용하고, 깨끗한 자재를 사용한다.\n- 질소질 비료의 과용을 피한다.')
    elif np.argmax(prediction)==4:
        st.header('\n처방:\n발병초기 등록약제를 살포하여 병의 확산을 막는다.')
    elif np.argmax(prediction)==5:
        st.header('\n처방:\n작물의 하위 잎에서 발생이 시작하여 새잎으로 확산된다.\n점박이응애 발생지점에 물을 뿌려주면 발생이 억제된다.\n작물재배 후에 작물 잔재물을 깨끗이 청소하여 발생원을 없애야 한다.\n발생초기에 약제를 살포하는 것이 방제효과가 높다.\n잎응애는 약제저항성이 쉽게 발달하므로 같은 계통의 약제를 계속 사용하지 말아야 한다.\n국내에 상업적으로 이용되는 칠레이리응애, 사막이리응애, 긴털이리응애, 꼬마무당벌레 등이 있다.\n점박이응애 밀도가 높으면 잔류기간이 짧은 응애약제를 살포한 후 천적을 방사한다.')
    elif np.argmax(prediction)==6:
        st.header('\n처방:\n- 관수 및 배수를 철저히 하고 균형 있는 시비를 한다.\n- 전정을 통해 수관내 통풍과 통광을 원활히 하고, 병에 걸린 낙엽을 모아 태우거나 땅 속 깊이 묻어 전염원을 제거한다.\n- 약제에 의한 방제는 6월 중순경(발병초)부터 8월까지 가능한 강우 전에 정기적으로 적용약제를 수관내부까지 골고루 묻도록 충분량을 살포한다.\n- 과수원에서 초기병반이 보이는 즉시 약제를 살포한다.\n이 병은 한번 발생하면 이후 방제하기가 매우 곤란한 병이므로 예방에 초점을 맞추어 방제한다.')
    elif np.argmax(prediction)==7:
        st.header('\n처방:\n병을 전염시키는 해충의 세대 기간이 짧아 연간 발생횟수가 많고 증식률이 높으므로 발생초기에 방제하고, 육묘 시 철저한 관리로 병의 확산 예방 한다.')
    elif np.argmax(prediction)==8:
        st.header('\n처방:\n○ CMV 방제 - 진딧물이 전염시키므로 진딧물의 기주를 제거한다.\n- 등록된 살충제를 살포하여 진딧물을 방제한다 - 전 작물의 잔재물을 제거하고, 작물의 파종시기 및 올멱심기 시기를 조절한다.\n- 바이러스의 잠재적인 보존원인 잡초나 중간기주를 제거한다.\n- 전염원이 되는 병든 식물은 발견 즉시 제거한다.\n○ PepMoV 방제 - 복숭아혹진딧물의 기주식물인 가지과, 배추과 등을 주위에 재배하지 않는다.\n- 자연 발병 기주인 담배, 고추, 감자 등을 연속 재배하지 말아야 한다.\n- 병든 식물체는 일찍 제거한다.\n- 등록된 살충제를 살포하여 진딧물을 방제한다.\n- 병에 잘 걸리지 않는 품종을 재배한다.\n○ TMV 및 ToMV 방제 - 고추와 토마토를 연속재배하지 말아야 한다.\n- 종자 소독을 철저히 한 후 파종한다.\n- 오염 토양, 옷, 손, 농기구들의 오염물을 제거한다.\n- 옮겨심기, 눈따기, 수확 등 작업 시에 전염이 되므로 주의하여야 하며 작업 전에는 반드시 손을 닦아야 한다 - 전염원으로부터 격리된 지역에서 재배한다.\n- 전염원이 되는 병든식물은 발견 즉시 제거한다.')
    elif np.argmax(prediction)==9:
        st.header('\n정상입니다.')    

st.markdown('<h4 style="color:white;"> 이 분류 모델은 다음 범주로 분류합니다.:</h2>', unsafe_allow_html=True)
st.markdown('<h5 style="color:white;"> 반점세균병(Bacterial spot),겹무늬병(Early blight),잎마름역병(Late blight),\n잎곰팡이병(Leaf Mold),흰무늬병(Septoria_leaf_spot),점박이응애(Spider mites Two spotted spider mite),\n갈색무늬병(Target Spot),황화잎말림바이러스(YellowLeaf Curl Virus),모자이크병(mosaic virus),정상healthy(정상)</h3>', unsafe_allow_html=True)



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
<p style = "align:center; color:white">saltware</p>
</div>
"""

st.markdown(footer, unsafe_allow_html = True)     