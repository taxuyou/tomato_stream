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

st.title('Tomato Leaf Disease Prediction')

def main() :
    add_bg_from_url()
    file_uploaded = st.file_uploader('Choose an image...', type = ['jpg','jpeg','png'])
    if file_uploaded is not None :
        image = Image.open(file_uploaded)
        st.write("Uploaded Image.")
        figure = plt.figure(figsize = (5,5))
        plt.imshow(image)
        plt.axis('off')
        st.pyplot(figure)
        result, confidence = predict_class(image)
        string = f'This image likely belongs to {result} with a confidence of {confidence}%'       
        st.success(string)
        #st.success(st.write('Prediction : {}'.format(result)))
        #st.success(st.write('Confidence : {}%'.format(confidence)))

def predict_class(image) :
    with st.spinner('Loading Model...'):
        classifier_model = keras.models.load_model(r'tomatos.h5', compile = False)

    shape = ((256,256,3))
    model = keras.Sequential([hub.KerasLayer(classifier_model, input_shape = shape)])
    test_image = image.resize((256, 256))
    test_image = keras.preprocessing.image.img_to_array(test_image)
    test_image /= 255.0
    test_image = np.expand_dims(test_image, axis = 0)
    class_name = ['Bacterial_spot(반점세균병)','Early_blight(겹무늬병)',
               'Late_blight(잎마름역병)','Leaf_Mold(잎곰팡이병)',
               'Septoria_leaf_spot(흰무늬병)',
               'Spider_mites_Two_spotted_spider_mite(점박이응애)',
               'Target_Spot(갈색무늬병)',
               'YellowLeaf_Curl_Virus(황화잎말림바이러스)',
               'mosaic_virus(모자이크병)',
               'healthy(정상)']

    prediction = model.predict(test_image)
    confidence = round(100 * (np.max(prediction[0])), 2)
    final_pred = class_name[np.argmax(prediction)]
    return final_pred, confidence

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
          
             background-image: url("https://media.istockphoto.com/photos/green-leaves-pattern-background-sweet-potato-leaves-nature-dark-green-picture-id1155672947?k=20&m=1155672947&s=170667a&w=0&h=Rbx7C6PzO3sCXdnPsOhEylL4i01k7ekfENUwVXpBB5U=");
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
<p style = "align:center; color:white">Developed with taeksu</p>
</div>
"""

st.markdown(footer, unsafe_allow_html = True)

if __name__ == '__main__' : main()
