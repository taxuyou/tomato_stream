import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.write("""
          # Grape Crop Disease Identification
          """
          )
upload_file = st.sidebar.file_uploader("Upload Crop Leaf Images", type=["jpg","jpeg","png","webP"])
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
    st.text("Please upload an image file")
else:
    image=Image.open(upload_file)
    with st.expander('Crop Image', expanded = True):
        st.image(image, use_column_width=True)
    prediction=import_n_pred(image, model)
    class_labels=['Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy']
    st.title("{}".format(class_labels[np.argmax(prediction)]))
    if np.argmax(prediction)==0:
        st.header('\nTreatment:\nspraying Bordeaux mixture (4:4:100) once or twice on young bunches prevents the infection')
    elif np.argmax(prediction)==1:
        st.header('\nTreatment:\nProphylactic sprays with Captan (0.2%) and Benomyl or Bavistin(Carbendazim) (0.1%) minimize the development of the fungus during transit and storage.')
    elif np.argmax(prediction)==2:
        st.header('\nTreatment:\nSpraying of the grapevines at 3-4 leaf stage with fungicides like Bordeaux mixture @ 0.8% or Copper Oxychloride @ 0.25% or Carbendazim @ 0.1% are effective against this disease.')
    elif np.argmax(prediction)==3:
        st.header('\nPlants are healthy')