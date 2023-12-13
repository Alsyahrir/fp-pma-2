import streamlit as st
from keras.models import load_model
from keras.preprocessing import image as keras_image
from PIL import Image
import numpy as np

from util import classify, set_background

set_background('./bgs/bg5.png')

# set title
st.title('Pneumonia classification')

# set header
st.header('Please upload a chest X-ray image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('./model/harist.h5')

# load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # Preprocess the image for the model
    image = image.resize((224, 224))  # Resize the image to match the model's expected sizing
    image_array = keras_image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array /= 255.0  # Normalize the image

    # classify image
    class_name, conf_score = classify(image_array, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
