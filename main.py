import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from util import set_background, classify

set_background('./bgs/bg5.png')

# Set title
st.title('Pneumonia classification')

# Set header
st.header('Please upload a chest X-ray image')

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load classifier
model = load_model('model/harist.h5')

# Load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]

# Display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # Convert image to array
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)

    # Classify image
    class_name, conf_score = classify(img_array, model, class_names)

    # Write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
