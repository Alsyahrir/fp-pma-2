import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import os  # Add this import for the os module
from util import set_background, classify

# Get the absolute path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set the background
background_image_path = os.path.join(current_dir, 'bgs', 'bg5.png')
set_background(background_image_path)

# Set title
st.title('Pneumonia classification')

# Set header
st.header('Please upload a chest X-ray image')

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load classifier
model_path = os.path.join(current_dir, 'model', 'harist.h5')
model = load_model(model_path)

# Load class names
labels_path = os.path.join(current_dir, 'model', 'labels.txt')
with open(labels_path, 'r') as f:
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
