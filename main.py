import streamlit as st
from keras.models import load_model
from keras.preprocessing import image as keras_image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from PIL import Image

# Load your model
model = load_model('./model/harist.h5')

# Load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]

# Set title and header
st.title('Pneumonia Classification')
st.header('Upload a chest X-ray image')

# Upload file through Streamlit
file = st.file_uploader('Choose a file', type=['jpeg', 'jpg', 'png'])

# Display image and perform classification
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image for your model
    img_array = keras_image.img_to_array(image)
    img_array = img_array / 255.0  # Normalize the image
    img_array = keras_image.smart_resize(img_array, (224, 224))
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    class_name, conf_score = decoded_predictions[0][1], decoded_predictions[0][2]

    # Write classification results
    st.write("## Prediction: {}".format(class_name))
    st.write("### Confidence Score: {}%".format(round(conf_score * 100, 2)))
