import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np


def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
        
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model, class_names):
    # Ensure the image is in RGB mode
    image = image.convert('RGB')

    # Resize the image to (224, 224) while maintaining its aspect ratio
    image = image.resize((224, 224))

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Normalize the image: pixel values to the range -1 to 1
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Expand the dimensions to match the model's input shape (1, 224, 224, 3)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make a prediction using the model
    prediction = model.predict(data)

    # Get the class label with the highest probability
    predicted_class_index = np.argmax(prediction)
    class_name = class_names[predicted_class_index]

    # Get the confidence score (probability) for the predicted class
    conf_score = prediction[0][predicted_class_index]

    return class_name, conf_score