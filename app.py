# import streamlit as st
# import numpy as np
# from PIL import Image
# from tensorflow.keras.models import load_model
# import segmentation_models as sm
# from keras import backend as K
# import matplotlib.pyplot as plt

# import warnings 
  

# # Settings the warnings to be ignored 
# warnings.filterwarnings('ignore') 

# # Define custom loss functions
# def jaccard_coef(y_true, y_pred):
#     y_true_flatten = K.flatten(y_true)
#     y_pred_flatten = K.flatten(y_pred)
#     intersection = K.sum(y_true_flatten * y_pred_flatten)
#     final_coef_value = (intersection + 1.0) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection + 1.0)
#     return final_coef_value

# # Load the saved model with custom loss functions
# weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
# dice_loss = sm.losses.DiceLoss(class_weights=weights)
# focal_loss = sm.losses.CategoricalFocalLoss()
# total_loss = dice_loss + (1 * focal_loss)

# satellite_model = load_model('dubai_seg.h5',
#                              custom_objects={'dice_loss_plus_1focal_loss': total_loss,
#                                              'jaccard_coef': jaccard_coef})

# # Function to perform segmentation prediction
# def predict_segmentation(image):
#     image = image.resize((256, 256))
#     image_array = np.array(image) / 255.0  # Normalize the image
#     image_array = np.expand_dims(image_array, 0)
#     prediction = satellite_model.predict(image_array)
#     predicted_image = np.argmax(prediction, axis=3)
#     return predicted_image[0, :, :]


# def main():
#     st.title("Satellite Image Segmentation App")

#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

#     if uploaded_file is not None:
#         # Display the uploaded image
#         original_image = Image.open(uploaded_file)
#         # st.image(original_image, caption="Uploaded Image", use_column_width=True)

#         # Perform segmentation prediction
#         st.subheader("Segmentation Prediction")
#         predicted_image = predict_segmentation(original_image)

#         # Display the original and predicted images side by side
#         col1, col2 = st.columns(2)

#         # Display original image in col1
#         col1.image(original_image, caption="Original Image", use_column_width=True)

#         # Display predicted image using Matplotlib in col2
#         col2.subheader("Predicted Image")
#         fig, ax = plt.subplots()
#         ax.imshow(predicted_image, cmap='viridis')  # Choose a suitable colormap
#         ax.axis('off')  # Turn off axis for better appearance
#         col2.pyplot(fig, use_container_width=True)  # Ensure equal width for col2

# if __name__ == "__main__":
#     main()


import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import segmentation_models as sm
from keras import backend as K
import matplotlib.pyplot as plt
from util import classify, set_background

import warnings 

# Settings the warnings to be ignored 
warnings.filterwarnings('ignore') 


background_image = '0.png'
set_background(background_image)

# Define custom loss functions
def jaccard_coef(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    final_coef_value = (intersection + 1.0) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection + 1.0)
    return final_coef_value

# Load the saved model with custom loss functions
weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

satellite_model = load_model('dubai_seg.h5',
                             custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                             'jaccard_coef': jaccard_coef})

# Function to perform segmentation prediction
def predict_segmentation(image):
    # Resize the image to (256, 256)
    image_resized = image.resize((256, 256))

    # Convert to array and normalize
    image_array = np.array(image_resized) / 255.0

    # Ensure the image has 3 channels (RGB)
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    # Expand dimensions to create a batch of size 1
    image_array = np.expand_dims(image_array, 0)

    # Perform prediction
    prediction = satellite_model.predict(image_array)

    # Get the predicted segmentation mask
    predicted_image = np.argmax(prediction, axis=3)

    return predicted_image[0, :, :]


def main():
    st.title("Satellite Image Segmentation App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        original_image = Image.open(uploaded_file)

        # Perform segmentation prediction
        # st.subheader("Segmentation Prediction")

        # Create two columns
        col1, col2 = st.columns(2)

        # Display original image in col1
        with col1:
            col1.subheader("Uploaded Image")
            st.write("  ")
            col1.image(original_image, caption="Original Image", use_column_width=True)

        # Display predicted image using Matplotlib in col2
        with col2:
            col2.subheader("Predicted Image")
            predicted_image = predict_segmentation(original_image)
            fig, ax = plt.subplots()
            ax.imshow(predicted_image, cmap='viridis')  # Choose a suitable colormap
            ax.axis('off')  # Turn off axis for better appearance
            col2.pyplot(fig, use_container_width=True)  # Ensure equal width for col2

if __name__ == "__main__":
    main()
