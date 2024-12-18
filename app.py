import streamlit as st
import tensorflow as tf
import numpy as np

st.title("Klasifikasi Citra")
upload = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

# Load MobileNetV2 model once
model_2 = tf.keras.models.load_model("./mobilenetv2_model.h5")

# Function to predict using MobileNetV2
def predict(img_path):
    class_names = ['Shrimp', 'Black_Sea_Sprat', 'Horse_Mackerel', 'Striped_Red_Mullet',
                   'Trout', 'Red_Mullet', 'Red_Sea_Bream', 'Sea_Bass', 'Gilt_Head_Bream']
    
    # Resize to the expected input size (224x224 for MobileNetV2)
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))  
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0  # Normalize the image (same as training)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    
    output = model_2.predict(img_array)
    score = tf.nn.softmax(output[0])
    
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)  # Confidence in percentage
    return predicted_class, confidence

if st.button("Predict", type="primary"):
    if upload is not None:
        st.image(upload)
        st.subheader("Hasil Prediksi menggunakan MobileNetV2")
        with st.spinner("Loading..."):
            predicted_class, confidence = predict(upload)
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.write("Please upload an image")
