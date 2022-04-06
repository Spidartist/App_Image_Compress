import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import cv2

st.set_page_config(
    page_title="Image Compressor Apps",
    page_icon=":spider:",
    menu_items={
        'Get Help': 'https://www.facebook.com/spidartist',
        'Report a bug': "https://github.com/Spidartist/",
        'About': "### This is my first app about *Image Compression* :spider:!\n  #### Just input the image and get the result :shark:\n Author:  Quan Hoang Danh :heart:"
    },
    layout="wide"
    # initial_sidebar_state="expanded",
)


@st.cache
def compress(img, K):
    width = img.shape[0]
    height = img.shape[1]
    img = img.reshape(width * height, 3)

    k_mean = KMeans(n_clusters=K).fit(img)

    labels = k_mean.predict(img)

    clusters = k_mean.cluster_centers_

    img2 = np.zeros((width, height, 3), dtype=np.uint8)
    index = 0
    for i in range(width):
        for j in range(height):
            label_of_pixel = labels[index]
            img2[i][j] = clusters[label_of_pixel]
            index += 1
    return img2

st.header("Input the image")
uploaded_file = st.file_uploader("Choose a image file")

if uploaded_file is not None:

    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.resize(opencv_image, (256, 256))
    K = 3

    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR")
    K = st.slider("Number of colors", 1, 44, 3)
    result = compress(opencv_image, K)
    h, w, _ = opencv_image.shape
    white_line = np.ones((h, 100, 3)) * 255.0
    all_images = [
        opencv_image,
        white_line,
        result
    ]
    st.subheader("Result")
    display_img = np.concatenate(all_images, axis=1)
    st.image(display_img / 255.0, channels="BGR", clamp=True)
    
    result_2 = result/255.0
#     result_2 = Image.fromarray(result_2, 'RGB')
    success, encoded_image = cv2.imencode('.png', result_2)
    content = encoded_image.tobytes()
    btn = st.download_button(
             label="Download compressed image",
             data=content,
             file_name="a.png",
             mime="image/png"
           )


