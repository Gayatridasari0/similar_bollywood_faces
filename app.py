# import os
# import pickle
#
# actorFiles = os.listdir('Bollywood_celeb_face_localized')
# print(actorFiles)
#
# filenames = []
#
# for actorFile in actorFiles:
#     for actor in os.listdir(os.path.join('Bollywood_celeb_face_localized',actorFile)):
#         for file in os.listdir(os.path.join('Bollywood_celeb_face_localized', actorFile,actor)):
#             filenames.append(os.path.join('Bollywood_celeb_face_localized',actorFile,actor,file))
#
# print(filenames)
# print(len(filenames))
#
# pickle.dump(filenames,open('filenames.pkl','wb'))
# commented because the work is done
import os
import pickle

import cv2
import keras.src.applications.resnet
import numpy as np
import streamlit as st
from PIL import Image
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity

detector = MTCNN()
model = keras.src.applications.resnet.ResNet50(
    include_top=False,
    input_shape=(224, 224, 3),
    pooling='avg',
    weights='imagenet'
)

feature_list = pickle.load(open('embeddings.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

st.title("Which bollywood celebrity are you?")
uploaded_img = st.file_uploader('Choose an image')
print(uploaded_img)


def save_uploaded_img(uploaded_img):
    try:
        with open(uploaded_img.name, 'wb') as f:
            f.write(uploaded_img.getbuffer())
        return True
    except:
        return False


def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"Image at path {img_path} could not be loaded.")

    results = detector.detect_faces(img)

    if not results:
        raise IndexError("No faces detected in the image.")

    x, y, width, height = results[0]['box']

    face = img[y:y + height, x:x + width]

    #  extract its features
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)

    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = keras.applications.resnet50.preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result


def recommend(feature_list, features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos


if uploaded_img is not None:
    if save_uploaded_img(uploaded_img):
        display_image = Image.open(uploaded_img)
        st.header('Image Uploaded!, Processing...')

        features = extract_features(uploaded_img.name, model, detector)

        index_pos = recommend(feature_list, features)
        predict_actor = filenames[index_pos]

        display_image_resized = display_image.resize((150, 150))
        col1, col2 = st.columns(2)
        # print(predict_actor)
        with col1:
            st.subheader('Your uploaded image')
            st.image(display_image_resized, width=150, caption='Uploaded Image')
        with col2:
            predicted_image_path = predict_actor.replace("\\", "/")  # convert to Unix-style path
            if os.path.exists(predicted_image_path):
                predicted_image = Image.open(predicted_image_path)
                st.subheader(f'Look like: {" ".join(os.path.split(predicted_image_path)[-2].split("_"))}')
                st.image(predicted_image, width=150, caption='Predicted Look Alike Image')
            else:
                st.error(f"Image not found at path: {predicted_image_path}")
            # st.subheader('Look like: ' + " ".join(predict_actor))
            # # st.subheader('Look like: ' + " ".join(predict_actor.split("/")[2].split("_")))
            # st.image(filenames[index_pos], width=150, caption='Predicted Look Alike Image')
