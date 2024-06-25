# import libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
import re
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import TextVectorization


@st.cache_data
def load_data():
    df = pd.read_csv('train_trimmed.csv')
    return df


def app():
    st.title('Prediction for User Recommendation')
    st.write('---')

    data_load_state = st.text('Loading data...')
    df = load_data()
    data_load_state.text("Done!")
    
    st.subheader('Dataset Preview')
    st.write(df)

    # get input
    st.subheader('Input Data')
    st.write('---')
    input = user_input(df)

    st.subheader('User Input')
    st.table(input)

    # preprocess data
    data_preprocessed = input['user_review'].apply(lambda x: text_preprocessing(x))

    # load models
    vectorization_data = pickle.load(open('vectorizer.pkl', 'rb'))
    vectorizer = TextVectorization.from_config(vectorization_data['config'])
    vectorizer.set_weights(vectorization_data['weights'])
    model = tf.keras.models.load_model('model.h5')

    # vectorize data
    data_vect = vectorizer(data_preprocessed)

    # predict user_suggestion
    predicted_user_suggestion_proba = model.predict(data_vect)

    # show result
    predict_result = ''
    threshold = 0.75
    if predicted_user_suggestion_proba[0] > threshold:
        predict_result = 'Recommended'
    else:
        predict_result = 'Not_Recommended'

    st.subheader('Prediction Result:')
    st.write(f"predicted user suggestion proba: {predicted_user_suggestion_proba[0]}")
    st.write(f"predicted user suggestion: {predict_result}")

def user_input(df):
    user_review = st.text_area('''**Insert User Review Here:**''')

    data = {
        'user_review' : user_review
    }

    features = pd.DataFrame(data, index=[0])
    return features

# create a function for text preprocessing

def text_preprocessing(text):
    # define stopwords
    nltk_stopword = set(stopwords.words('english'))

    # define lemmatizer
    lemmatizer = WordNetLemmatizer()

    # case folding
    text = text.lower()

    # mention removal
    text = re.sub("@[a-za-z0-9_]+", " ", text)

    # hashtags removal
    text = re.sub("#[a-za-z0-9_]+", " ", text)

    # newline removal (\n)
    text = re.sub(r"\\n", " ",text)

    # whitespace removal
    text = text.strip()

    # url removal
    text = re.sub(r"http\s+", " ", text)
    text = re.sub(r"www.\s+", " ", text)

    # non-letter removal (such as emoticon, symbol (like μ, $, 兀), etc
    text = re.sub("[^a-za-z\s']", " ", text)
    text = re.sub("'", "", text)

    # tokenization
    tokens = word_tokenize(text)

    # stopwords removal
    tokens = [word for word in tokens if word not in nltk_stopword]

    # lemmatizing
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # combining tokens
    text = ' '.join(tokens)

    return text

