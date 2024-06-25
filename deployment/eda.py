# import libraries
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image


@st.cache_data
def load_data():
    df = pd.read_csv('train_trimmed.csv')
    return df


def app():
    st.title('Exploratory Data Analysis')
    st.write('---')

    data_load_state = st.text('Loading data...')
    df = load_data()
    data_load_state.text("Done!")

    st.subheader('Dataset Preview')
    st.write(df)

    st.subheader('**Data Analysis Questions**')
    st.write('---')

    st.write('### **1. What is the balance ratio between the target?**')
    st.write('''
             - **Insight:**
                - `32.7%` for `not_recommended` 
                - `67.3%` for `recommended`
                - our data is quite imbalanced 
             ''')
    image = Image.open('eda_1.png')
    st.image(image)

    st.write('### **2. How is the data distribution of user_review\'s word count?**')
    st.write('''
             - **Insight:**
                - The data distribution of user_review's word count is highly positively skewed (outliers on the right side)
             ''')
    image = Image.open('eda_2.png')
    st.image(image)

    st.write('### **3. What are the words that are associated with recommended user_suggestion?**')
    st.write('''
             - **Insight:**
                - Some words that's associated with recomended user_suggestion that take up a lot of space in our wordcloud are 'game', 'NOPE', 'play', 'early', 'access'
             ''')
    image = Image.open('eda_3.png')
    st.image(image)

