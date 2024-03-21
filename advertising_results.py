import streamlit as st
import pandas as pd
import seaborn as sns
import pickle

st.write("# Advertising Sales Prediction App")
st.write("This app predicts the **Sales!**")

st.sidebar.header('User Input Parameters') #can only have 1 sidebar

def user_input_features():
    TV = st.sidebar.slider('TV Ads', 0.0, 300.0, 150.0) #(features,min,max,default)
    Radio = st.sidebar.slider('Radio Ads', 0.0, 100.0, 50.0)
    Newspaper = st.sidebar.slider('Newspaper Ads', 0.0, 100.0, 25.0)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
#st.write('## User Input parameters')
st.write(df)

loaded_model = pickle.load(open("ModelSales.h5", "rb")) #rb: read binary
new_pred = loaded_model.predict(df) # testing (examination)

st.subheader('Prediction')
st.write(new_pred)
