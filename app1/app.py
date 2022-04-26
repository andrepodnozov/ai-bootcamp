import streamlit as st
import pickle
from io import BytesIO
import requests
import pandas as pd 
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
# Code from Best Pipeline.py here


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('https://raw.githubusercontent.com/andrepodnozov/ai-bootcamp/main/app1/prepared_data.csv')
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -0.46883208314205954
exported_pipeline = GradientBoostingRegressor(alpha=0.95, learning_rate=0.1, loss="huber", max_depth=7, max_features=0.35000000000000003, min_samples_leaf=17, min_samples_split=20, n_estimators=100, subsample=0.8500000000000001)

exported_pipeline.fit(training_features, training_target)


######################
# User defined values
title = 'Streamlit demo Guild 18'
encoder_location = 'https://github.com/andrepodnozov/ai-bootcamp/blob/main/app1/encoder.pkl?raw=true'
target_encoder_location = ''
if len(encoder_location) > 5:
    mfile = BytesIO(requests.get(encoder_location).content)
    encoder = pickle.load(mfile)
    df = encoder.inverse_transform(features)
else:
    df = features.copy()
if len(target_encoder_location) > 5:
    mfile = BytesIO(requests.get(target_encoder_location).content)
    target_encoder = pickle.load(mfile)
st.title(title)
st.sidebar.header('User Input Parameters')
st.subheader('User Input parameters')
selected_data = dict()
for column in df.columns:
    if column != 'target':
        label = column.replace('_id.','')
        label = label.replace('_',' ').title()
        if df[column].dtype == 'O':
            selected_value = st.sidebar.selectbox(label, list(df[column].unique()))
        elif df[column].dtype == 'int64':
            selected_value = st.sidebar.number_input(label, min_value=df[column].min(), max_value=df[column].max(), value=df[column].iloc[0], step=1)
        elif df[column].dtype == 'float64':
            selected_value = st.sidebar.number_input(label, min_value=df[column].min(), max_value=df[column].max(), value=df[column].iloc[0])
        
        selected_data[column] = selected_value
test_data = pd.DataFrame(selected_data, index=[0])
st.write(test_data)
st.subheader('Prediction')
if len(encoder_location) > 5:
    test_data = encoder.transform(test_data) 
prediction = exported_pipeline.predict(test_data)
if len(target_encoder_location) > 5:
    prediction = target_encoder.inverse_transform(prediction)
if 'float' in str(type(prediction[0])):
    st.write(round(prediction[0],2))
else:
    st.write(prediction[0])
# Dataset
st.subheader('Data Set')
if len(target_encoder_location) > 5:
    df['target'] = target_encoder.inverse_transform(tpot_data['target'])
else:
    df['target'] = tpot_data['target']
st.write(df)
#pandas profling-report
### To-Do: Uncomment the below three lines of code if you want to see profiling report in your web app.
# st.subheader('Profiling Report of your dataset')
# pr = df.profile_report()
# st_profile_report(pr)
    