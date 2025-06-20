import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

with open('C:/Projects/AIML/ANN/one_hot_encoderv2.pkl','rb') as file:
    one_hot_encoder = pickle.load(file)

with open ('C:/Projects/AIML/ANN/label_encoder.pkl','rb') as file:
    label_encoder = pickle.load(file)

with open('C:/Projects/AIML/ANN/scaler.pkl','rb') as file:
    scaler= pickle.load(file)

model = tf.keras.models.load_model('C:/Projects/AIML/ANN/model.h5')

st.title('Customer Churn Prediction')

# input data

geography = st.selectbox('Geography', one_hot_encoder.categories_[0])
gender = st.selectbox('Gender',label_encoder.classes_)
age= st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score= st.number_input('credit score')
estimated_salary= st.number_input('Estimated salary')
tenure=st.slider('Tenure',0,10)
num_of_products = st.slider('Number of products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member= st.selectbox('Is Active Member', [0,1])

input_data= pd.DataFrame({
'CreditScore' : [credit_score],
'Gender': [label_encoder.transform([gender])[0]],
'Age': [age],
'Tenure': [tenure],
'Balance': [balance],
'NumOfProducts': [num_of_products],
'HasCrCard': [has_cr_card],
'IsActiveMember': [is_active_member],
'EstimatedSalary': [estimated_salary]
})


geo_encoded =  one_hot_encoder.transform([[geography]]).toarray()
geo_encoded_df= pd.DataFrame(geo_encoded,columns= one_hot_encoder.get_feature_names_out(['Geography']))

input_data= pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
#input_df= pd.concat([input_df.drop('Geography',axis=1),OneHotGeo_df],axis=1)

input_scaler = scaler.transform(input_data)

prediction = model.predict(input_scaler)
prediction_prob= prediction[0][0]
st.write("Churn prob", prediction_prob)
if prediction_prob>0.5:
    st.write("This will churn")
else:
    st.write("Not churn")

    