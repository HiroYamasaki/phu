import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Set the number of cores to use for joblib
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Adjust this number based on your system

# Load the models and encoders
model = joblib.load('depression_model_best.pkl')
scaler = joblib.load('scaler_best.pkl')
label_encoder_new_degree = joblib.load('label_encoder_new_degree_best.pkl')
label_encoder_depression = joblib.load('label_encoder_depression_best.pkl')

# City encoding
city_encoding = {
    'Agra': 0, 'Ahmedabad': 1, 'Bangalore': 2, 'Bhopal': 3, 'Chennai': 4, 'Delhi': 5, 'Faridabad': 6, 'Ghaziabad': 7,
    'Hyderabad': 8, 'Indore': 9, 'Jaipur': 10, 'Kalyan': 11, 'Kanpur': 12, 'Kolkata': 13, 'Lucknow': 14, 'Ludhiana': 15,
    'Meerut': 16, 'Mumbai': 17, 'Nagpur': 18, 'Nashik': 19, 'Patna': 20, 'Pune': 21, 'Rajkot': 22, 'Srinagar': 23,
    'Surat': 24, 'Thane': 25, 'Vadodara': 26, 'Varanasi': 27, 'Vasai-Virar': 28, 'Visakhapatnam': 29
}

# Streamlit app
st.title("Depression Prediction")

# Form for user input
with st.form("prediction_form"):
    st.header("Enter your details")
    
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    age = st.number_input("Age", min_value=18, max_value=34, step=1)
    academic_pressure = st.slider("Academic Pressure", min_value=0, max_value=5, step=1)
    cgpa = st.number_input("CGPA (Cumulative Grade Point Average)", min_value=0.0, max_value=10.0, step=0.01)
    study_satisfaction = st.slider("Study Satisfaction", min_value=0, max_value=5, step=1)
    sleep_duration = st.selectbox("Sleep Duration", options=[0, 1, 2, 3], format_func=lambda x: ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"][x])
    work_study_hours = st.slider("Work/Study Hours", min_value=0, max_value=12, step=1)
    financial_stress = st.slider("Financial Stress", min_value=0, max_value=5, step=1)
    dietary_habits = st.selectbox("Dietary Habits", options=[0, 1, 2], format_func=lambda x: ["Healthy", "Unhealthy", "Moderate"][x])
    new_degree = st.selectbox("New Degree", options=[0, 1, 2], format_func=lambda x: ["Graduated", "Post Graduated", "Higher Secondary"][x])
    suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    family_history = st.selectbox("Family History of Mental Illness", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    city = st.selectbox("City", options=list(city_encoding.keys()))

    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    data = {
        'gender': gender,
        'age': age,
        'academic_pressure': academic_pressure,
        'cgpa': cgpa,
        'study_satisfaction': study_satisfaction,
        'sleep_duration': sleep_duration,
        'work_study_hours': work_study_hours,
        'financial_stress': financial_stress,
        'dietary_habits': dietary_habits,
        'new_degree': new_degree,
        'suicidal_thoughts': suicidal_thoughts,
        'family_history': family_history,
        'city': city
    }
    
    df = pd.DataFrame([data])

    # Preprocess the input data
    try:
        df['Gender'] = df['gender'].astype(int)
        df['Age'] = df['age'].astype(int)
        df['Academic Pressure'] = df['academic_pressure'].astype(float)
        df['CGPA'] = df['cgpa'].astype(float)
        df['Study Satisfaction'] = df['study_satisfaction'].astype(float)
        df['Sleep Duration'] = df['sleep_duration'].astype(int)
        df['Work/Study Hours'] = df['work_study_hours'].astype(float)
        df['Financial Stress'] = df['financial_stress'].astype(float)
        df['Dietary Habits'] = df['dietary_habits'].astype(int)
        df['New_Degree'] = df['new_degree'].astype(int)
        df['Have you ever had suicidal thoughts ?'] = df['suicidal_thoughts'].astype(int)
        df['Family History of Mental Illness'] = df['family_history'].astype(int)
        df['City_encoded'] = df['city'].map(city_encoding).astype(int)

        # Ensure CGPA is a float
        df['CGPA'] = df['CGPA'].astype(float)

        # Cluster CGPA into specified ranges
        bins = [0, 5, 6, 7, 8, 9, 10]
        labels = ['<5', '5 - <6', '6 - <7', '7 - <8', '8 - <9', '9 - 10']
        df['CGPA_range'] = pd.cut(df['CGPA'], bins=bins, labels=labels, right=False)

        # Print the CGPA range
        st.write(f"CGPA {df['CGPA'].values[0]} falls into range {df['CGPA_range'].values[0]}")

        # Convert CGPA labels to numerical values
        cgpa_map = {'<5': 0, '5 - <6': 1, '6 - <7': 2, '7 - <8': 3, '8 - <9': 4, '9 - 10': 5}
        df['CGPA'] = df['CGPA_range'].map(cgpa_map)
    except ValueError as e:
        st.error(f"Error in input data: {e}")
    
    # Ensure the columns are in the correct order
    column_order = ['Gender', 'Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Sleep Duration', 
                    'Dietary Habits', 'Have you ever had suicidal thoughts ?', 'Work/Study Hours', 'Financial Stress', 
                    'Family History of Mental Illness', 'New_Degree', 'City_encoded']
    df = df[column_order]

    # Scale numerical features
    numerical_features = ['Gender', 'Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Sleep Duration', 
                          'Dietary Habits', 'Have you ever had suicidal thoughts ?', 'Work/Study Hours', 
                          'Financial Stress', 'Family History of Mental Illness', 'New_Degree', 'City_encoded']
    df[numerical_features] = scaler.transform(df[numerical_features])

    # Predict the result
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)

    prediction_label = label_encoder_depression.inverse_transform(prediction)[0]
    prediction_probability = prediction_proba[0][prediction[0]]

    # Convert prediction_label to a native Python type
    prediction_label = int(prediction_label)

    if prediction_label == 1:
        st.error(f"Depression Detected with probability {prediction_probability * 100:.2f}%")
    else:
        st.success(f"No Depression Detected with probability {prediction_probability * 100:.2f}%")