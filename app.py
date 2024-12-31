import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Adjust this number based on your system

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the models and encoders
model = joblib.load('depression_model_best.pkl')
scaler = joblib.load('scaler_best.pkl')
label_encoder_new_degree = joblib.load('label_encoder_new_degree_best.pkl')
label_encoder_depression = joblib.load('label_encoder_depression_best.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/depress')
def depress():
    return render_template('depress.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    print("Form data received:", data)  # Debugging statement
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
    except ValueError as e:
        return jsonify({'error': f"Error in input data: {e}"})

    # Ensure the columns are in the correct order
    column_order = ['Gender', 'Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Sleep Duration', 
                    'Dietary Habits', 'Have you ever had suicidal thoughts ?', 'Work/Study Hours', 'Financial Stress', 
                    'Family History of Mental Illness', 'New_Degree']
    df = df[column_order]

    # Scale numerical features
    numerical_features = ['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Sleep Duration', 
                          'Work/Study Hours', 'Financial Stress']
    df[numerical_features] = scaler.transform(df[numerical_features])

    # Predict the result
    prediction = model.predict(df)
    prediction_label = label_encoder_depression.inverse_transform(prediction)[0]

    # Convert prediction_label to a native Python type
    prediction_label = int(prediction_label)

    return jsonify({'prediction': prediction_label})

if __name__ == '__main__':
    app.run(debug=True)