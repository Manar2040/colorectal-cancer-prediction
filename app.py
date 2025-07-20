from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import pandas as pd
import numpy as np
from helper import predict_cancer_risk, load_model_components

app = Flask(__name__)
app.secret_key = '5795835867#95797589057'

# Load model components on startup
model_components = load_model_components()


@app.route('/')
def index():
    # Get the list of features needed for the model
    selected_features = model_components['selected_features']

    # Get categorical feature options for dropdowns
    categorical_options = {}
    for col in ['Gender', 'Location', 'Dukes Stage']:
        if col in model_components['label_encoders']:
            categorical_options[col] = list(model_components['label_encoders'][col].classes_)

    return render_template('index.html',
                           features=selected_features,
                           categorical_options=categorical_options)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get data from form
            patient_data = {}

            # Process categorical features
            for col in ['Gender', 'Location', 'Dukes Stage']:
                if col in model_components['label_encoders'] and request.form.get(col):
                    value = request.form.get(col)
                    encoder = model_components['label_encoders'][col]
                    patient_data[col] = encoder.transform([value])[0]

            # Process numerical features
            numerical_features = [f for f in model_components['selected_features']
                                  if f not in ['Gender', 'Location', 'Dukes Stage']]

            for feature in numerical_features:
                if request.form.get(feature):
                    patient_data[feature] = float(request.form.get(feature))
                else:
                    patient_data[feature] = 0  # Default value

            # Make prediction
            predictions = predict_cancer_risk(patient_data, model_components)

            # Prepare data for display
            return render_template('results.html',
                                   predictions=predictions,
                                   patient_data=request.form)

        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'danger')
            return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)