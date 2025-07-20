import pickle
import pandas as pd
import numpy as np
import os


def load_model_components():
    """Load all necessary model components from files"""
    components = {}
    required_files = [
        'models/knn_model.pkl',
        'models/nb_model.pkl',
        'models/scaler.pkl',
        'models/label_encoders.pkl',
        'models/selected_features.pkl'
    ]

    try:
        with open('models/knn_model.pkl', 'rb') as f:
            components['knn_model'] = pickle.load(f)
        with open('models/nb_model.pkl', 'rb') as f:
            components['nb_model'] = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            components['scaler'] = pickle.load(f)
        with open('models/label_encoders.pkl', 'rb') as f:
            components['label_encoders'] = pickle.load(f)
        with open('models/selected_features.pkl', 'rb') as f:
            components['selected_features'] = pickle.load(f)
        return components

    except Exception as e:
        print(f"ERROR loading model components: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def predict_cancer_risk(patient_data, model_components, model_type='both'):
    """
    Predict colorectal cancer risk for a new patient

    Args:
        patient_data: Dictionary of feature values for a new patient
        model_components: Dictionary containing all model components
        model_type: 'knn', 'nb', or 'both' (default)

    Returns:
        Dictionary containing predictions and probabilities
    """
    try:
        # Unpack components
        knn = model_components['knn_model']
        nb = model_components['nb_model']
        scaler = model_components['scaler']
        selected_features = model_components['selected_features']

        # Create DataFrame with the exact column structure
        X_test_new = pd.DataFrame(columns=selected_features)

        # Fill in the values
        for feature in selected_features:
            if feature in patient_data:
                X_test_new.at[0, feature] = patient_data[feature]
            else:
                X_test_new.at[0, feature] = 0

        # Scale numerical features
        numeric_cols = X_test_new.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            X_test_new[numeric_cols] = scaler.transform(X_test_new[numeric_cols])

        # Make predictions
        results = {}

        if model_type in ['knn', 'both']:
            knn_prob = knn.predict_proba(X_test_new)[0, 1]
            knn_pred = knn.predict(X_test_new)[0]
            results['knn'] = {
                'prediction': int(knn_pred),
                'probability': float(knn_prob),
                'label': 'Cancer' if knn_pred == 1 else 'No Cancer',
                'probability_percent': f"{float(knn_prob) * 100:.1f}%"
            }

        if model_type in ['nb', 'both']:
            nb_prob = nb.predict_proba(X_test_new)[0, 1]
            nb_pred = nb.predict(X_test_new)[0]
            results['nb'] = {
                'prediction': int(nb_pred),
                'probability': float(nb_prob),
                'label': 'Cancer' if nb_pred == 1 else 'No Cancer',
                'probability_percent': f"{float(nb_prob) * 100:.1f}%"
            }

        if model_type == 'both':
            avg_prob = (results['knn']['probability'] + results['nb']['probability']) / 2
            avg_pred = 1 if avg_prob >= 0.5 else 0
            results['ensemble'] = {
                'prediction': avg_pred,
                'probability': avg_prob,
                'label': 'Cancer' if avg_pred == 1 else 'No Cancer',
                'probability_percent': f"{float(avg_prob) * 100:.1f}%"
            }

        return results

    except Exception as e:
        import traceback
        print(f"Error in prediction function: {str(e)}")
        print(traceback.format_exc())
        return None