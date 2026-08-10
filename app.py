from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

# Load the trained model and preprocessing objects
try:
    model = joblib.load('yield_prediction_model.joblib')
    scaler = joblib.load('scaler.joblib')
    label_encoders = joblib.load('label_encoders.joblib')
    print("Model and preprocessing objects loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Load the dataset for analysis
try:
    df = pd.read_csv('dataset_for_yield_prediction.csv')
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    raise

@app.errorhandler(Exception)
def handle_exception(e):
    # Handle HTTP exceptions
    if isinstance(e, HTTPException):
        response = e.get_response()
        response.data = json.dumps({
            "code": e.code,
            "name": e.name,
            "description": e.description,
        })
        response.content_type = "application/json"
        return response

    # Handle other exceptions
    return jsonify({
        "success": False,
        "error": str(e)
    }), 500

@app.route('/')
def home():
    try:
        # Calculate some statistics for the dashboard
        stats = {
            'total_farms': len(df),
            'crop_types': df['Crop_Type'].nunique(),
            'avg_yield': round(df['Yield(tons)'].mean(), 2),
            'max_yield': round(df['Yield(tons)'].max(), 2),
            'min_yield': round(df['Yield(tons)'].min(), 2)
        }
        
        # Get unique values for dropdowns
        crop_types = sorted(df['Crop_Type'].unique().tolist())
        irrigation_types = sorted(df['Irrigation_Type'].unique().tolist())
        soil_types = sorted(df['Soil_Type'].unique().tolist())
        seasons = sorted(df['Season'].unique().tolist())
        
        return render_template('index.html', 
                             stats=stats,
                             crop_types=crop_types,
                             irrigation_types=irrigation_types,
                             soil_types=soil_types,
                             seasons=seasons)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error loading dashboard: {str(e)}"
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate input data
        required_fields = ['crop_type', 'farm_area', 'irrigation_type', 'fertilizer', 
                          'pesticide', 'soil_type', 'season', 'water_usage']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"Missing required field: {field}"
                }), 400
        
        # Validate numeric fields
        try:
            farm_area = float(data['farm_area'])
            fertilizer = float(data['fertilizer'])
            pesticide = float(data['pesticide'])
            water_usage = float(data['water_usage'])
        except ValueError:
            return jsonify({
                "success": False,
                "error": "Invalid numeric values provided"
            }), 400
        
        # Validate ranges
        if farm_area <= 0 or fertilizer < 0 or pesticide < 0 or water_usage < 0:
            return jsonify({
                "success": False,
                "error": "Values must be positive"
            }), 400
        
        # Create a DataFrame with the input data
        input_data = pd.DataFrame([{
            'Crop_Type': data['crop_type'],
            'Farm_Area(acres)': farm_area,
            'Irrigation_Type': data['irrigation_type'],
            'Fertilizer_Used(tons)': fertilizer,
            'Pesticide_Used(kg)': pesticide,
            'Soil_Type': data['soil_type'],
            'Season': data['season'],
            'Water_Usage(cubic meters)': water_usage
        }])
        
        # Create a copy of the input data for encoding
        input_encoded = input_data.copy()
        
        # Add derived features
        input_encoded['Fertilizer_per_acre'] = input_encoded['Fertilizer_Used(tons)'] / input_encoded['Farm_Area(acres)']
        input_encoded['Pesticide_per_acre'] = input_encoded['Pesticide_Used(kg)'] / input_encoded['Farm_Area(acres)']
        input_encoded['Water_per_acre'] = input_encoded['Water_Usage(cubic meters)'] / input_encoded['Farm_Area(acres)']
        
        # Encode categorical variables using the appropriate encoder for each column
        for column in ['Crop_Type', 'Irrigation_Type', 'Soil_Type', 'Season']:
            # Get unique values from training data
            unique_values = df[column].unique()
            # Replace unseen categories with the most common category
            input_encoded[column] = input_encoded[column].apply(
                lambda x: x if x in unique_values else df[column].mode()[0]
            )
            # Now encode the values using the appropriate encoder
            input_encoded[column] = label_encoders[column].transform(input_encoded[column])
        
        # Ensure the input data has all required features in the same order as training
        required_features = ['Crop_Type', 'Farm_Area(acres)', 'Irrigation_Type', 'Fertilizer_Used(tons)', 
                           'Pesticide_Used(kg)', 'Soil_Type', 'Season', 'Water_Usage(cubic meters)',
                           'Fertilizer_per_acre', 'Pesticide_per_acre', 'Water_per_acre']
        input_encoded = input_encoded[required_features]
        
        # Scale the features
        input_scaled = scaler.transform(input_encoded)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Calculate confidence interval (assuming normal distribution)
        std_dev = np.std(df['Yield(tons)'])
        confidence_interval = {
            'lower': round(prediction - 1.96 * std_dev, 2),
            'upper': round(prediction + 1.96 * std_dev, 2)
        }
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'confidence_interval': confidence_interval
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 