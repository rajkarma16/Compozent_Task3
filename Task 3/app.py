from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and feature list
model = joblib.load(r'C:\Users\Raj Vishwakarma\Desktop\Compozent Task\Task 3\car_price_model.pkl')
feature_list = joblib.load(r'C:\Users\Raj Vishwakarma\Desktop\Compozent Task\Task 3\feature_list.pkl') # Load saved feature list

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form

        # Create a dictionary of input values
        input_data = {
            'Year': int(data['year']),
            'Kilometer': int(data['kilometer']),
            'Engine': float(data['engine']),
            'Max Power': float(data['max_power']),
            'Max Torque': float(data['max_torque']),
            'Length': float(data['length']),
            'Width': float(data['width']),
            'Height': float(data['height']),
            'Seating Capacity': float(data['seating_capacity']),
            'Fuel Tank Capacity': float(data['fuel_tank_capacity']),
            'Fuel Type_' + data['fuel_type']: 1,
            'Transmission_' + data['transmission']: 1,
            'Owner_' + data['owner_type']: 1,
        }

        # Fill missing categorical features with 0 (to match one-hot encoding during training)
        for feature in feature_list:
            if feature not in input_data:
                input_data[feature] = 0

        # Convert input data to DataFrame and align with feature list
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_list] # Ensure the same column order

        # Predict price
        prediction = model.predict(input_df)
        predicted_price = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'Estimated Car Price: {predicted_price}')
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)