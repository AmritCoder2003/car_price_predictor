from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    name = request.form['name']
    company = request.form['company']
    year = int(request.form['year'])
    kms_driven = int(request.form['kms_driven'])
    fuel_type = request.form['fuel_type']
    
    # Make prediction
    input_data = pd.DataFrame([[name, company, year, kms_driven, fuel_type]],
                              columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    prediction = model.predict(input_data)[0]
    
    # Return result
    return render_template('index.html', prediction_text=f'Estimated Car Price: ₹{int(prediction)}')

if __name__ == '__main__':
    app.run(debug=True)
