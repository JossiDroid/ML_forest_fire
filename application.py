from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd  
import pickle 
import os 

application = Flask(__name__)
app = application


ridge_model = pickle.load(open("C:/Users/HM869LV/OneDrive - EY/Documents/Data_Science_Projects/Flask_deployment/models/ridge.pkl", "rb"))
scaler_model = pickle.load(open("C:/Users/HM869LV/OneDrive - EY/Documents/Data_Science_Projects/Flask_deployment/models/scaler.pkl", "rb"))

@app.route('/')
def index():
    print(request)
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_dataset():
    print(request.method)
    if request.method == "POST":
        data = [float(request.form.get(key)) for key in ['temperature', 'rh', 'ws', 'rain', 'ffmc', 'dmc','isi','classes', 'region']]
        scaled_data = scaler_model.transform([data])
        prediction = ridge_model.predict(scaled_data)
        return render_template('home.html', prediction=prediction[0])
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
