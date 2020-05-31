# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:24:28 2020

@author: Bagavathi Priya
"""


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__,template_folder='D:/Education/Salary price prediction/templates', static_folder='D:/Education/Salary price prediction/static')
model = pickle.load(open('D:/Education/Salary price prediction/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('D:/Education/Salary price prediction/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    @app.route('/')
    def home():
        return render_template('index.html')
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=False)
    

    