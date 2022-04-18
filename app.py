# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:55:56 2020

@author: Leela Pavan Kumar
"""

from flask import Flask, render_template, request
import pickle
import numpy as np
from statistics import mode
filename=str("model_rf.pkl")
classifier1=pickle.load(open(filename, 'rb'))
filename=str("model_et.pkl")
classifier2=pickle.load(open(filename, 'rb'))
filename=str("model_ada.pkl")
classifier3=pickle.load(open(filename, 'rb'))
filename=str("model_gb.pkl")
classifier4=pickle.load(open(filename, 'rb'))
filename=str("model_svc.pkl")
classifier5=pickle.load(open(filename, 'rb'))
filename=str("model_xgb.pkl")
classifier6=pickle.load(open(filename, 'rb'))

app=Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = int(request.form['gender'])
        age = int(request.form['age'])
        educ = float(request.form['educ'])
        ses = float(request.form['ses'])
        mmse = float(request.form['mmse'])
        cdr = float(request.form['cdr'])
        etiv = float(request.form['etiv'])
        nwbv = float(request.form['nwbv'])
        asf = float(request.form['asf'])
        data = np.array([[gender, age, educ, ses, mmse, cdr, etiv, nwbv, asf]])
        my_prediction=[]
        a=classifier1.predict(data)
        a=a.tolist()
        my_prediction.append(a[0])
        a=classifier2.predict(data)
        a=a.tolist()
        my_prediction.append(a[0])
        a=classifier3.predict(data)
        a=a.tolist()
        my_prediction.append(a[0])
        a=classifier4.predict(data)
        a=a.tolist()
        my_prediction.append(a[0])
        #a=classifier5.predict(data)
        #a=a.tolist()
        my_prediction.append(0)
        a=classifier6.predict(data)
        a=a.tolist()
        my_prediction.append(a[0])
        return render_template('results.html', prediction=mode(my_prediction))

if __name__ == '__main__':
	app.run()
