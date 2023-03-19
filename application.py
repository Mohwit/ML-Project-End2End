from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predict', methods=['POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    
    else:
        pass