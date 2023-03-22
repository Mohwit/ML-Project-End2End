from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline,CustomData

application = Flask(__name__)

@application.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@application.route('/predict', methods=['POST'])
def predict():
    data = CustomData(
        gender= request.form.get('gender'),
        race_ethnicity= request.form.get('ethnicity'),
        parental_level_of_education= request.form.get('parental_level_of_education'),
        lunch= request.form.get('lunch'),
        test_preparation_course= request.form.get('test_preparation_course'),
        writing_score= request.form.get('writing_score'),
        reading_score= request.form.get('reading_score')
    )
    
    predict_df = data.getData_as_DataFrame()
    print(predict_df)
    
    predict_pipeline = PredictPipeline()
    result = predict_pipeline.predict(predict_df)
    
    return render_template('index.html', result = result)

if __name__=="__main__":
    application.run(host="0.0.0.0", debug=True) 