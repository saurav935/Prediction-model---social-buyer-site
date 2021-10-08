# -*- coding: utf-8 -*-


from flask import Flask,request
import pandas as pd
import numpy as np
import pickle


app = Flask(__name__)
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)


@app.route('/')
def welcome():
    return "Welcome All"


@app.route('/predict')
def predict_purchasing_decision():
    Age = request.args.get('Age')
    EstimatedSalary = request.args.get('EstimatedSalary')
    prediction = classifier.predict([[Age,EstimatedSalary]])
    return "The predicted value is" + str(prediction)




if __name__ == '__main__':
    app.run()
    
 