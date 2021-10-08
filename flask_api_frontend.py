# -*- coding: utf-8 -*-


from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)


@app.route('/')
def welcome():
    return "Welcome All"


@app.route('/predict', methods=["Get"])
def predict_purchasing_decision():
    
    """Let's figure out the purchasing decision
    This is using docstrings for specifications.
    ---
    parameters:
        - name: Age
          in: query
          type: number
          required: true
        - name: EstimatedSalary
          in: query
          type: number
          required: true
    responses:
        100:
            description: The output values
    """
    Age = request.args.get('Age')
    EstimatedSalary = request.args.get('EstimatedSalary')
    prediction = classifier.predict([[Age,EstimatedSalary]])
    return "The predicted value is" + str(prediction)




if __name__ == '__main__':
    app.run()
    
 
