from get_skill import get_skills
from get_vacancy import recommend_job
from flask import Flask,request,jsonify
import pandas as pd
import numpy as np
import scipy.stats
import scipy.spatial
from sklearn.model_selection import cross_validate,KFold
import random
from sklearn.metrics import mean_squared_error
from math import sqrt
import math
import warnings
import sys
import csv



app = Flask(__name__)

@app.route("/get_vacancy",methods=["GET","POST"])
def vacancy():
    try:
        #return "aman"
        if request.method == 'POST':
            strng = request.values
            jsonData = request.get_json()
            result = recommend_job(jsonData)
            print(result)
            return jsonify({'vacancy' : result})
    except Exception as e:
         print("ll")
         return str(e)
         pass
    return "{'vacancy':result}"


@app.route("/get_skill",methods=["GET","POST"])
def pdf():
    try:
        #return "aman"
        if request.method == 'POST':
            strng = request.values
            jsonData = request.args['url']
            result = get_skills(jsonData)
            print(result)

            return jsonify({'skill' : result})
    except Exception as e:
         print("ll")
         return str(e)
         pass
    return "{'skill':result}"

if __name__ == "__main__":
    app.run(debug=True)
