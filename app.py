import os
import pandas as pd
import pickle
from flask import Flask, render_template, request
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

import joblib
import sklearn.externals




app = Flask(__name__, template_folder="template")
model = pickle.load(open("dtc.pkl", "rb"))
print("Model Loaded")



app = Flask(__name__)
app.config['upload folder']='uploads'

@app.route('/')
def home():
    return render_template('index.html')
global path



@app.route('/prediction',methods = ['POST','GET'])
def prediction():
    if request.method == 'POST':
        a = float(request.form['a'])
        b = float(request.form['b'])
        c = float(request.form['c'])
        d = float(request.form['d'])
        e = float(request.form['e'])
        f = float(request.form['f'])
        g = float(request.form['g'])
        h = float(request.form['h'])
        i = float(request.form['i'])
        # print('ads')
        values = [[float(a),float(b),float(c),float(d),float(e),float(f),float(g),float(h),float(i)]]

        pred = model.predict(values)
        # print('asdfg')

        return render_template('prediction.html',msg ='success',result = pred)
    return render_template('prediction.html')

if __name__ == '__main__':
            app.run(debug=True)