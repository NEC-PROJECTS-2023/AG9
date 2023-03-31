from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
import pickle
import folium

app=Flask(__name__)
filename='model.pkl'
cls=pickle.load(open(filename,'rb'))
@app.route('/')
def home():
    return render_template('input.html')

@app.route('/search', methods=['GET','POST'])
def search():
    if request.method=='POST':
        c=request.form['Lat']
        d=request.form['Lon']
        data=np.array([(c,d)])
        my_prediction=cls.predict(data)

        

        """for i in load_model:               
            if load_model[1][i]==c and load_model[2][i]==d:
                print(load_model[Base])"""

    return render_template('map.html',prediction=my_prediction)

if __name__=='__main__':
    app.run(debug=False)