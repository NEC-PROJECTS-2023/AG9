import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import pickle
from math import radians,cos,sin,asin,sqrt
import folium
import datetime
from folium.plugins import HeatMap
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.tree import DecisionTreeClassifier

matplotlib.rcParams.update({'font.size': 12})
uber_data=pd.read_csv("uber-raw-data-sep14.csv")

x,y=train_test_split(uber_data,test_size=0.2,random_state=100)
uber_data['Date/Time'] = pd.to_datetime(uber_data['Date/Time'])
rush = uber_data.groupby(["Lat", "Lon"])["Date/Time"].count().reset_index()
rush.columns = ["Lat", "Lon", "Number of trips"]
uber_data["month"]=uber_data["Date/Time"].apply(lambda x:x.month)
uber_data["day"]=uber_data["Date/Time"].apply(lambda x:x.day)
uber_data["weekdays"]=uber_data["Date/Time"].apply(lambda x:x.weekday())
uber_data["hour"]=uber_data["Date/Time"].apply(lambda x:x.hour)

df_day_grouped = uber_data.groupby(['day']).count()

#Creating the grouped DataFrame
df_day = pd.DataFrame({'Number_of_trips':df_day_grouped.values[:,0]}, index = df_day_grouped.index) 

df_weekday_grouped = uber_data.groupby(['weekdays'], sort = False).count()

#Creating the grouped DataFrame
df_weekday = pd.DataFrame({'Number_of_trips':df_weekday_grouped.values[:,0]}, index = df_weekday_grouped.index) 
clus = uber_data[['Lat', 'Lon']]
kmeans = KMeans(n_clusters = 6, random_state = 0)
kmeans.fit(clus)
centroids = kmeans.cluster_centers_
clocation = pd.DataFrame(centroids, columns = ['Latitude', 'Longitude'])
label = kmeans.labels_
data_new = uber_data.copy()
data_new['Clusters'] = label
feature_cols=['Lon','Lat']
X=uber_data[feature_cols]
y=uber_data['Base']
uber_data=uber_data.drop('Date/Time',axis=1)

X_train,X_test,y_train,y_Test=train_test_split(X,y,test_size=0.3,random_state=1)
clf=DecisionTreeClassifier()
clf1=clf.fit(X_train,y_train)

filename='model.pkl'
pickle.dump(clf1,open(filename,'wb'))