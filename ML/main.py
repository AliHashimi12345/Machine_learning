import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import sweetviz as sv
from geopy.geocoders import Nominatim
#data set

from sklearn.datasets import fetch_california_housing 
data = fetch_california_housing()
print(data.DESCR)

df = pd.DataFrame(data = data.data, columns = data.feature_names)
print(df.head())


#EXPLORING DATA ANALYSIS

report =  sv.analyze(df)
report.show_html("./report.html")

#data pre-processing

#using feature engineering to tell location of the house

geolocator = Nominatim(user_agent= 'geaoapiExercises')
print(geolocator.reverse("37.88"  +  ","  +  "-122.23")[0])



def location(cord):
    Latitude  =  str(cord[0])
    Longitude  = str(cord[1])

    location = geolocator.reverse(Latitude+","+Longitude).raw['address']

if location.get('Road') is None:
    location['Road'] = None

if location.get('Country') is None:
    location['Country'] = None

loc_update['Country'].append(location['Country'])
loc_update['Road'].append(location['Road'])   

#loc_update = {"Country":[],"Road":[],"Neighbourhood":[]}

#for i,cord in enumerate(df.iloc[:,6:-1].values):
 #   location(cord)
  #  pickle.dump(loc_update,open('loc_update.pickle','wb'))

   # if i%100 == 0:
    #    print(i)
