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
print(geolocator.reverse("37.88"  +  ","  +  "-122.23").raw['address'])



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

loc_update = {"Country":[],"Road":[],"Neighbourhood":[]}

for i,cord in enumerate(df.iloc[:,6:-1].values):

    location(cord)
    pickle.dump(loc_update,open('loc_update.pickle','wb'))

    if i%100 == 0:
        print(i)

for in in range(df.shape[0]):
    if df['Road'][i] is None:
        missing_idx.append(i)

missing_Road_X_train = np.array([[df['MedInc'][i].df['AverageRooms'][i],df['AveBedrms'][i]']]])

missing_Road_y_train = np.array[df('Road')][i] for i in range(df.shape[0]) if i not in missing_idx])
missing_Road_X_test = np.array()

from sklearn.linear_model import SGDClassifier
#model initialization

model_1 = SGDClassifier()
#MODEL TRAINING
model_1.fit(missing_Road_X_train,missing_Road_y_train)

missing_Road_X_test = model_1.predict(missing_Road_X_test)

#addd model back to dataa frame
for n,i  in enumerate(missing_idx):
    df['Road'][i] = missing_Road_y_pred[n]

print(df.info())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Road'] = le.fit_transform(df['Road'])
