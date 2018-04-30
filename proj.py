# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import ensemble
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

from sklearn.metrics import roc_curve,auc
from statsmodels.tools import categorical
import matplotlib.pylab as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing

from urllib.request import urlopen
from urllib.parse import urlencode
from bs4 import BeautifulSoup

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from collections import OrderedDict
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap
import mpl_toolkits

rcParams['figure.figsize'] = 12, 4

import pandas as pd
import numpy as np


#To create individual data sets for each Airline

data = pd.read_csv('flights.csv',low_memory = True)
airline = data.AIRLINE.unique()

for i in range(len(data.AIRLINE.value_counts())):
    data_t= data[data.AIRLINE == airline[i]]
    data_t.to_csv( str(airline[i])+'.csv')
    
#read airports data
airports = pd.read_csv("airports.csv",low_memory = True)

#read passenger data
pass_data=  pd.read_csv('cy15-commercial-service-enplanements.csv')

#merge passanger data with airport data
airport2 = pd.merge(airports , pass_data[['IATA_CODE','Hub','CY 15 Enplanements']],how = 'left', on = 'IATA_CODE')
airport2.to_csv('airports.csv',encoding = 'utf-8',index =False)

#read the Airline dataset
df0 = pd.read_csv('US.csv')

#details and types of variables in the dataset with null values
tab_info=pd.DataFrame(df0.dtypes).T.rename(index={0:'column type'})
tab_info=tab_info.append(pd.DataFrame(df0.isnull().sum()).T.rename(index={0:'null values (nb)'}))
tab_info=tab_info.append(pd.DataFrame(df0.isnull().sum()/df0.shape[0]*100)
                         .T.rename(index={0:'null values (%)'}))
tab_info.to_csv('data_info.csv')

#variables to remove which will not be used.
variables_to_remove = ['TAXI_OUT', 'TAXI_IN', 'WHEELS_ON', 'WHEELS_OFF', 'ELAPSED_TIME',
                       'AIR_SYSTEM_DELAY','SECURITY_DELAY', 'AIRLINE_DELAY', 
                       'LATE_AIRCRAFT_DELAY','WEATHER_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON']
df0.drop(variables_to_remove, axis = 1, inplace = True)

# creating date variable
df0['DATE'] = pd.to_datetime(df0[['YEAR','MONTH', 'DAY']])

#keeping only those variables where departure delay is populated
df0 = df0[np.isfinite(df0['DEPARTURE_DELAY'])]

#Reading airport dataset to get features and details for each airport
airports =  pd.read_csv('airports.csv')
airports.rename(columns ={'IATA_CODE' :'ORIGIN_AIRPORT' } ,inplace = True)

#list of departure airports present in the airline dataset in consideration
list_arpt = list(df0.ORIGIN_AIRPORT.unique())
#keeping common airports from list of airports available
avlbl_arpt = [value for value in list_arpt if value in list(airports['ORIGIN_AIRPORT']) ]

df0 = df0[df0['ORIGIN_AIRPORT'].isin(avlbl_arpt) ]

#merge airports information to main dataset
df0 = pd.merge(df0 , airports, how = 'left', on = 'ORIGIN_AIRPORT')

#creating new variable which represents the hour of the day in which the flight departure was scheduled.
df0['Departure_hour'] = df0['SCHEDULED_DEPARTURE'].apply(lambda x: (np.int((x-1)/100) +1 )*100 )

#****************merge weather data------------------------------------------------------------------------
# this  will extract weather information  from "https://mesonet.agron.iastate.edu" and will merge the weather data as per the time of flight with the main data
#list of features to be used from weather dataset
list_w = ['valid', 'station','tmpf', ' dwpf', ' relh', ' sknt', ' p01i', ' alti' , ' vsby',' skyc1'  ]


weather = pd.DataFrame()

for i in range(len(avlbl_arpt)):
    p1 = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station="
    p2 = "&data=all&year1=2015&month1=1&day1=1&year2=2015&month2=12&day2=31&tz=Etc%2FUTC&format=onlycomma&latlon=no&direct=no&report_type=1&report_type=2"
    url = p1+ str(avlbl_arpt[i]) + p2 
    
    data = urlopen(url).read()
    soup = BeautifulSoup(data)
    data2 = soup.get_text()
    
    print ('  ' + str(i) + '  '  + str(avlbl_arpt[i]))
    
    #saving weather data
    f = open( str(avlbl_arpt[i]) + ".csv","w")
    f.write( str(data2) )
    f.close()
    
    #reading weather data from the dataset saved above
    w = pd.read_csv( str(avlbl_arpt[i]) + ".csv")
    
    #replace missing values with nan
    w=w.replace('M' ,np.nan)
    w = w[list_w]
    w.rename(columns ={'station' :'ORIGIN_AIRPORT' } ,inplace = True)
    
    w['date'] = pd.to_datetime(w['valid'])
    w['MONTH'] = w['date'].dt.month
    w['DAY'] = w['date'].dt.day
    w['time'] = w['date'].dt.time
    w['Departure_hour'] = w['time'].apply(lambda x: (x.hour+1)*100 )
    w2 = w.drop_duplicates(['MONTH' , 'DAY','Departure_hour'])
    
    weather  = weather.append(w2)

#saving prepared weather data for the respective Airline
weather.to_csv('weather.csv')
weather = pd.read_csv('weather.csv')

#merge prepared weather data with the main data
df1 = pd.merge(df0,weather, how = 'left', on = ['ORIGIN_AIRPORT','MONTH', 'DAY','Departure_hour'], indicator = True)

df1.isnull().sum()
df2 = df1.dropna()

#converting hub information from string to numeric
df2['Hub2'] = np.where(df2.Hub_x =='S' ,1 , np.where(df2.Hub_x =='M' ,2 , np.where(df2.Hub_x =='L' ,3 , np.where(df2.Hub_x =='N' ,0 , np.where(df2.Hub_x =='None' ,0 , 0)))))

#creating binary Target variable with threshold of 10 minutes
df2['Delayed'] = np.where(df2.DEPARTURE_DELAY >=10 ,1 , 0)

df2["Delay"] = np.where(df2['DEPARTURE_DELAY'] <0 ,0 ,df2['DEPARTURE_DELAY'] )

pctl_99_2 = df2['Delay'][df2['Delay'] > 0].quantile(0.99)

#creating a new delay variable with capping of 200 minutes
df2['Delay2'] = np.where(df2['Delay'] >= 200, 200 , df2['Delay'])

#converting city variable from string to numeric
le = LabelEncoder()
le.fit(df2['CITY'].values)
df2['CITY2']=le.transform(df2['CITY'])

#converting origin airport values from string to numeric
le.fit(df2['ORIGIN_AIRPORT'].values)
df2['AIRPORT']=le.transform(df2['ORIGIN_AIRPORT'])

#defing function to compute statistics
def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean() , 'sum' : group.sum()}



stats_sky= df2['Delayed'].groupby(df2[' skyc1']).apply(get_stats).unstack()

sky_key_1 = {'VV ':4, 'SCT':3 ,'OVC':3 , 'BKN':2 , 'FEW':1 , 'CLR':0}
sky_key_2 = {'VV ':1, 'SCT':1 ,'OVC':1 , 'BKN':1 , 'FEW':1 , 'CLR':0}

#creating two features representing sky conditions and converting from string to numeric
df2['sky1'] = df2[' skyc1'].replace(sky_key_1)
df2['sky2'] = df2[' skyc1'].replace(sky_key_2)

#creating a second bindary target vriable as an alternate option to use
df2['Delayed2'] = np.where(df2['Delay'] >= 30, 1, 0)


#computing grouped statistics
stats_del = df2['DEPARTURE_DELAY'].groupby(df2['ORIGIN_AIRPORT']).apply(get_stats).unstack()
stats_delay = df2['Delay'].groupby(df2['ORIGIN_AIRPORT']).apply(get_stats).unstack()
stats_delayed = df2['Delayed'].groupby(df2['ORIGIN_AIRPORT']).apply(get_stats).unstack()
stats_Hub= df2['Delayed'].groupby(df2['Hub_x']).apply(get_stats).unstack()
stats_Month= df2['Delayed'].groupby(df2['MONTH']).apply(get_stats).unstack()
stats_dep_hour = df2['Delayed'].groupby(df2['Departure_hour']).apply(get_stats).unstack()

stats_City = df2['Delayed'].groupby(df2['CITY']).apply(get_stats).unstack()



#converting AIRPORT string values to numeric to be used as predictor variable
stats_delayed2 = df2['Delayed2'].groupby(df2['ORIGIN_AIRPORT']).apply(get_stats).unstack()
stats_delayed2.sort_values('mean', inplace = True)
stats_delayed2= stats_delayed2.reset_index()
AI_dict = dict(zip(stats_delayed2.ORIGIN_AIRPORT, stats_delayed2.index))

df2.loc[:,'AIRPORT2'] = df2.loc[:,'ORIGIN_AIRPORT'].replace(AI_dict)

#creating samples for training and validation
dev_mth = [1,2 ]#,3,4]
vld_mth = [5]#,6]

dev = df2[df2['MONTH'].apply(lambda x: x in dev_mth)]
vld = df2[df2['MONTH'].apply(lambda x: x in vld_mth)]

#saving final model data
df2.to_csv('Model_data.csv', encoding = 'utf-8')

#plot airport statistics on a map----------------------------------------------------------------

count_flights = df2['ORIGIN_AIRPORT'].value_counts()

stats_delayed2 = stats_delayed[stats_delayed['mean'] > 0]
list22 = list(stats_delayed2.index)

airport2 = airports[airports['ORIGIN_AIRPORT'].isin(list22)]

plt.figure(figsize=(11,11))

colors = ['yellow', 'red', 'lightblue', 'purple', 'green']
size_limits = [0, 0.1, 0.2, 0.3, 0.4]
labels = []
for i in range(len(size_limits)-1):
    labels.append("{} <.< {}".format(size_limits[i], size_limits[i+1])) 
    
map = Basemap(resolution='i',llcrnrlon=-180, urcrnrlon=-50,
              llcrnrlat=10, urcrnrlat=75, lat_0=0, lon_0=0,)
map.drawcoastlines()
map.drawcountries(linewidth = 3)
map.drawstates(color='0.3')

#_____________________
# put airports on map
for index, (code, y,x) in airport2[['ORIGIN_AIRPORT', 'LATITUDE', 'LONGITUDE']].iterrows():
    x, y = map(x, y)
    isize = [i for i, val in enumerate(size_limits) if val < stats_delayed2['mean'][code]]
    ind = isize[-1]
    map.plot(x, y, marker='o', markersize = ind+5, markeredgewidth = 1, color = colors[ind],
             markeredgecolor='k', label = labels[ind])
#_____________________________________________
# remove duplicate labels and set their order
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
key_order = ('0 <.< 0.1',  '0.1 <.< 0.2','0.2 <.< 0.3', '0.3 <.< 0.4')
new_label = OrderedDict()
for key in key_order:
    new_label[key] = by_label[key]
plt.legend(new_label.values(), new_label.keys(), loc = 1, prop= {'size':11},
           title='% Flights Delayed ', frameon = True, framealpha = 1)
plt.show()

# modeling  ---------------------------------------------------------------------------------

#list of features to use in the model
list_c1 = [ 'AIRPORT', 'Departure_hour','DAY', 'DAY_OF_WEEK' , 'AIR_TIME', 'DISTANCE','CY 15 Enplanements' 
              ,'tmpf', ' dwpf', ' relh',' sknt', ' alti', ' vsby', 'CITY2','sky1','Hub2' ]
# 'sky2' ,'FLIGHT_NUMBER'  ' p01i', ,'MONTH', , 'Hub2'

list_c=['AIRPORT2', 'Departure_hour','DAY', 'DAY_OF_WEEK' , ' vsby', 'CITY2','sky1' ,'DISTANCE','CY 15 Enplanements_y' , ' relh',' sknt','Hub2']

#specifying target variable to be used to train model
target = 'Delayed'

X_train = dev[list_c]
y_train = dev[target]

X_test = vld[list_c]
y_test = vld[target]


# using GradientBoostingClassifier to train the data for classification

gbm0 = GradientBoostingClassifier(learning_rate=0.01,n_estimators=1000, min_samples_split=40,
                      min_samples_leaf=10,max_depth =15,
                      max_features=0.8,subsample=0.5,random_state=10)

gbm0.fit(X_train, y_train)

print (gbm0.score(X_train, y_train))
print (gbm0.score(X_test,y_test))

predictions = gbm0.predict(X_test)
prob = gbm0.predict_proba(X_test)


temp = pd.DataFrame(prob)
temp['pred'] = np.where(temp[1] >0.3, 1, 0)
yt = y_test.reset_index()
pd.crosstab(yt['Delayed'], temp['pred'], rownames=['Actual'], colnames=['Predicted'], margins=True)                  

df_confusion = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'], margins=True)
df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    
print (df_confusion)

#creating plot to visualize feature importance
feat_imp = pd.Series(gbm0.feature_importances_, list(X_train.columns)).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.show()
