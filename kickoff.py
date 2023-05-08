# -*- coding: utf-8 -*-

import pandas as pd
from math import cos, asin, sqrt, sin, atan2

import geopy.distance
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from collections import Counter
from prettytable import PrettyTable

import os
import math
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import geopy.distance
from tqdm import tqdm
from IPython.display import display

from statsmodels.formula import api
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10,6]



hubs = pd.read_csv('us_salesforce_hubs.csv')
hubs = hubs.drop(hubs.columns[[0]], axis=1)

emps = pd.read_excel('Fake Flight Prices from Zips.xlsx')
emps = emps.drop(emps.columns[[0, 1]], axis=1)




emps['lat_long'] = emps[['latitude', 'longitude']].apply(tuple, axis=1)

hubs['lat_long'] = hubs[['latitude', 'longitude']].apply(tuple, axis=1)


atlanta_coord = hubs.loc[hubs['salesforce_hub'] == "Atlanta", 'lat_long']
bellevue_coord = hubs.loc[hubs['salesforce_hub'] == "Bellevue", 'lat_long']
chicago_coord = hubs.loc[hubs['salesforce_hub'] == "Chicago", 'lat_long']
dallas_coord = hubs.loc[hubs['salesforce_hub'] == "Dallas", 'lat_long']
denver_coord = hubs.loc[hubs['salesforce_hub'] == "Denver", 'lat_long']
indianapolis_coord = hubs.loc[hubs['salesforce_hub'] == "Indianapolis", 'lat_long']
newyork_coord = hubs.loc[hubs['salesforce_hub'] == "New York", 'lat_long']
sanfrancisco_coord = hubs.loc[hubs['salesforce_hub'] == "San Francisco", 'lat_long']
seattle_coord = hubs.loc[hubs['salesforce_hub'] == "Seattle", 'lat_long']
theranch_coord = hubs.loc[hubs['salesforce_hub'] == "The Ranch", 'lat_long']


coords_list = [atlanta_coord, bellevue_coord, chicago_coord, dallas_coord, denver_coord, indianapolis_coord, newyork_coord, sanfrancisco_coord, seattle_coord, theranch_coord]

hub_cities = ["atlanta","bellevue","chicago","dallas","denver","indianapolis","new_york","san_francisco","seattle","the_ranch"]


ind = 0
for z in coords_list:
    dist_col = []
    for i in emps["lat_long"]:
        dist = geodesic(z, i).miles
        dist_col.append(dist) 
       # print(dist_col)
    emps[hub_cities[ind]] = dist_col
    ind = ind + 1


        

for i in emps["lat_long"]:
    print(i)
    

emps


# Calculating Distance to Atlanta


atl_dist = []
for i in emps["lat_long"]:
   dist = geodesic(atlanta_coord,i).miles
   atl_dist.append(dist);
   
emps["atlanta_distance"] = atl_dist 
print(emps)


test1= (42.37313998, -71.62163001)
test2= (60.4839, -151.152595) 

# Making sure its accurate

print(geodesic(atlanta_coord, test1).miles)
print(geodesic(atlanta_coord, test2).miles)

# Calculating Distance to Bellevue

bell_dist = []
for i in emps["lat_long"]:
   dist = geodesic(bellevue_coord,i).miles
   bell_dist.append(dist);
   
emps["bellevue_distance"] = bell_dist 

print(emps[["bellevue_distance","seattle_distance"]])


# Calculating Distance for Chicago

chicago_dist = []
for i in emps["lat_long"]:
   dist = geodesic(chicago_coord,i).miles
   chicago_dist.append(dist);
   
emps["chicago_distance"] = chicago_dist 
print(emps)

# Calculating Distance for Dallas

dallas_dist = []
for i in emps["lat_long"]:
   dist = geodesic(dallas_coord,i).miles
   dallas_dist.append(dist);
   
emps["dallas_distance"] = dallas_dist 
print(emps)


# Calculating Distance to Denver

denver_dist = []
for i in emps["lat_long"]:
   dist = geodesic(denver_coord,i).miles
   denver_dist.append(dist);
   
emps["denver_distance"] = denver_dist
print(emps)

# Calculating Distance to Indianapolis

ind_dist = []
for i in emps["lat_long"]:
   dist = geodesic(indianapolis_coord,i).miles
   ind_dist.append(dist);
   
emps["indianapolis_distance"] = ind_dist
print(emps)


# Calculating Distance to New York

nyc_dist = []
for i in emps["lat_long"]:
   dist = geodesic(newyork_coord,i).miles
   nyc_dist.append(dist);
   
emps["new_york_distance"] = nyc_dist
print(emps)

# Calculating Distance to San Francisco

sf_dist = []
for i in emps["lat_long"]:
   dist = geodesic(sanfrancisco_coord,i).miles
   sf_dist.append(dist);
   
emps["sf_distance"] = sf_dist
print(emps)


# Calculating Distance to Seattle

seattle_dist = []
for i in emps["lat_long"]:
   dist = geodesic(seattle_coord,i).miles
   seattle_dist.append(dist);
   
emps["seattle_distance"] = seattle_dist
print(emps)


# Calculating Distance to The Ranch

ranch_dist = []
for i in emps["lat_long"]:
   dist = geodesic(theranch_coord,i).miles
   ranch_dist.append(dist);
   
emps["the_ranch_distance"] = ranch_dist
print(emps)


# Calculating closest office

odfe = emps.iloc[:, -10:]

closest = odfe.idxmin(axis="columns")
print(closest)

emps["closest_office"] = closest


# Cleaning 


emps.closest_office.unique()

emps['closest_office'] = emps['closest_office'].str.split('_').str[0]
print(emps['closest_office'])


emps.loc[emps["closest_office"] == "new", 'closest_office'] = "new york"
emps.loc[emps["closest_office"] == "the", 'closest_office'] = "the ranch"
emps.loc[emps["closest_office"] == "san", 'closest_office'] = "san francisco"

emps.closest_office.unique()

emps.closest_office.unique().size


## Save to excel
  
emps.to_excel("Employee_distances.xlsx")



## Formula to get the office with lowest distance from all employees

def closest_office_to_all(df, columns):
    values = {}
    for i in columns:
        values[i] = df[i].sum()
    return min(values, key=values.get)

offices = emps.columns[6:15].tolist()

simdf1 = emps.sample(n=2)
closest_office_to_all(simdf1,offices)


## Showing that if we only use the office that has closest distance to all simulated employees
## Indianapolis will be the office that shows up the most



emps['closest_office'].value_counts()




value_counts = emps['closest_office'].value_counts()

plt.bar(value_counts.index, value_counts.values)
plt.xlabel('Closest Office')
plt.ylabel('Frequency')
plt.title('Frequency of Closest Office to Each Employee')
plt.xticks(rotation=90)

plt.show()


num_simulations = 1000
num_emps = 30
results = []

for i in range(1000):
    simulated_df = emps.sample(n=30)
    result = closest_office_to_all(simulated_df, offices)
    results.append(result)

print(results)


office_counts = Counter(results)
office_counts




#plt.bar(office_names, office_values)
#plt.xlabel("Office Names")
#plt.xticks(rotation=90)
#plt.ylabel("Frequency")
#plt.title("Frequency of each office in the simulations")
#plt.show()




emps.state.value_counts()
emps.state.unique().size



len(results)


def frequency_bar_chart(df, column):
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = [column, 'frequency']
    top_five = value_counts.head(5)
    fig, ax = plt.subplots()
    ax.bar(top_five[column], top_five['frequency'])
    ax.set_xlabel(column)
    ax.set_ylabel('Employees')
    ax.set_title('# of employees per state')
    plt.xticks()
    plt.tight_layout()
    plt.show()


frequency_bar_chart(emps, 'state')



#### Office with highest frequency as closest office.

def random_employees(df):
    
    zip_codes = np.random.choice(emps['postal_code'], size=10, replace=False)
    employee_counts = np.random.randint(1, 11, size=10)
    selected_zips = pd.DataFrame({'postal_code': zip_codes, 'employee_count': employee_counts})
    return selected_zips
    
    
def ideal_office(df):
    selected_zips = random_employees(df)
    selected_zips = pd.merge(selected_zips, emps[['postal_code', 'closest_office']], on='postal_code')
    office_counts = selected_zips.groupby('closest_office')['employee_count'].sum()
    top_offices = office_counts.nlargest(3)
    
    return top_offices

ideal_office(emps)
random_employees(emps)


# create a variable column name name it "Flight to " +hubname

## Function to show flying cost to hub including uber to airport and uber from airport to hub

def ubfly(df, hubs, hubname):
    
    selected_zips = random_employees(emps)
    uber_prices = df.loc[df['postal_code'].isin(selected_zips['postal_code']), 'Uber house to airport']
    selected_zips['uber_price'] = uber_prices.reset_index(drop=True)
    flight_prices = df.loc[df['postal_code'].isin(selected_zips['postal_code']), f'Flight to {hubname}']
    selected_zips['flight_price'] = flight_prices.reset_index(drop=True)
    airport_to_hub = hubs.loc[hubs['salesforce_hub'] == hubname, 'airport to hub'].values[0]
    selected_zips['uber to airport total'] = selected_zips['uber_price'] * selected_zips['employee_count']
    selected_zips['uber to hub total'] = airport_to_hub * selected_zips['employee_count']
    selected_zips['total flight price'] = selected_zips['flight_price'] * selected_zips['employee_count']
    selected_zips['total_travel_cost'] = selected_zips['uber to airport total'] + selected_zips['uber to hub total'] + selected_zips['total flight price']
    
    return selected_zips




    # get a random sample of zip codes and number of employees
    selected_zips = random_employees(df)
    
    # merge with the emps dataframe to get the closest airport for each zip code
    selected_zips = pd.merge(selected_zips, df[['postal_code', 'Closest Airport']], on='postal_code')
    
    # get the Uber price for each selected zip code by merging with the 'emps' dataframe
    selected_zips = pd.merge(selected_zips, df[['Closest Airport', 'Uber house to airport']], on='Closest Airport')
    
    # get the flight price for each selected zip code and hub
    flight_col = f'Flight to {hub_name}'
    selected_zips = pd.merge(selected_zips, df[['postal_code', flight_col]], on='postal_code')
    
    return selected_zi
test1 = ubfly(emps,hubs,"New York")
print(test1)





## Creating a function to calculate driving cost for randomly selected employees

drive = pd.read_excel('driving calculations.xlsx')
drive = drive.drop(drive.columns[[0]], axis=1)


def driveprice (df, hubname):
    
    selected_zips = random_employees(emps)
    drive_prices = df.loc[df['postal_code'].isin(selected_zips['postal_code']), f'Driving Cost to {hubname}']
    selected_zips['drive_price'] = drive_prices.reset_index(drop=True)
    selected_zips['driving price total'] = selected_zips['drive_price'] * selected_zips['employee_count']
    
    return selected_zips


driveprice(drive, "Atlanta")
    
    






