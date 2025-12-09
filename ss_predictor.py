#convert ukc data into useful format - pandas dataframe with only date and column with binary encoding
# convert weather data into useful format - pandas dataframe with mean of each metric per day
# investigate randomforestregressor with two data inputs, or merge tables 

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def logbook_reader(logbook_file):
    "reads logbook txt file and converts it to pandas dataframe with date and if route was climbed"

    event_date = []

    with open(logbook_file,'r', encoding='utf-8') as file:
        for line in file:
            columns = line.split(" \t")
            if len(columns) >= 2:
                date_str = columns[1].strip()

                try:
                    date_obj = datetime.strptime(date_str, "%d %b, %Y")
                    event_date.append(date_obj)
                except ValueError:
                    continue 
            
    event_df = pd.DataFrame({'date': pd.to_datetime(event_date)})
    event_df['route climbed'] = 1            

    return event_df

def weather_reader_1(weather_file, year):
    """reads weather file and converts it to pandas dataframe with relevent weather info. Also converts julian date to convential date
    works from 2013 to 2017 and 2019 to 2022"""

    weather_df = pd.read_csv(weather_file, sep=',', skiprows=2,
                            names=['julian date','time','mean wind','max wind','min wind','wind dir','std dev','t1','t2','stat1','stat2','TH'])
    
    weather_df = weather_df.drop(['time','stat1','stat2','TH','mean wind','max wind','min wind','wind dir','std dev'], axis=1)
    weather_df['t1'] = pd.to_numeric(weather_df['t1'], errors='coerce') #nuisance value somewhere on t1
    weather_df = weather_df.groupby('julian date').mean(numeric_only=True).reset_index() #average by day
    weather_df['date'] = pd.to_datetime(weather_df['julian date'].astype(int).astype(str).apply(lambda x: f"{year}{x.zfill(3)}"), format='%Y%j')
    weather_df = weather_df.drop(['julian date'], axis=1)
    
    return weather_df

def weather_reader_2(weather_file, year):
    """for 2018! reads weather file and converts it to pandas dataframe with relevent weather info. Also converts julian date to convential date"""

    weather_df = pd.read_csv(weather_file, sep=',', skiprows=2,
                            names=['int','julian date','time','mean wind','max wind','min wind','wind dir','std dev','t1','t2','stat1','stat2','TH','TH2'],
                            on_bad_lines='skip')
    
    weather_df = weather_df.drop(['time','stat1','stat2','TH','TH2','int','mean wind','max wind','min wind','wind dir','std dev'], axis=1)
    weather_df['t1'] = pd.to_numeric(weather_df['t1'], errors='coerce') #nuisance value somewhere on t1
    weather_df = weather_df.groupby('julian date').mean(numeric_only=True).reset_index() #average by day
    weather_df['date'] = pd.to_datetime(weather_df['julian date'].astype(int).astype(str).apply(lambda x: f"{year}{x.zfill(3)}"), format='%Y%j')
    weather_df = weather_df.drop(['julian date'], axis=1)
    
    return weather_df

#combine dataframes into one 

event_df = logbook_reader("ss_logbook_data.txt")

df2013 = weather_reader_1("2013.txt", 2013)
df2014 = weather_reader_1("2014.txt", 2014)
df2015 = weather_reader_1("2015.txt", 2015)
df2016 = weather_reader_1("2016.txt", 2016)
df2017 = weather_reader_1("2017.txt", 2017)
df2018 = weather_reader_2("2018.txt", 2018)
df2019 = weather_reader_1("2019.txt", 2019)
df2020 = weather_reader_1("2020.txt", 2020)
df2021 = weather_reader_1("2021.txt", 2021)
df2022 = weather_reader_1("2022.txt", 2022)

df = pd.concat([df2013,df2014,df2015,df2016,df2017,df2018,df2019,df2020,df2021,df2020], axis=0)
df['was_climbed'] = 0

df.loc[df['date'].isin(event_df['date']), 'was_climbed'] = 1

# one issue is that there are many days when the route is climbable but it is not climbed
# these will show up as false negatives to the model

# this removes any really crazy values for temperature
df = df[
    (df['t1'] >= -40) & (df['t1'] <= 40) &
    (df['t2'] >= -40) & (df['t2'] <= 40)
]

# to try and get around the false negative issue I have defined conditions in which the route is not climbable 
df['definitely_not_climbable'] = (
    (df['t1'] > 0) |
    (df['t2'] > 0)
)

training_df = df[
    (df['was_climbed']==1)|
    (df['definitely_not_climbable']==True)].copy()

training_df['climbable'] = training_df['was_climbed']

X = training_df[['t1']]
y = training_df['was_climbed']

# create and train model:
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X,y)

# take user input for conditions (now only temp...) and return prediction 
print("what is the temperature tomorrow?")
user_input = float(input(">"))

prediction = model.predict(pd.DataFrame([[user_input]], columns=['t1']))

if prediction[0] == 1:
    print("Savage slit is in condition")
elif prediction[0] == 0:
    print("Savage slit is not in condition")