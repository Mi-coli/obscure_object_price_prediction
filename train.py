
"""
The code for model training and data preparation cleaning code go here.
"""
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

def read_file(filename):
    '''
    This function reds in the file
    '''
    return pd.read_csv(filename)

def process_data(df):
    '''
    Extracts numerical values from object columns
    '''
    #Before I check for outliers or missing values, I have to address certain columns first
    #Some colums are in the json format and are also recorded in inconsistent units
    #I want to extract the values and the units, convert everything that's in mm to inches, and then remove
    #the columns that I no longer need

    #Create column for "units" and "value"
    df['surface_area_units'] = df['surface_area'].apply(lambda x: json.loads(x)['units'])
    df['surface_area_value'] = df['surface_area'].apply(lambda x: json.loads(x)['value'])
    df['bounding_box_volume_units'] = df['bounding_box_volume'].apply(lambda x: json.loads(x)['units'])
    df['bounding_box_volume_value'] = df['bounding_box_volume'].apply(lambda x: json.loads(x)['value'])
    df['max_x_length_units'] = df['max_x_length'].apply(lambda x: json.loads(x)['units'])
    df['max_x_length_value'] = df['max_x_length'].apply(lambda x: json.loads(x)['value'])
    df['volume_units'] = df['volume'].apply(lambda x: json.loads(x)['units'])
    df['volume_value'] = df['volume'].apply(lambda x: json.loads(x)['value'])
    df['max_y_length_units'] = df['max_y_length'].apply(lambda x: json.loads(x)['units'])
    df['max_y_length_value'] = df['max_y_length'].apply(lambda x: json.loads(x)['value'])
    df['max_z_length_units'] = df['max_z_length'].apply(lambda x: json.loads(x)['units'])
    df['max_z_length_value'] = df['max_z_length'].apply(lambda x: json.loads(x)['value'])
 
    #Convert everything that's in mm to inches
    df.loc[df["surface_area_units"] == "mm^2/", "surface_area_value"] *= 25.4
    df.loc[df["bounding_box_volume_units"] == "mm^3", "bounding_box_volume_value"] *= 25.4
    df.loc[df["max_x_length_units"] == "mm", "max_x_length_value"] *= 25.4
    df.loc[df["max_y_length_units"] == "mm", "max_y_length_value"] *= 25.4
    df.loc[df["max_z_length_units"] == "mm", "max_z_length_value"] *= 25.4
    df.loc[df["volume_units"] == "mm^3", "volume_value"] *= 25.4


    #And I also want to drop the original columns, since I won't need them anymore after creating new ones
    df = df.drop(["surface_area", 
                "bounding_box_volume", 
                "volume", 
                "max_x_length", 
                "max_y_length", 
                "max_z_length",
                'surface_area_units',
                'bounding_box_volume_units',
                'max_x_length_units',
                'volume_units',
                'max_y_length_units',
                'max_z_length_units'], axis = 1)

    #The "material" column is an object, so I have to convert it to numerical
    #Use LabelEncoder() to assign numerical values to each 
    encoder = LabelEncoder()
    df["material"] = encoder.fit_transform(df["material"]) 

    return(df)

def clean_data(df):
    '''
    Cleans data by addressing missing values and outliers
    '''
    #Some rows are almost completely empty. Will drop them
    df = df.dropna(subset = ["volume_value"])

    #Fill in NaN's using average (since mean and std are not too far off from each other)
    average = df["optimal_fit_on_hp_build_plate"].mean()
    df["optimal_fit_on_hp_build_plate"].fillna(value = average, inplace = True)

    #There are a few columns with extreme outliers
    maximum = df["per_unit_cost"].quantile(0.9)
    df = df[df["per_unit_cost"] < maximum]

    maximum2 = df["quantity"].quantile(0.9)
    df = df[df["quantity"] < maximum2]
    return(df)


def training(df):
    '''
    Training the decision tree regressor
    '''
    X = df.iloc[:, df.columns != "per_unit_cost"]

    y = df["per_unit_cost"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    model = DecisionTreeRegressor(random_state = 100)
    model.fit(X_train, y_train)
    return(model)

def load_and_clean_data():
    '''
    Combine function which loads the data and one that cleans it, 
    for simplicity
    '''
    df = read_file('best_ever_costs.csv')
    df = process_data(df)
    return(clean_data(df))