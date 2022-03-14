import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def get_prediction(data1, data2, model):
  """
  Predict the class of a given data point.
  """
  mapped_cols = ['Driving_experience', 'Sex_of_driver', 'Age_band_of_driver', 'Educational_level', 'Time', 'Accident_severity']
  colsToMap = ['Day_of_week', 'Owner_of_vehicle', 'Area_accident_occured',
       'Lanes_or_Medians', 'Road_allignment', 'Types_of_Junction',
       'Road_surface_type', 'Road_surface_conditions', 'Light_conditions',
       'Weather_conditions', 'Type_of_collision',
       'Number_of_vehicles_involved', 'Number_of_casualties',
       'Vehicle_movement', 'Pedestrian_movement', 'Cause_of_accident']
  df1 = pd.DataFrame(data1, columns=mapped_cols)
  df2 = pd.DataFrame(data2, columns=colsToMap)
  df2 = pd.get_dummies(df2)
  final_df = pd.concat([df1, df2], axis=1)

  prediction = model.predict(final_df)
  return prediction


def encodeTime(time):
  if time >=6 and time<18: 
    return 0
  else:
    return 1

def encodeSex(value):
  if value == 'Female':
    return 0
  else:
    return 1
  
def encodeAgeBand(value):
  if value == 'Under 18':
    return 0
  elif value == '18-30':
    return 1
  elif value == '31-50':
    return 2
  elif value == 'Over 51':
    return 3

def encodeDrivingExp(value):
  mapper = {'No Licence': 0, 'Below 1yr': 1, '1-2yr': 2, '2-5yr': 3, '5-10yr': 4, 'Above 10yr': 5}
  return mapper[value]

def encodeEdu(value):
  mapper = {'Illiterate': 0, 'Writing & reading': 1, 'Elementary school': 2, 'Junior high school': 3, 'High school': 4, 'Above high school': 5}
  return mapper[value]






