import pandas as pd
import numpy as np
import seaborn as sns
import re
import nltk
from matplotlib import pyplot as plt
from scipy.sparse.construct import random
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.svm import SVR,SVC
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import shapiro

def buildLinearModel(X,y):
  linear_model = LinearRegression()
  linear_model.fit(X,y)
  return linear_model

def buildLogisticRegressionModel(X,y):
  logistic_model = LogisticRegression()
  logistic_model.fit(X,y)
  return logistic_model

def buildKNNModel_Reg(X,y,k):
  knn_model = KNeighborsRegressor(n_neighbors=k)
  knn_model.fit(X,y)
  return knn_model

def buildKNNModel_Cls(X,y,k):
  knn_model = KNeighborsRegressor(n_neighbors=k)
  knn_model.fit(X,y)
  return knn_model

  

def see_Reg_Model_Performance(model,model_name,features,y_test,prediction):
  print("For {}".format(model_name))
  print("Mean Absolute Error: ", metrics.mean_absolute_error(y_test,prediction))
  print("Mean Squared Error: ", metrics.mean_squared_error(y_test,prediction))
  model_coef = pd.DataFrame(model.coef_,index=features,columns=["Coefficients"])
  print(model_coef)
  plot_prediction_ytest_plot(prediction,y_test,model_name)
  print()



def plot_prediction_ytest_plot(prediction,y_test,model_name):
  plt.figure(figsize=(12,6))
  plt.legend(model_name)
  plt.title("Plot for {}".format(model_name))
  plt.xlabel("Predicted Values")
  plt.ylabel("Actual Values")
  sns.scatterplot(prediction,y_test)
  plt.show()



def see_Log_Model_Performance(model,model_name,y_test,prediction):
  print("For {}".format(model_name))
  print("Accuracy Score: ",accuracy_score(y_test,prediction))
  print("Confusion Metrix: ",confusion_matrix(y_test,prediction))
  #plot_prediction_ytest_plot(prediction,y_test,model_name)
  print()



def clean_data_Fake_News_Predictor(news_dataset):
  news_dataset = news_dataset.fillna(" ")
  news_dataset_cleaned = news_dataset.drop(['id','text'],axis=1)
  news_dataset_cleaned['Author-Title'] = news_dataset_cleaned['author'] + " " + news_dataset_cleaned['title'] 
  news_dataset_cleaned.drop(['author','title'],axis=1,inplace=True)
  
  news_dataset_cleaned['Author-Title-Stemmed'] = news_dataset_cleaned['Author-Title'].apply(stem_data)

  X = news_dataset_cleaned['Author-Title-Stemmed'].values
 
  tfidf_vectorizer = TfidfVectorizer()
  X = tfidf_vectorizer.fit_transform(X)

  return X



def stem_data(text):
  #Here we are filtering the special characters from the text.
  cleaned_text = re.sub("[^a-zA-Z]", " ",text)
  cleaned_text = cleaned_text.lower()
  words_list = cleaned_text.split()
  #Here we are making a new list of stemmed words. These stemmed words are the stemmed version of words which are in words_list but not in stop_words.
  stem_porter = PorterStemmer()
  nltk.download("stopwords")
  stop_words = stopwords.words('english')
  stemmed_words = [stem_porter.stem(word) for word in words_list if word not in stop_words]
  return " ".join(stemmed_words)

def plot_features(dataset, feature):
  plt.figure(figsize=(10,5))
  plt.legend(feature)
  sns.distplot(dataset[feature])
  plt.title("For feature {}".format(feature))
  plt.show()

def find_outliers(dataset,feature):
  outliers = []
  mean = np.mean(dataset[feature])
  std = np.std(dataset[feature])
  q1, q3 = np.percentile(dataset,[25,75])
  IQR = q3- q1
  lower_bound = q1 - 1.5*IQR
  upper_bound = q3 + 1.5*IQR

  #check if data is gaussian or not?
  stats, p = shapiro(dataset[feature])

  for data_value in dataset[feature]:
    point = (data_value-mean)/std

    if p>.05:
      if point > 3*std : 
        outliers.append(data_value)
    else:
      if lower_bound > point > upper_bound :
        outliers.append(data_value)

  print("For the feature {}, there are {} outliers".format(feature,len(outliers)))
  #print("For the feature {}".format(feature))
  #print("Mean: {}".format(mean))
  #print("STD : {}".format(std))
  #print("Number of outliers: {}".format(len(outliers)))
  #print("----------")

def plot_cat_features(dataset,feature):
  plt.figure(figsize=(10,5))
  sns.catplot(x=feature,y='Loan_Status',data=dataset,kind='point',aspect=2)
  plt.title("Categorical Plot for {}".format(feature))
  plt.xlabel(feature)
  plt.ylabel('Loan_Status')
  plt.show()







