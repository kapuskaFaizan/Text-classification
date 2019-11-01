import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
import pickle
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


#load data in to dataframe
df = pd.read_csv('hm_train.csv')
df['predicted_category'] =df['predicted_category'].map({'affection':1,'exercise':2, 'bonding':3, 'leisure':4 ,'achievement':5,'enjoy_the_moment':6, 'nature':7})

#data preprocessing
def prep(df):
	df.replace(r'\b\w{1,2}\b', '', regex =True, inplace = True)
	vectorizer = CountVectorizer()
	vectorizer.fit(df['cleaned_hm'])
	vec = vectorizer.transform(df['cleaned_hm'])
	pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))
	return vec
#train and save model
def train_model(vec):
	Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(train_vec,df['predicted_category'],test_size=0.1)
	print(Train_X.shape)
	SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
	SVM.fit(Train_X , Train_Y)
	predictions_SVM = SVM.predict(Test_X)
	filename = 'vec_svm.sav'
	pickle.dump(SVM, open(filename, 'wb'))
	return predictions_SVM, Test_Y

#create confusion matraix and accuracy_score
train_vec = prep(df)
predictions_SVM ,Test_Y = train_model(train_vec)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
print(classification_report(Test_Y,predictions_SVM))
print(f1_score(Test_Y,predictions_SVM, average='weighted'))