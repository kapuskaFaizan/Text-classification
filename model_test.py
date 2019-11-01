import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
import pickle
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.model_selection import train_test_split


# load models
test_d = pd.read_csv('hm_test.csv')
model = pickle.load(open('vec_svm.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
vec =vectorizer.transform(test_d['cleaned_hm'])

#make predictions
pred = model.predict(vec)
test_d['predicted_category'] = pred
data_out=test_d.drop(columns = ['reflection_period','cleaned_hm','num_sentence','cleaned_hm'])
data_out['predicted_category'] = data_out['predicted_category'].map({1:'affection',2:'exercise', 3:'bonding',4: 'leisure' ,5:'achievement',6:'enjoy_the_moment',7 :'nature'})

#write to csv file
data_out.to_csv('submission.csv',index = False)






                                                                                 

