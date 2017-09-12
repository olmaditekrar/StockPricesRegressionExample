import pandas as pd
import quandl ,math ,datetime
import numpy as np
from sklearn import preprocessing , cross_validation , svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import time
import pickle


style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(-99999,inplace=True) #If the data isn't exist ! We are filling this data with a high value .
forecast_out = int(math.ceil(0.01*len(df))) # Forecast 1 percent of the data . This provide us to predict a price from 1 percent ealier prices of it .
df['Label'] = df[forecast_col].shift(-forecast_out) # We shifted the label coloumn (Adj. Close column before we shift) into the 1 percent future(Minus shifted) ! And assign it.
#So now , we have a label that will present %1 of future price of the current row's Adj. Close feature ! 



X = np.array(df.drop(['Label'],1)) # x is standby our features !
X = preprocessing.scale(X) # Scale the whole x because it will be more efficent .
X_lately = X[-forecast_out:] #We are actually predicting this array because , we use %1 (33) of data .
X = X[:-forecast_out]


df.dropna(inplace=True) # Droppen NaN's out of the list !

y = np.array(df['Label']) # y is standby our labels !
  
#Now we are going to get test and train datas from X and y . 
X_train , X_test , y_train , y_test = cross_validation.train_test_split(X,y,test_size = 0.2) # It will shuffle all the data inside X and y and it will return 4 outputs : X_train , y_train , X_test , y_test !


#TRAINING THE CLASSIFIER 

	# clf = LinearRegression(n_jobs = 10) #Classifier is the Linear Regression and its so much better than Support Vector Machines . And we are threading 10 jobs at a time.
	# clf.fit(X_train,y_train) # We fit the training data. 

	# with open('linearregression.pickle','wb') as f: #Create a file and use f as a variable.
	# 	pickle.dump(clf,f) #Dumb the trained classifier into that file.

#END OF TRAINING THE CLASSIFIER

#We don't need to train the classifier everytime we want to run the program , so we saved the trained classifier's data into the file and we simply read it .
pickle_in = open ('linearregression.pickle','rb') # Open the file and assign it to the pickle_in.
clf = pickle.load(pickle_in) # Load it into the classifier.
 

accuracy = clf.score(X_test,y_test) # We test the data and get the accuracy score . 
# The reason why we want different data to train and test is : We need to send different data for testing because it will be more learnable with the data it didn't see before.


forecast_set = clf.predict(X_lately) #We are predicting the prices based on X_lately as we said before.

print(forecast_set,accuracy,forecast_out)


df['Forecast'] = np.nan



last_date = df.iloc[-1].name # Get the last date !
last_unix = time.mktime(last_date.timetuple())# Time to iterate
one_day = 86400 #1 day in seconds.
next_unix = last_unix + one_day #Next day in seconds!
for i in forecast_set: # For every predicted value .
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix+=one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i] #Get or create the index of date and add the value of predicted forecast !

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()







