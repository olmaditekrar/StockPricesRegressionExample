import pandas as pd
import quandl
import math
df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(-99999,inplace=True) #If the data isn't exist ! We are filling this data with a high value .
forecast_out = int(math.ceil(0.01*len(df))) # Forecast 1 percent of the data . This provide us to predict a price from 1 percent ealier prices of it .
df['Label'] = df[forecast_col].shift(-forecast_out) # We shifted the label coloumn (Adj. Close oloumn before we shift) into the 1 percent future(Minus shifted) ! And assign it.
#So now , we have a label that will present %1 of future price of the current row's Adj. Close feature ! 
df.dropna(inplace=True) # Droppen NaN's out of the list !




print(df.tail())
print(forecast_out)
print(df['Label'].tail())