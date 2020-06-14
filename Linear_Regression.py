import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#data set loading 
dataset = pd.read_csv("Price_list.csv")
print(dataset)

#plot a scatter plot
plt.scatter(dataset.area,dataset.price , color ="red" , marker ="+")

#Labeling
plt.xlabel("area")
plt.ylabel("price")

#fitting 
reg = LinearRegression()
reg.fit(dataset[["area"]],dataset.price)

#plotting the regression line 
plt.plot(dataset.area ,reg.predict(dataset[["area"]]),color = "blue")

#prediction
Key = reg.predict([[3300]])
print(Key)
 
# Coeffiecient (m) and intercept(b) values in Y= mx+b
print(reg.coef_)#m
print(reg.intercept_)#b

#checking the predicted value
print(135.78767123*3300+180616.43835616432)#mx+b

#loading the prediciton Csv file
d = pd.read_csv("prediction.csv")
print(d)


#predicting the values in prediction csv
s = reg.predict(d)
print(s)

#new column
d['prices'] = s

#exporting the predicted value
d.to_csv("prediction.csv",index = False)
