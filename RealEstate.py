import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


#GETTING DATASET AND CLEANING
dataset=pd.read_csv("D:\college\Real_Estate_Price_Predictor\Real-Estate-Price-Predictor\Real estate.csv")
dataset.drop(columns=["X5 latitude","X6 longitude","X1 transaction date"],inplace=True)
dataset.rename(columns={"X2 house age":"age","X3 distance to the nearest MRT station":"distance_nearest_mrt_station",
                        "X4 number of convenience stores":"convinience_stores",
                        "Y house price of unit area":"price_per_unit_area"},inplace=True)


#differentiating to dependent and independent attributes
x=dataset[["age","distance_nearest_mrt_station","convinience_stores"]]
y=dataset["price_per_unit_area"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)


#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
scx=StandardScaler()
x_train=scx.fit_transform(x_train)
x_test=scx.transform(x_test)#never fit for the testing data bcs it will create 
                            #separate statistical values for testing dataset.


#TRAINING AND PREDICTING THE TEST CASES
from sklearn import linear_model
lm=linear_model.LinearRegression()
lm.fit(x_train,y_train)
test_predict=lm.predict((x_test))


#PREDICTING CUSTOM INPUT
Age=eval(input("enter the age of the house (in yrs) : "))
nmrts=eval(input("Enter the distance ot nearest MRT station (in meters) : "))
stores=int(input("Enter the number of convinience stores near the house : "))
prediction=lm.predict(scx.transform(np.array([[Age,nmrts,stores]])))
print("The price for the land per square meters is : ",prediction)
print(" ")


#PLOTTING THE GRAPH OF THE FEATURES
sns.scatterplot(x=y_test,y=test_predict)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', linewidth=2)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual vs Predicted price")
plt.show()


#Getting the wieghts of the attributes
cols=x.columns
weight=lm.coef_

weightcols=pd.DataFrame({"Features":cols,"Weight":weight})
print("The weights of the columns are : ")
print(weightcols)
sns.barplot(x="Weight",y="Features",data=weightcols,palette="deep")
sns.set_style("darkgrid")
plt.title("Weight_of_Features")
plt.show()


#ERROR CALCULATION
error=np.where(y_test!=0,(abs(test_predict-y_test)/y_test)*100,np.nan)
print("The error percentage is : ",error)
print(" ")