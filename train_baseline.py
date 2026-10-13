import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Train the model using linear regression


#load the csv to DataFormat using pandas
df = pd.read_csv('data/insurance.csv')

#Turn the data to 0 and 1 so the computer can understand them
#get_dummies convert the data into computer readable data
#drop_first - removes one reduntant category which lowers memory usage
#without it data is converted as smoker_yes and smoker_no
#but we actually only need smoker_yes
df = pd.get_dummies(df, drop_first=True)

#target(y) 
#We want the model to predict charges, so we seperate the charges field
y = df["charges"]

#feature(x)
#We provide the following data to help the model predict
#Everything else other than the charges field
#axis 1 represents column(0 means row), remove the column named charges
X = df.drop("charges", axis=1)

#we split the dataset usking sklearn
#test_size what percentage of data to keep for testing, in here 20% data is kept for testing, rest go for training
#the purpose is to truly test if the model learned, otherwise the model could memorize
#random state, data is shuffled it should be always from the same point otherwise the model could always learn simple data or complicated data
#train_test_split provides four sets
#X_train and y_train are used for training X_test and y_test for test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#intiate the linear regression model
regr = LinearRegression()

#Train the model
regr.fit(X_train, y_train)

#Test the model, get the predictions using the Feture test data portion X_test
y_pred = regr.predict(X_test)

#check the prediction against the desired results

#MEA Mean absolute error -> check all the error and average it
#eg - if the model predicted 2000 but expected 1500 error was 500
#we calculate all the errors and take the mean
mea = mean_absolute_error(y_test, y_pred)

#r2 (r square) coefficient of determination
#How well the model learn
#The lazy guess - the initial guess could be off by a lot, the algorythem adjust weights to imporve this
#The smart guess - the model predicts informative figues as the weights are adjusted
#r2 represents how much the model has improved compared to the initiate state
#r2 = 1.00 (100%) is perfect state
r2 = r2_score(y_test, y_pred)

#Train the model using random forest regression

#Random forest regression is teqnique that combines multiple decision trees to predict a continuos numeric value
#Data. set is analysed in decision trees,
#Eg - decision tree one root = BMI , nodes = smoker or non smoker 
#Decisions are made in leafs and each tree outputs mathematical mean as the prediction
#we use the same splitting and same random state
#n_estimator defines how many decision trees are used
rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(X_train, y_train)
rfry_pred = rfr.predict(X_test)
rfrmea = mean_absolute_error(y_test, rfry_pred)
rfrr2 = r2_score(y_test, rfry_pred)



print(f"MEA Linear Regression: {mea}")
print(f"R2 Linear Regression: {r2}")

print(f"\nMEA Random forest: {rfrmea}")
print(f"R2 Random forest: {rfrr2}")


#MEA Linear Regression: 4181.1944737536505
#R2 Linear Regression: 0.7835929767120724

#MEA Random forest: 2544.6370639518086
#R2 Random forest: 0.8652657352502301