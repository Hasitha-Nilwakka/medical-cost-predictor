import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump

df = pd.read_csv('data/insurance.csv')

#target(y) what we want the model to predit
y = df["charges"]

#feature(X) the data we provide to the model for prediction
X = df.drop("charges", axis=1)

#split the feature(X) dataset into test and train
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#Building a Scikit-Learn Pipeline
#Explanation why we do this
#When we train the model it has multiple sets of data, so when the getdummies convert them computer readable data and drop firs removes redundant comparision
#for example - smoker status - without drop_first it would create two columns smoker_yes and smoker_no
#with drop_first it creates only one column smoker_yes = 0 or 1 which is the only column we need
#but the real problmen arise when the model is actually in use
#user inputs data but the model see smoker = yes / no and get dummies might not create smoker_yes or smoker_no column at all
#because the getdummies doesnt see enough data to identify the creation of the column
#same goes to the region, in the training model seems multiple values like nothwest, northeast and southwest
#but user selects only one so model doesnt see the need of comparision
#after building this pipeline the model remembers the possible categorical column values which solves the problem
#model expects x number of columns, with getdummies when the user input it might creates only 3 columns and gives error
#but when the model remembers the possible categorical columns OneHotEncoder still creates the exact number of columns expected by the model

##devide the numerical and categorical columns

#numeric columns
numeric_cols = ["age", "bmi", "children"]

#categorical columns - the problem we are addressing with creating pipeline
categorical_cols = ["sex", "smoker", "region"]

#setting up automatic data assembly line / preprocessing data
#ColumnTransformer act as a routing station, it routes specif columns to specific tools depending on the data they hold
#categorical and numeric are just lables 
#OneHitEncoder does the getdummies job but only for categorical values
#handle_unknown = ignore basically ignores any unknown value that has never seen in training
#for example if someone enters Paris, it would convert all to 0
#numbers are just passed through nothing needs to be done hence the passthrough

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ("numeric", "passthrough", numeric_cols)
])

#build the pipeline
#in contrast to before we are initating the random forest regressor in the pipeline
#why set the random state again, here the random state refers how the data is randomize across data trees
pipeline = Pipeline([
    ("preprocessor", preprocessor), 
    ("model", RandomForestRegressor(random_state=42, n_estimators=100))
])

#train the model
pipeline.fit(X_train, y_train)

#get the prediction using test data
y_pred = pipeline.predict(X_test)

mea = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MEA : {mea}")
print(f"R2 : {r2}")

#save the model
#Model serialization is the process of converting a trained machine learning model from a live, 
# in-memory Python object into a storable byte stream or file on disk
dump(pipeline, 'models/model.joblib')