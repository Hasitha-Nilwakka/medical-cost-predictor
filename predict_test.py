#testing the model
import joblib
import pandas as pd

#load the model
pipeline = joblib.load('models/model.joblib')

test_data = {
    "age": 30,
    "sex": 'female',
    "bmi": 32.4,
    "children": 1,
    "smoker": 'no',
    "region": 'southwest'
}

df = pd.DataFrame([test_data])

#predict
cost = pipeline.predict(df)

print(cost)