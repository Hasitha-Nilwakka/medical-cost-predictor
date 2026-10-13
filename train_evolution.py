#We are training another model using different learning alogorithem
#Previous model was trained using linier regression and random forest and the r2 was lifted to 86
#This evolution tests if the new model perform better with GradientBoosingRegression
#Gradient boosting refers a learning technique how a model learn from its previous mistakes
#In random forest the random samples created number of data trees in our case we created 100
#Then we took the mean of the leaf decisions
#But Gradient Boosting creates data trees one by one
#Example first tree guess the cost for a given scenario, calculates the residual residual = actual price - predicted price
#Second tree tries to correct the previous trees errors, this process goes on and subsiquent trees tries to correct the previous trees mistake

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump

#Load data
df = pd.read_csv('data/insurance.csv')

#We also combine feature engineering
#Feature engineering is using raw data and domain knowledge to create, modify or select input variables
#That help machine learning more accurate predictions
#Here we use the domain knowledge which is in the insurance field, a smoker is always given additional load
#We also assume that higher the BMI higher the obeisity which increases health risks
#So we create a hypothesis that patients with obeisity and smokers would have higher charges compared to non smoking non-obeise people
#We do not need to one hot encode this new variable becase we can convert the conditional boolean in to numerical values in our calculation
df["is_obese_smoker"] = (
    (df["bmi"] >= 30) & (df["smoker"].str.lower() == 'yes')
).astype(int)

#target(y)
y = df["charges"]

#below we create two sets of numerical and categorical variable
#base and engineered with the purpose of analysing
#if the model predicts with feature engineering after training with gradient boosting

#Original features(X_base)
base_numeric_cols = ["age", "bmi", "children"]

#categorical cols remains the same for both - these will be one hot encoded
categorical_cols = ["sex", "smoker", "region"]

base_feature_cols = base_numeric_cols + categorical_cols
X_base = df[base_feature_cols]

#Engineered numerical cols, this includes the new variable is_obese_smoker
engineered_numeric_cols = ["age", "bmi", "children", "is_obese_smoker"]

engineered_feature_cols = engineered_numeric_cols + categorical_cols
X_engineered = df[engineered_feature_cols]

#create the train split
#Now we have seperate train and tests splits to train two seperate models
#base_train will train the model with only base dataset
#eng_train will add the featured column and train the model
#test size is still 20%
X_base_train, X_base_test, X_eng_train, X_eng_test, y_train, y_test = train_test_split(X_base, X_engineered, y, test_size=0.2, random_state=42)

#following function returns the preprocessing pipeline
#function takes the parameter numeric cols, thats the difference between base and engineered
def create_preprocessing_pipeline(numeric_cols):
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_cols),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols
            )
        ]
    )

#create preprocessors for both datasets
base_preprocessor = create_preprocessing_pipeline(base_numeric_cols)
engineered_preprocessor = create_preprocessing_pipeline(engineered_numeric_cols)

#define three models
#1. random forest 
#2. gradient boosting
#3. gradient boosting + feature engineering

random_forest_pipeLine = Pipeline([
    ("preprocessor", base_preprocessor),
    ("model", RandomForestRegressor(random_state=42, n_estimators=100))
])

gradient_boost_pipeline = Pipeline([
    ("preprocessor", base_preprocessor),
    ("model", GradientBoostingRegressor(random_state=42))
])

gradient_with_feature_pipeline = Pipeline([
    ("preprocessor", engineered_preprocessor),
    ("model", GradientBoostingRegressor(random_state=42))
])

#train the random forest model
random_forest_pipeLine.fit(X_base_train, y_train)

#train the gradient boosting model
gradient_boost_pipeline.fit(X_base_train, y_train)

#train the gradient boost + feature model
gradient_with_feature_pipeline.fit(X_eng_train, y_train)

#get predictions
y_random_forest_pred = random_forest_pipeLine.predict(X_base_test)
y_gradient_boost_pred = gradient_boost_pipeline.predict(X_base_test)
y_gradient_feature_pred = gradient_with_feature_pipeline.predict(X_eng_test)

#get mean 
rf_mea = mean_absolute_error(y_test, y_random_forest_pred)
gb_mea = mean_absolute_error(y_test, y_gradient_boost_pred)
gbf_mea = mean_absolute_error(y_test, y_gradient_feature_pred)

#get r2
rf_r2 = r2_score(y_test, y_random_forest_pred)
gb_r2 = r2_score(y_test, y_gradient_boost_pred)
gbf_r2 = r2_score(y_test, y_gradient_feature_pred)

print("MEA and R2 using model trained with Random forest")
print(f"MEA : {rf_mea}")
print(f"R2 : {rf_r2}")
print("-" * 10)
print("MEA and R2 using model trained with Gradient boosting")
print(f"MEA : {gb_mea}")
print(f"R2 : {gb_r2}")
print("-" * 10)
print("MEA and R2 using model trained with Gradient boosting + feature engineering")
print(f"MEA : {gbf_mea}")
print(f"R2 : {gbf_r2}")

#MEA and R2 using model trained with Random forest
#MEA : 2544.6370639518086
#R2 : 0.8652657352502301
#----------
#MEA and R2 using model trained with Gradient boosting
#MEA : 2446.202801885396
#R2 : 0.8791672491881917
#----------
#MEA and R2 using model trained with Gradient boosting + feature engineering
#MEA : 2486.914566039453
#R2 : 0.8752292127298418

#from the results we observed that gradient boosting without the feature engineering showed the better results
#feature engineering always doesnt enhance the model prediction capacity
#domeain knowledge can propose features but the evaluation shows if it actually works
#in our case the feature might be redundant, data trees might already have a similar comparision

#Hyperparameter tuning with GridSearchCV is basically finding better settings for the algorythem
#The GradientBoostingRegressor uses default settings such as
#n_estimators = 100, learning_rate = 0.1, max_depth = 3 these are the hyperparameters
#These hyperparameters control HOW the moedl learns
#n_estimators -> no of trees - how many sequential corrections steps the model take default is 100 more trees imporve prediction but too many can overfit
#learning_rate -> size of each correction - how much should each new tree contribute, smaller steps often require more trees
#max_depth -> complexity of each tree. how many levels of decisions can each tree make, deeper trees can learn more complex patters, but also may learn noise
#min_samples_split -> minimum samples needed to split a node. this controlls when a tree is allowed to devide a group of samples into smaller groups
#in GridSerchCV what happens under the hood is testing every combination of chosen settings, without the need to do it manually

#step 1. create a parameter grid
#this grid containes different hyperparameter settings that we want the model to train with
#the double underscore after model tells that this settings should be applied to a step called "model" in the pipeline
# This grid will allow 54 combinations 3 x 3 x 3 x 2
param_grid = {
    "model__n_estimators" : [100, 150, 200], 
    "model__learning_rate" : [0.03, 0.05, 0.1],
    "model__max_depth" : [2,3,4],
    "model__min_samples_split" : [2,5]
}

#create the seperate pipeline for gridsearchcv
gridcvsearchcv_pipeline = Pipeline([
    ("preprocessor", base_preprocessor),
    ("model", GradientBoostingRegressor(random_state=42))
])

#apply the gridsearch
#cv=5 cross validation runs five times
#This feature devides the train dataset into five different folds and the each combination is tested 5 times
#Example we have 1000 rows of data, its devided to five folds (0-200 -> fold 1, 200 -> 400 fold 2 and so on)
#these folds are tested in five rounds
#Round one -> test fold 1,2,3,4 and test with fold 4
#Round two -> test 2,3,4,5 and test with fold 5 and so on , this ensures cross validation
#neg_means_absolute_error -> messures how far the model guesses are from the actual answers but uses negative numbers so the higher number is always better
#n_jobs=1 use one computer processor core, -1 mens use all available cores
#refit=True -> retrain the winning model on all the training data. 
##Once the gridsearchcv finds the best setting combination from the param_grid run the setting on the full training dataset again
grid_search = GridSearchCV(
    estimator=gridcvsearchcv_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=1,
    refit=True
)

#search best hyperparameter 
grid_search.fit(X_base_train, y_train)

#get the bestmodel from the grid search
best_model = grid_search.best_estimator_

print("-" * 10)
print("Best hyperparaeters selected by the gridsearch")
print(grid_search.best_params_)
print("\nBest cross validation mea found from the grid search")
print(grid_search.best_score_)
print("-" * 10)

#predit the base test data with the best model choosen by grid search cv
gridsearchcv_predict = best_model.predict(X_base_test)

#get MEA and r2
grid_mea = mean_absolute_error(y_test, gridsearchcv_predict)
grid_r2 = r2_score(y_test, gridsearchcv_predict)

print("-" * 10)
print(f"GridSerachCV mea : {grid_mea}")
print(f"GridSerachCV r2 : {grid_r2}")
print("-" * 10)

#Best hyperparaeters selected by the gridsearch
#{'model__learning_rate': 0.1, 'model__max_depth': 2, 'model__min_samples_split': 5, 'model__n_estimators': 150}
#Best cross validation mea found from the grid search
#-2600.3657703134313
#----------
#----------
#GridSerachCV mea : 2392.8879397809847
#GridSerachCV r2 : 0.8803568035957766

#save all four models
dump(random_forest_pipeLine, 'models/model_rf.joblib')
dump(gradient_boost_pipeline, 'models/model_gb.joblib')
dump(gradient_with_feature_pipeline, 'models/model_feat_eng.joblib')
dump(best_model, 'models/model_tuned.joblib')