#Exploratory Data Analysis (EDA)
import pandas as pd

#Load the csv
df = pd.read_csv('data/insurance.csv')

#Get no of rows and columns
rows, columns = df.shape

#get the total nulls in dataframe across columns
total_nulls = df.isna().sum()

#charges min, max and median
charges_min = df['charges'].min()
charges_max = df['charges'].max()
charges_mean = df['charges'].mean()
charges_median = df['charges'].median()

#age min, max and median
age_min = df['age'].min()
age_max = df['age'].max()
age_mean = df['age'].mean()
age_median = df['age'].median()

#bmi min, max and median
bmi_min = df['bmi'].min()
bmi_max = df['bmi'].max()
bmi_mean = df['bmi'].mean()
bmi_median = df['bmi'].median()

#Average charge for smokers and non smokers
average_charge_by_smokers = df.groupby('smoker')["charges"].mean()
smokers_average = average_charge_by_smokers.loc['yes']
non_smokers_average = average_charge_by_smokers.loc['no']

print(f"\nNumber of rows: {rows}")
print(f"\nNumber of columns: {columns}")
print(f"\nData types of each column: ")
print(df.dtypes) #extract data type
print(f"\nTotal Null columns : {total_nulls}")

print(f"\nCharges min value : {charges_min}")
print(f"Charges max value : {charges_max}")
print(f"Charges mean value : {charges_mean}")
print(f"Charges median value : {charges_median}")

print(f"\nAge min value : {age_min}")
print(f"Age max value : {age_max}")
print(f"Age mean value : {age_mean}")
print(f"Age median value : {age_median}")

print(f"\nBMI min value : {bmi_min}")
print(f"BMI max value : {bmi_max}")
print(f"BMI mean value : {bmi_mean}")
print(f"BMI median value : {bmi_median}")

print(f"\nSmokers charges average : {smokers_average}")
print(f"Non-smokers charges average : {non_smokers_average}")
print(f"Difference : {smokers_average - non_smokers_average}")