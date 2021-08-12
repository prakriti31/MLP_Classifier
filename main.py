import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


#Data preparation
path = 'heart_disease_data.csv'
dd = pd.read_csv(path)

#Data cleaning
print("Number of rows before cleaning: ", len(dd.index))

#Deleting NaN values (if any of the column value is NaN then whole row is deleted)
dd.dropna(how='any', inplace=True)

#Converting age column from days to years
dd['age'] //= 365

#Data normalization
dd.drop(dd[dd['weight'] > 170].index, inplace=True)
dd.drop(dd[dd['weight'] < 27].index, inplace=True)

dd.drop(dd[dd['height'] > 200].index, inplace=True)
dd.drop(dd[dd['height'] < 130].index, inplace=True)

dd.drop(dd[dd['age'] > 4500].index, inplace=True)
dd.drop(dd[dd['age'] < 35].index, inplace=True)

dd.drop(dd[dd['ap_hi'] > 220].index, inplace=True)
dd.drop(dd[dd['ap_hi'] < 50].index, inplace=True)

dd.drop(dd[dd['ap_lo'] > 120].index, inplace=True)
dd.drop(dd[dd['ap_lo'] < 50].index, inplace=True)

dd['ap_lo'] = dd['ap_lo'].astype(int)
dd['active'] = dd['active'].astype(int)

#Transforming objects to number values
dd['cholesterol'].replace(to_replace="normal", value=1, inplace=True)
dd['cholesterol'].replace(to_replace="above normal", value=2, inplace=True)
dd['cholesterol'].replace(to_replace="well above normal", value=3, inplace=True)

dd['glucose'].replace(to_replace="normal", value=1, inplace=True)
dd['glucose'].replace(to_replace="above normal", value=2, inplace=True)
dd['glucose'].replace(to_replace="well above normal", value=3, inplace=True)

print("Number of rows after cleaning: ", len(dd.index), "\n")

#Data preparation for binary classification
features = ['weight','ap_hi','ap_lo','smoke','alco','active','cholesterol','glucose']
X = dd[features]
y = dd.cardio

#Data preparation and cleaning for the BMI task
bmi_data = dd[:]

bmi_data['height'] = bmi_data['height'].astype(float)
bmi_data['height'] /= 100

bmi_data['bmi'] = bmi_data.apply(lambda row: (row.weight / (row.height ** 2 )), axis=1)
bmi_data.drop(bmi_data[bmi_data['bmi'] > 70].index, inplace=True)
bmi_data.drop(bmi_data[bmi_data['bmi'] < 16].index, inplace=True)

bmi_features = ['age','ap_hi','ap_lo','smoke','alco','active','cholesterol','glucose','cardio']
BX = bmi_data[bmi_features]
By = bmi_data.bmi

#print(bmi_data.active.unique())
bmi_unique = bmi_data[bmi_data.active != 2]
print()
#By this line all of the data cleaning and normalization should be complete
#After this we continue with configuring the models, splitting the data to train and test sets and also fitting it


#Binary regressor
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.30)

data_model = MLPClassifier()
data_model.fit(train_X,train_y)

expected_y = val_y
predicted_y = data_model.predict(val_X)

#BMI (MLP Regressor)
BX_train, BX_val, By_train, By_val = train_test_split(BX, By, test_size=0.30)

bmi_model = MLPRegressor()
bmi_model.fit(BX_train, By_train)

expected_By  = By_val
predicted_By = bmi_model.predict(BX_val)

#BMI (Linear Regression)
bmi_linear_model = LinearRegression()
bmi_linear_model.fit(BX_train, By_train)
predicted_By_linear = bmi_linear_model.predict(BX_val)

#Results

print("MLP Classification (Binary Regressor):")
print(metrics.classification_report(expected_y, predicted_y))
print("Confusion matrix: \n", metrics.confusion_matrix(expected_y, predicted_y))
print("")

print("MLP Regressor: ")
print("R2 = %.2f" % metrics.r2_score(expected_By, predicted_By))
print("MSE = %.2f" % metrics.mean_squared_error(expected_By, predicted_By))
print("")

print("Linear Regression: ")
print("MSE = %.2f" % metrics.mean_squared_error(By_val, predicted_By_linear))
print('R2 = %.2f' % metrics.r2_score(By_val, predicted_By_linear))


















