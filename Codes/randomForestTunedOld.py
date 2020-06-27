
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler, scale
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

np.set_printoptions(threshold=np.inf)
pd.set_option('max_colwidth', 800)

df = pd.read_csv('./baseline_data.csv',header=0)
print(df.head())

Catdata = df.loc[0:,'PTETHCAT':'APOE4'].join(df['PTGENDER']).join(df['imputed_genotype'])
Numdata = df.loc[0:,:'Thickness..thickinthehead..2035'].join(df['AGE']).join(df['PTEDUCAT']).join(df['MMSE'])
labels = df['DX.bl']

for i in Catdata.columns:
    the_value = str(Catdata[i].mode().values[0])
    Catdata[i].replace('NaN',the_value,inplace = True)
    dummy_data = pd.get_dummies(Catdata[i], prefix=i+"_", drop_first=True)
    Catdata = pd.concat([Catdata, dummy_data], axis=1)
    Catdata.drop(i, axis=1, inplace=True)

print (Catdata.head())
# for i in Numdata.columns:
#     the_value = str(Catdata[i].median().values[0])
#     Catdata[i].replace('NaN',the_value,inplace = True)

le = LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
labels = labels.astype('float')

data = Numdata.join(Catdata)
data.head()

X, X_test, y, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 1)
scaler = StandardScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X_test = scaler.transform(X_test)
X_train, X_CV, y_train, y_CV = train_test_split(X,y, test_size = 0.25, random_state = 1)


base_model = RandomForestClassifier()
base_model.fit(X_train, y_train)
predictions_cv = base_model.predict(X_CV)
print ("base model accuracy: ", accuracy_score(y_CV, predictions_cv))

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)

rf_random = RandomizedSearchCV (estimator = base_model, param_distributions = random_grid, n_iter = 100, cv =3, verbose = 2, random_state = 42, n_jobs = -1)
rf_random.fit(X_train, y_train)

print (rf_random.best_params_)

best_random = rf_random.best_estimator_
predictions_cv_random = best_random.predict(X_CV)
print ("tuned model accuracy: ", accuracy_score(y_CV, predictions_cv_random))

para_grid = {'bootstrap' : [False],
             'max_features' : ['sqrt'], 
             'max_depth': [10,20,30,40],
             'min_samples_leaf': [1,2,3],
             'min_samples_split' : [8,10,12],
             'n_estimators' : [250, 500, 800 ,1200]}


para_grid = {'min_samples_split': [12], 'n_estimators': [500], 'max_features': ['sqrt'], 'min_samples_leaf': [2], 'max_depth': [10], 'bootstrap': [False]}


rf_grid = GridSearchCV(estimator = base_model, param_grid = para_grid, cv =3, n_jobs = -1, verbose =2)
rf_grid.fit (X_train, y_train)

print (rf_grid.best_params_)
best_grid = rf_grid.best_estimator_
predictions_cv_grid = best_grid.predict(X_train)
print ("tuned model accuracy after grid: ", accuracy_score(y_train , predictions_cv_grid))

tuned_model = RandomForestClassifier(min_samples_split = 12, n_estimators = 250, max_features = None , min_samples_leaf= 5, max_depth= 10, bootstrap = True)
tuned_model.fit(X_train, y_train) 
predictions_tuned = tuned_model.predict(X_test)
print ("tuned model accuracy after tuning and fresh model: ", accuracy_score(y_test, predictions_tuned))