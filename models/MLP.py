import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import os
import pickle
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

data = pd.read_csv('../mfcc13_df.csv', index_col=False)  # if needed change to mfcc22_df.csv
labels = data.iloc[:, [-1]]

data = data.drop(labels.columns, axis=1)  # dropping labels column
data.describe()

# Check if data is balanced
print(labels.value_counts())

print(data.isnull().sum())
print(data)

print(data.isna().sum())
print(data.loc[0][:])

# Outliers
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
print(data.any() < (Q1 - 1.5 * IQR)) and (data.any() > (Q3 + 1.5 * IQR))

# from scipy import stats
# data=data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]

for column in data:
    plt.figure()
    data.boxplot([column])

# Standardization
data += 848.919070  # adding abs min value to whole data just to elimiate negative values
x = data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data = pd.DataFrame(x_scaled)
data

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=50)

print("Number of training samples:", X_train.shape[0])
print("Number of testing samples:", X_test.shape[0])
print("Number of features:", X_train.shape[1])

# Undersampling data or Oversampling - to choose

rus = RandomUnderSampler(random_state=0)
# rus = RandomOverSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
X_resampled.shape, y_resampled.shape

y_resampled.value_counts()

# PCA
pca = PCA()
X_train_new = pca.fit_transform(X_resampled)

explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(6, 4))
plt.bar(np.arange(0, len(explained_variance)), explained_variance, alpha=0.5, align='center')
plt.ylabel('Wariancja')
plt.xlabel('Główne składowe')
plt.show()

# PCA worsen the accuracy, uncomment to use it

# pca1 = PCA(n_components=10) 
# X_resampled = pca1.fit_transform(X_resampled)
# print(X_resampled.shape)

# pca1 = PCA(n_components=10) 
# X_test = pca1.fit_transform(X_test)
# print(X_test.shape)

# Grid Search - finding best combination of hyperparameters
# Uncoment to use grid search to find best hyperparameters - already found ->model_params
# model = MLPClassifier()

# parameter_space = {
#     'hidden_layer_sizes': [(200,),(300,)],
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [0.01, 0.1],
#     'learning_rate': ['constant','adaptive'],
#     'max_iter': [200, 500],
#     'batch_size': [100, 254]

# }

# clf = GridSearchCV(model, parameter_space, n_jobs=-1, cv=5)
# clf.fit(X_train, y_train)

# print('Best parameters found:\n', clf.best_params_)

# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

model_params = {
    'activation': 'relu',
    'alpha': 0.01,
    'batch_size': 254,
    'hidden_layer_sizes': (300,),
    'learning_rate': 'adaptive',
    'max_iter': 200,
    'solver': 'adam'
}

model = MLPClassifier(**model_params)
model.fit(X_resampled, y_resampled)

y_true, y_pred = y_test, model.predict(X_test)
accuracy = accuracy_score(y_true, y_pred)

print("Accuracy: {:.2f}%".format(accuracy * 100))

print(model.n_layers_)
print(model.n_iter_)
print(model.loss_)

print('Results on the test set:')
print(classification_report(y_true, y_pred))
confusion_matrix(y_true, y_pred)