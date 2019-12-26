from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import GridSearchCV

# - Predicting if the cancer diagnosis is benign or malignant based on several observations/features
# - 30 features are used, examples:
#         - radius (mean of distances from center to points on the perimeter)
#         - texture (standard deviation of gray-scale values)
#         - perimeter
#         - area
#         - smoothness (local variation in radius lengths)
#         - compactness (perimeter^2 / area - 1.0)
#         - concavity (severity of concave portions of the contour)
#         - concave points (number of concave portions of the contour)
#         - symmetry
#         - fractal dimension ("coastline approximation" - 1)
#
# - Datasets are linearly separable using all 30 input features
# - Number of Instances: 569
# - Class Distribution: 212 Malignant, 357 Benign
# - Target class:
#          - Malignant
#          - Benign
from sklearn.svm import SVC

cancer = load_breast_cancer()

# print(cancer.keys())

df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(cancer['feature_names'], ['target']))

# printing the Top 5 rows
print(df.head())

# Visualizing the data relation using the pairplot and decide which algorithm can solve the problem
sns.pairplot(df, hue='target', vars=['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])

plt.show()

# Chceking whether datas are balanced using count plot
sns.countplot(df['target'], label="Count")
plt.show()

sns.scatterplot(x='mean area', y='mean smoothness', hue='target', data=df)

plt.show()

# Let's check the correlation between the variables
# Strong correlation between the mean radius and mean perimeter, mean area and mean primeter
plt.figure(figsize=(20, 10))
sns.heatmap(df.corr(), annot=True)

plt.show()

X = df.drop(['target'], axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

svc_model = SVC()
svc_model.fit(X_train, y_train)

y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)

# Lets see confusion matrix in plot
sns.heatmap(cm, annot=True)

plt.show()

# Scaling the data.
# Tried MinMaxScaler, StandardScaler both all scaler given same result
scaled_df = RobustScaler().fit_transform(df)

X = df.drop(['target'], axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

# Parameter for grid search
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=4)
grid.fit(X_train, y_train)

# Checking the best prams and best estimater to implement the algorithm
print(grid.best_params_)
print(grid.best_estimator_)

y_predict = grid.predict(X_test)
cm = confusion_matrix(y_test, y_predict)

# Lets see confusion metrix in plot after Confusion matrix
sns.heatmap(cm, annot=True)

plt.show()

print(classification_report(y_test, y_predict))
