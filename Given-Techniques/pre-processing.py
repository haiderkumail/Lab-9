import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression, Lasso

data = pd.DataFrame({
    'Age': [25, np.nan, 30, 45, 35, np.nan, 40],
    'Income': [50000, 60000, 55000, 80000, 70000, 85000, 62000],
    'Category': ['A', 'B', 'A', 'B', 'A', 'B', 'A'],
    'target': [1, 0, 1, 0, 1, 0, 1]
})

print("Original data:")
print(data)

data['Age'] = data['Age'].fillna(data['Age'].mean())
print("After handling missing values:")
print(data)

data.drop_duplicates(inplace=True)
print("\nAfter removing duplicates:")
print(data)

data = data[(np.abs(stats.zscore(data['Age'])) < 3)]
print("\nAfter outlier detection and removal:")
print(data)

scaler = MinMaxScaler()
data[['Age']] = scaler.fit_transform(data[['Age']])
print("\nAfter normalization of 'Age' column:")
print(data)

data[['Income']] = scaler.fit_transform(data[['Income']])
print("\nAfter normalization of 'Income' column:")
print(data)

data = pd.get_dummies(data, columns=['Category'])
print("\nAfter encoding categorical variable 'Category':")
print(data)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data[['Age', 'Income']])
print("\nAfter dimensionality reduction:")
print(reduced_data)

X = data.drop('target', axis=1)
y = data['target']
selector = SelectKBest(chi2, k=2)
X_selected = selector.fit_transform(X, y)
print("\nAfter feature selection using chi2:")
print(X_selected)

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=3)
X_selected_rfe = rfe.fit_transform(X, y)
print("\nAfter recursive feature elimination:")
print(X_selected_rfe)

model = Lasso(alpha=0.1)
model.fit(X, y)
selected_features = X.columns[model.coef_ != 0]
print("\nAfter feature selection via regularization:")
print(selected_features)
