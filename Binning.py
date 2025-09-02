import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'Age': [25, 32, 30, 45, 35, 50, 40],
    'Income': [50000, 60000, 55000, 80000, 70000, 85000, 62000]
})

print("Original data:")
print(data)

data['Age_Binned'] = pd.cut(data['Age'], bins=3, labels=['Young', 'Middle-Aged', 'Old'])
print("\nAfter Binning Age column:")
print(data)

scaler = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
data['Income_Binned'] = scaler.fit_transform(data[['Income']])
data['Income_Binned'] = data['Income_Binned'].astype(int)
print("\nAfter Binning Income column:")
print(data)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
age_counts = data['Age_Binned'].value_counts()
age_counts.plot(kind='bar', color='skyblue')
plt.title('Age Binning')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.subplot(1, 2, 2)
income_counts = data['Income_Binned'].value_counts()
income_counts.plot(kind='bar', color='salmon')
plt.title('Income Binning')
plt.xlabel('Income Group')
plt.ylabel('Count')
plt.tight_layout()
plt.show()