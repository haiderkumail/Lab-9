import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = {'Value': [1.2, 4.6, 10.8, 50.3, 100.9, 250.1, 500.5, 1000.7]}
df = pd.DataFrame(data)

train, validation = train_test_split(df, test_size=0.2, random_state=42)

train_augmented = train.copy()
train_augmented['Augmented_Value'] = train['Value'] + np.random.normal(0, 0.5, size=len(train))

train_recovered = train_augmented.drop(columns=['Augmented_Value'])

print("Training Set:\n", train)
print("\nValidation Set:\n", validation)
print("\nAugmented Training Set:\n", train_augmented)
print("\nRecovered Original Training Set:\n", train_recovered)

plt.figure(figsize=(10, 6))
plt.plot(train['Value'].values, label='Original Training Data', marker='o')
plt.plot(train_augmented['Augmented_Value'].values, label='Augmented Training Data', marker='o')
plt.plot(validation['Value'].values, label='Validation Data', marker='o', linestyle='--', color='red')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Original, Augmented, and Validation Data')
plt.legend()
plt.grid(True)
plt.show()