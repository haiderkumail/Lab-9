import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Value1': [1.2, 4.6, 10.8, 50.3, 100.9, 250.1, 500.5, 1000.7],
    'Value2': [2.3, 5.7, 11.9, 51.4, 101.8, 251.2, 501.6, 1001.8]
}
df = pd.DataFrame(data)

shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

augmented_df = shuffled_df.copy()
augmented_df['Value1'] = augmented_df['Value1'] + np.random.normal(0, 0.5, len(augmented_df))
augmented_df['Value2'] = augmented_df['Value2'] + np.random.normal(0, 0.5, len(augmented_df))

print("Shuffled and Augmented Data:\n", augmented_df)

plt.figure(figsize=(10, 6))
plt.scatter(df['Value1'], df['Value2'], color='blue', label='Original Data')
plt.scatter(augmented_df['Value1'], augmented_df['Value2'], color='red', label='Augmented Data', alpha=0.7)
plt.title('Original vs Augmented Data')
plt.xlabel('Value1')
plt.ylabel('Value2')
plt.legend()
plt.grid(True)
plt.show()