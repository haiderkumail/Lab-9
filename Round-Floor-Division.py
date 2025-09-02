import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

data = {'Value': [1.2, 4.6, 10.8, 50.3, 100.9, 250.1, 500.5, 1000.7]}
df = pd.DataFrame(data)

df['Rounded_Value'] = df['Value'].round()
df['Floored_Value'] = df['Value'] // 1

print(df)

df.plot(kind='bar', figsize=(10, 6))
plt.title("Comparison of Original, Rounded, and Floored Values")
plt.xlabel("Index")
plt.ylabel("Values")
plt.xticks(rotation=0)
plt.legend(["Original Value", "Rounded Value", "Floored Value"])
plt.show()