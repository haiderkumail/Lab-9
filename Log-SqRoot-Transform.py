import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

data = {'Value': [1, 4, 10, 50, 100, 250, 500, 1000]}
df = pd.DataFrame(data)

df['Log_Value'] = np.log(df['Value'] + 1)
df['Sqrt_Value'] = np.sqrt(df['Value'])

print(df)

plt.figure(figsize=(8, 6))
plt.plot(df['Value'], df['Log_Value'], label='Log(Value + 1)', marker='o')
plt.plot(df['Value'], df['Sqrt_Value'], label='Sqrt(Value)', marker='x')
plt.title('Log and Sqrt Transformations of Value')
plt.xlabel('Value')
plt.ylabel('Transformed Value')
plt.legend()
plt.grid(True)
plt.show()