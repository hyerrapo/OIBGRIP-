import pandas as pd
from google.colab import files
data_to_load = files.upload()
import io
df = pd.read_csv(io.BytesIO(data_to_load['Unemployment in India.csv']))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
data=pd.read_csv("Unemployment in India.csv")
print(data.head())
print(data.isnull().sum())
data.columns= ["Region","Date","Frequency",
               "Estimated Unemployment Rate",
               "Estimated Employed",
               "Estimated Labour Participation Rate",
               "Area"]
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr())
plt.show()
data.columns= ["Region","Date","Frequency",
               "Estimated Unemployment Rate","Estimated Employed","Estimated Labour Participation Rate","Area"]
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Employed", hue="Area", data=data)
plt.show()
plt.figure(figsize=(12, 10))
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Unemployment Rate", hue="Area", data=data)
plt.show()
plt.figure(figsize=(14, 12))
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Unemployment Rate", hue="Region", data=data)
plt.show()
sns.heatmap(unemployment.isnull(),cbar=False)
df.columns
