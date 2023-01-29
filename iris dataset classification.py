import pandas as pd
from google.colab import files
data_to_load = files.upload()
import io
df = pd.read_csv(io.BytesIO(data_to_load['iris.csv']))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
df=pd.read_csv('iris.csv')
df
df.head()
df.isnull().sum()
//catplot
sns.catplot(x='class',hue='class',kind='count', data=df)
//Barplot for class vs petal_width
plt.bar(df['class'],df['petal_width'])
sns.set()
sns.pairplot(df[['sepal_length','sepal_width','petal_length','petal_width','class']], hue="class")
df.describe()
df.columns
df.info
df
#dropping the species column
x=df.drop(['class'],axis=1)
x
Label_Encode=LabelEncoder()
y=df['class']
y=Label_Encode.fit_transform(y)
y
df['class'].nunique()
x=np.array(x)
x
//SPLITTING THE DATASET
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train
x_train.shape
x_test.shape
y_test.shape
y_train.shape
pip install standard-scaler
//MODEL PREPARATION-KNN ALGORITHM
#USE KNN CLASSIFIER
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
#TRAIN KNN CLASSIFIER
knn.fit(x_train,y_train)
#Evaluate the model
y_pred=knn.predict(x_test)
y_pred
y_test
#check the accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
