# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:

1.Filter Method

2.Wrapper Method

3.Embedded Method

# CODING AND OUTPUT:
```py
import numpy as np
from scipy import stats
import pandas as pd
df=pd.read_csv('/content/bmi.csv')
df.head()
```
![image](https://github.com/user-attachments/assets/7fa86a6a-2485-411d-833d-2ff5c1825908)

```py
df.dropna()
```
![image](https://github.com/user-attachments/assets/9c0a528b-7f1e-4aab-85a2-a6fccb07abec)

```py
max_vals=np.max(np.abs(df[['Height','Weight']]))
```
![image](https://github.com/user-attachments/assets/1ba67656-ce10-45f0-9b29-6ba97b875b0b)


```py
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Head','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```

![image](https://github.com/user-attachments/assets/8fc97d9f-65fb-4858-b495-5bba326128b4)


```py
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/cd9e4cd2-f6f0-49b3-952b-2ff0a9a938ba)

```py
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/60d371ee-691e-491f-87a7-47fca20f22e7)

```py
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/ad318030-5816-4450-8ebe-9fa4da4b7945)

```py
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/1da52f2e-d207-49f1-a8ad-a9d926b46ad2)

```py
import pandas as pd
from sklearn.feature_selection import SelectKBest,mutual_info_classif, f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
df
```
![image](https://github.com/user-attachments/assets/38eb0644-f099-42c1-b42d-45ff5f0f5c81)

```py
x=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/d93cc8cb-ea0e-4a04-8238-a639c4e26c30)


```py
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head
```
![image](https://github.com/user-attachments/assets/5a9285c7-1a03-47bb-9472-e4ba14f38dcb)


```py
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/42251fc2-bc23-4181-955a-1f1789513225)

```py
chi2, p, _, _=chi2_contingency(contingency_table)
print("Chi-Square Statistic: {chi2}")
print(f"P-value:{p}")
```
![image](https://github.com/user-attachments/assets/0537aad8-3278-466a-b3b4-7f2af05cfc4a)


# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
