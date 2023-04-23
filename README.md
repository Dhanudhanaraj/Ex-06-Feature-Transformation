# Ex-06-Feature-Transformation

## AIM:

To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION:

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
  
## ALGORITHM:
  
### STEP 1:
  
Read the given Data
  
### STEP 2:
  
Clean the Data Set using Data Cleaning Process
  
### STEP 3:
  
Apply Feature Transformation techniques to all the features of the data set
  
### STEP 4:
  
Print the transformed features
  
## PROGRAM:
  
NAME:D.DHANUMALYA.
  
REGISTER NUMBER:212222230030.
   
### Importing Libraries
```
# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
```

### Reading CSV File
```
# READ CSV FILES
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
### Basic Process
```
# BASIC PROCESS
df.head()

df.info()

df.describe()

df.tail()

df.shape

df.columns

df.isnull().sum()

df.duplicated()
```
### Before Transformation
```
# BEFORE TRANSFORMATION
# HIGHLY POSITIVE SKEW
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

# HIGHLY NEGATIVE SKEW
sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

# MODERATE POSITIVE SKEW
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

# MODERTE NEGATIVE SKEW
sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

```
### Log Transformation
```
# LOG TRANSFORMATION
# HIGHLY POSITIVE SKEW
df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

# MODERATE POSITIVE SKEW
df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

```
### Reciprocal Transformation
```
# RECIPROCAL TRANSFORMATION
# HIGHLY POSITIVE SKEW
df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

```
### Square Root Transformation
```
# SQUARE ROOT TRANSFORMATION
# HIGHLY POSITIVE SKEW
df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

```
### Power Transformation
```
# POWER TRANSFORMATION
# MODERATE POSITIVE SKEW
df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

# MODERATE NEGATIVE SKEW
from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")

df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

```
### Quantile Transformation
```
# QUANTILE TRANSFORMATION
# MODERATE NEGATIVE SKEW
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')

df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))

sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

```
## OUTPUT:

### Importing Libraries

![FT1](https://user-images.githubusercontent.com/119218812/233821812-e7f1a9c7-cc17-4564-b30d-aee78ff42d11.png)

### Reading CSV File

![FT2](https://user-images.githubusercontent.com/119218812/233821821-118ef8ba-3f4c-4568-9227-d78f6476a8c1.png)


### Basic Process

![FT3](https://user-images.githubusercontent.com/119218812/233821869-62f86286-ab87-4ff7-95cd-781c70e54958.png)

![FT4](https://user-images.githubusercontent.com/119218812/233821856-086abeac-6349-4299-b107-48e4c8381a3b.png)

![FT5](https://user-images.githubusercontent.com/119218812/233821866-b0c20096-b8ca-44c4-be7b-52dc26abb0ab.png)


### Before Transformation

#### Highly Positive Skew

![FT6](https://user-images.githubusercontent.com/119218812/233821917-59a8f8ab-c5b0-472e-b445-6fd08c63a494.png)


#### Highly Negative Skew

![FT7](https://user-images.githubusercontent.com/119218812/233821920-e9b8f64e-3686-4273-b9fc-134c9102f2ec.png)

#### Moderate Positive Skew

![FT8](https://user-images.githubusercontent.com/119218812/233821924-8d1a5917-05ab-4a03-98b3-6357300c8241.png)

#### Moderate Negative Skew

![FT9](https://user-images.githubusercontent.com/119218812/233821928-9ce47e99-1bdf-4f7d-89e8-f8a6b4a2e023.png)

### Log Transformation

#### Highly Positive Skew

![FT10](https://user-images.githubusercontent.com/119218812/233821938-06ca628b-7b87-4f1d-b517-28fc3c26d4a4.png)

#### Moderate Positive Skew

![FT11](https://user-images.githubusercontent.com/119218812/233821948-c8f44e61-3ae8-4c4c-8901-7d751e7f5ccf.png)

### Reciprocal Transformation

#### Highly Positive Skew

![FT12](https://user-images.githubusercontent.com/119218812/233821953-95b1f6bb-cb7e-4f73-b895-8d4b409404b8.png)

### Square Root Transformation

#### Highly Positive Skew

![FT13](https://user-images.githubusercontent.com/119218812/233821956-affca858-79a0-401d-9159-d1e4e5cf7d49.png)

### Power Transformation

#### Moderate Positive Skew

![FT14](https://user-images.githubusercontent.com/119218812/233821959-248734d4-0cf3-42c6-be26-9ddb0a6faff0.png)

#### Moderate Negative Skew

![FT15](https://user-images.githubusercontent.com/119218812/233821962-939c9194-0f14-4ac4-9080-d29ce168a075.png)

### Quantile Transformation

#### Moderate Negative Skew

![FT16](https://user-images.githubusercontent.com/119218812/233821967-f1779b92-72fb-41c9-ad41-a6c1359052f7.png)

## RESULT:

Thus feature transformation is done for the given dataset.
