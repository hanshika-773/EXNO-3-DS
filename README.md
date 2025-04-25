## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/a31b52d3-dbc6-4ade-ac0a-cf87d81a22e4)

# Ordinal Encoder
```
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

# Sample data for df (assuming 'ord_2' is a column with ordinal data)
data = {
    'ord_2': ['Hot', 'Warm', 'Cold', 'Hot', 'Cold', 'Warm', 'Warm']
}

# Create DataFrame
df = pd.DataFrame(data)

# OrdinalEncoder initialization
e1 = OrdinalEncoder(categories=[['Cold', 'Warm', 'Hot']])

# Apply Ordinal Encoding
df['ord_2_encoded'] = e1.fit_transform(df[['ord_2']])

# Display the DataFrame
df

```
![image](https://github.com/user-attachments/assets/7459d163-5b2a-492f-81eb-a8417e2a3887)
# Label Encoder
```
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Sample data for df (assuming we are encoding a categorical column 'category')
data = {
    'category': ['apple', 'banana', 'apple', 'orange', 'banana', 'orange', 'apple']
}

# Create DataFrame
df = pd.DataFrame(data)

# Create a copy of df (to preserve the original dataframe)
dfc = df.copy()

# Initialize LabelEncoder
le = LabelEncoder()

# Apply LabelEncoder to the 'category' column
dfc['category_encoded'] = le.fit_transform(dfc['category'])

# Display the DataFrame
dfc
```
![image](https://github.com/user-attachments/assets/ea5d8bc0-7bed-4611-a0bf-3678d45c320a)
# OneHot Encoder
```
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Sample data for df (assuming we are encoding a column 'nom_0')
data = {
    'nom_0': ['cat', 'dog', 'cat', 'bird', 'dog', 'bird', 'cat']
}

# Create DataFrame
df = pd.DataFrame(data)

# Create a copy of df (to preserve the original dataframe)
df2 = df.copy()

# Initialize OneHotEncoder (using sparse_output=False for scikit-learn v0.24+)
ohe = OneHotEncoder(sparse_output=False)

# Apply OneHotEncoder to the 'nom_0' column and convert to DataFrame
encoded = ohe.fit_transform(df2[['nom_0']])

# Convert the result into a DataFrame and add column names
encoded_df = pd.DataFrame(encoded, columns=ohe.categories_[0])

# Concatenate the original DataFrame with the encoded DataFrame
df2 = pd.concat([df2, encoded_df], axis=1)

# Display the DataFrame with One-Hot Encoding
df2

```
![image](https://github.com/user-attachments/assets/a8e5cd7b-1091-487d-97d5-997786f2a227)
```
pip install --upgrade category_encoders
print(df.columns)
```
# Binary Encoder
```
from category_encoders import BinaryEncoder
import pandas as pd

# Load the dataset
df = pd.read_csv("/content/data.csv")

# Initialize BinaryEncoder for the 'City' column
be = BinaryEncoder(cols=['City'])

# Apply the encoder to the DataFrame
df_encoded = be.fit_transform(df)

# Display the resulting DataFrame
df_encoded
```
![image](https://github.com/user-attachments/assets/9cb95aa1-77c2-4b2e-a387-4df35b5c1171)
# Target Encoder
```
from category_encoders import TargetEncoder
import pandas as pd

# Load your data
df = pd.read_csv("/content/data.csv")

# Make a copy
cc = df.copy()

# Initialize TargetEncoder on 'City' column
te = TargetEncoder(cols=['City'])

# Fit and transform the encoder using the target column
cc['City_encoded'] = te.fit_transform(cc['City'], cc['Target'])

# Show the resulting DataFrame

# Make a copy
cc = df.copy()

# Initialize TargetEncoder on 'City' column
te = TargetEncoder(cols=['City'])

# Fit and transform the encoder using the target column
cc['City_encoded'] = te.fit_transform(cc['City'], cc['Target'])

# Show the resulting DataFrame
cc
```
![image](https://github.com/user-attachments/assets/546e4e03-2a05-49d4-a2ff-e3f7f09faa3d)
# Feature Transformation
```
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import PowerTransformer
from scipy.stats import boxcox

# Load the dataset
df = pd.read_csv("/content/Data_to_Transform.csv")

# Check original skewness
print("Original Skewness:\n", df.skew())

# Log Transformation (after shifting)
shifted = df['Moderate Negative Skew'] - df['Moderate Negative Skew'].min() + 1
df['Log_Transformed'] = np.log(shifted)

# Reciprocal Transformation
df['Reciprocal_Transformed'] = 1 / shifted

# Square Root Transformation
df['Sqrt_Transformed'] = np.sqrt(shifted)

# Square Transformation
df['Square_Transformed'] = np.square(df['Moderate Negative Skew'])

# Box-Cox Transformation (only for positive values)
df['BoxCox_Transformed'], _ = boxcox(shifted)

# Yeo-Johnson Transformation
pt = PowerTransformer(method='yeo-johnson')
df['YeoJohnson_Transformed'] = pt.fit_transform(df[['Moderate Negative Skew']])

# Skewness after transformation
print("\nSkewness After Transformation:")
print(df[['Log_Transformed', 'Reciprocal_Transformed', 'Sqrt_Transformed',
          'Square_Transformed', 'BoxCox_Transformed', 'YeoJohnson_Transformed']].skew())

# QQ-Plots
sm.qqplot(df['Moderate Negative Skew'], line='45')
plt.title('Original')
plt.show()

sm.qqplot(df['Log_Transformed'], line='45')
plt.title('Log Transformed')
plt.show()

sm.qqplot(df['BoxCox_Transformed'], line='45')
plt.title('Box-Cox Transformed')
plt.show()

sm.qqplot(df['YeoJohnson_Transformed'], line='45')
plt.title('Yeo-Johnson Transformed')
plt.show()
```
![image](https://github.com/user-attachments/assets/b7d2094e-d52f-4066-b868-9fef13ec35e4)

![image](https://github.com/user-attachments/assets/afa660dd-5002-4c01-987c-724c18d4a878)

![image](https://github.com/user-attachments/assets/67dcb4c1-4068-4372-8a6d-3d81a9cb500d)

![image](https://github.com/user-attachments/assets/5cd86051-67f3-47a2-aaea-834267db3c8e)

![image](https://github.com/user-attachments/assets/4236e87c-bc3c-45cf-b4e8-4b3e94b9bbe4)

# RESULT:
Thus, successfully read the given data and performed Feature Encoding and Transformation process and saved the data to a file.

       
