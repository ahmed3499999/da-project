import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"# Predicting House price in california"
"### Loading data"
df = pd.read_csv(r"Houses.csv")
df
"### Dropping unnecessary columns"
"Id, Alley,MasVnrType, PoolQC, Fence, MiscFeature, FireplaceQu, GarageCond, GarageQual, GarageYrBlt,Utilities, MiscVal"
cols_to_drop = [
    'Id', 'Alley','MasVnrType', 'PoolQC', 'Fence', 'MiscFeature', 
    'FireplaceQu', 'GarageCond', 'GarageQual', 'GarageYrBlt',
    'Utilities', 'MiscVal'
]
df = df.drop(columns=cols_to_drop , axis=1)
x = df.head(10)
x

"### EDA"
"Numerical data description"
x = df.describe()
x
"Categorical data description"
x = df.select_dtypes(include="object").describe()
x

"### Missing value handling"
"Percentage of missing values in each column"
df.isnull().sum() / df.shape[0] * 100

"#### Simple imputer with mean for certain columns"
"LotFrontage,MasVnrArea,GarageCars"
from sklearn.impute import SimpleImputer
imput = SimpleImputer(strategy="mean")
df[["LotFrontage","MasVnrArea"]] = imput.fit_transform(df[["LotFrontage","MasVnrArea"]])
df[["GarageCars"]] = imput.fit_transform(df[["GarageCars"]])

"#### Simple imputer with mode for certain columns"
"BsmtQual , BsmtCond , BsmtExposure , BsmtFinType1 , BsmtFinType2 , Electrical , GarageType , GarageFinish"
imputer = SimpleImputer(strategy="most_frequent")
df[["BsmtQual" , "BsmtCond" , "BsmtExposure" , "BsmtFinType1" , "BsmtFinType2" , "Electrical" , "GarageType" , "GarageFinish"]] = imputer.fit_transform(df[["BsmtQual" , "BsmtCond" , "BsmtExposure" , "BsmtFinType1" , "BsmtFinType2" , "Electrical" , "GarageType" , "GarageFinish"]])

"Percentage of missing values in each column after imputing"
df.isnull().sum() / df.shape[0] * 100

"### Detect and handle duplicates"
f"Number of duplicates: {df.duplicated().sum()}"

num_col = df.select_dtypes(include="number")
cat_col = df.select_dtypes(include="object")

def detect_handling_outlier(col_name):
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mean_value = df[col_name].mean()
    df[col_name] = np.where((df[col_name] < lower_bound) | (df[col_name] > upper_bound) , mean_value, df[col_name])


def draw_plots(plotfunc, include_num=False, include_cat=False):        
    cols = st.columns(5)
    n = 0

    if include_num:
        for c in num_col:
            with cols[n]:
                fig = plt.figure(figsize=(10, 4))
                plotfunc(df[c])
                st.pyplot(fig)
            
            n += 1
            if n >= len(cols): n = 0
    
    if include_cat:
        for c in cat_col:
            with cols[n]:
                fig = plt.figure(figsize=(10, 4))
                plotfunc(df[c])
                st.pyplot(fig)
            
            n += 1
            if n >= len(cols): n = 0
    
"### Detecting and handling outliers"
"Before handling outlier"
draw_plots(sns.boxplot, True, False)
    
"After handling outlier for features"
for c in num_col:
    detect_handling_outlier(c)
draw_plots(sns.boxplot, True, False)

"After handling outlier for labels"
@st.cache_data
def remove_outliers_iqr(dataframe, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]


df = remove_outliers_iqr(df, "SalePrice")
st.success("Handled succesfully")

"### Visualization"
heat_tab, bar_tab, hist_tab, count_tab = st.tabs(['Heat Map', 'Bar Plot', 'Histogram', 'Count Plot'])

with heat_tab:
    st.snow()
    fig = plt.figure(figsize=(20,20))
    sns.heatmap(num_col.corr(), annot=True)
    st.pyplot(fig)
with bar_tab:
    st.balloons()
    draw_plots(sns.barplot, False, True)
with hist_tab:
    st.balloons()
    draw_plots(sns.histplot, True, False)
with count_tab:
    st.balloons()
    draw_plots(sns.countplot, False, True)
    
"### Label encoding categories"
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
for c in cat_col:
    df[c] = encode.fit_transform(df[c])
st.success("Succesfully Encoded features")

"### Split data"
x = df.drop("SalePrice", axis=1) 
y = df["SalePrice"]
st.success("Succesfully split into feature and label")

num_col2 = x.select_dtypes(include="number")
cat_col2 = x.select_dtypes(include="object")
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42) 
st.success("Succesfully split into train and test data")

"### Scaling data using minmax"
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
for c in num_col2:
    x_train[[c]] = min_max_scaler.fit_transform(x_train[[c]])
    x_test[[c]] = min_max_scaler.fit_transform(x_test[[c]])
st.success("Succesfully scaled feature data")

y_train_scaled = min_max_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = min_max_scaler.transform(y_test.values.reshape(-1, 1))
st.success("Succesfully scaled label data")

"### Model selection and training"
"Using Random Forest Regressor"
from sklearn.ensemble import  RandomForestRegressor
model = RandomForestRegressor()
model.fit(x_train,y_train)
st.success("Succesfully trained model")

"### Model Prediction"
m_pridect = model.predict(x_test)
st.success("Succesfully ran model prediction")

"### Model Performance"
from sklearn.metrics import r2_score
st.success(f"Performace metric: **{r2_score(y_test,m_pridect)}**")