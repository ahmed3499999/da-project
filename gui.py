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

"Data info"
from io import StringIO
buffer = StringIO()
df.info(buf=buffer)
st.code(buffer.getvalue())

"### Missing value handling"
"Percentage of missing values in each column"
df.isnull().sum() / df.shape[0] * 100

"#### Simple imputer with mean for certain columns"
from sklearn.impute import SimpleImputer
imput = SimpleImputer(strategy="mean")
df[["LotFrontage","MasVnrArea"]] = imput.fit_transform(df[["LotFrontage","MasVnrArea"]])
df[["GarageCars"]] = imput.fit_transform(df[["GarageCars"]])
df[['LotFrontage','MasVnrArea','GarageCars']]

"#### Simple imputer with mode for certain columns"
imputer = SimpleImputer(strategy="most_frequent")
df[["BsmtQual" , "BsmtCond" , "BsmtExposure" , "BsmtFinType1" , "BsmtFinType2" , "Electrical" , "GarageType" , "GarageFinish"]] = imputer.fit_transform(df[["BsmtQual" , "BsmtCond" , "BsmtExposure" , "BsmtFinType1" , "BsmtFinType2" , "Electrical" , "GarageType" , "GarageFinish"]])
df[['BsmtQual' , 'BsmtCond' , 'BsmtExposure' , 'BsmtFinType1' , 'BsmtFinType2' , 'Electrical' , 'GarageType' , 'GarageFinish']]

"Percentage of missing values in each column after imputing"
df.isnull().sum() / df.shape[0] * 100

"### Detect and handle duplicates"
df.drop_duplicates(inplace=True)
st.success(f"Succesfully removed **{df.duplicated().sum()}** duplicates")

num_col = df.select_dtypes(include="number")
cat_col = df.select_dtypes(include="object")

def detect_handling_outlier(col_name):
    global df
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # mean_value = df[col_name].mean()
    # df[col_name] = np.where((df[col_name] < lower_bound) | (df[col_name] > upper_bound) , mean_value, df[col_name])
    df = df[(df[col_name] >= lower_bound) & (df[col_name] <= upper_bound)]


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
before_outlier_count = len(df)
draw_plots(sns.boxplot, True, False)

"After handling outlier for features"
for c in num_col:
    detect_handling_outlier(c)
st.success(f"Succesfully removed **{before_outlier_count - len(df)}** outliers")
draw_plots(sns.boxplot, True, False)

# "After handling outlier for labels"
# @st.cache_data
# def remove_outliers_iqr(dataframe, column):
#     Q1 = df[column].quantile(0.25)
#     Q3 = df[column].quantile(0.75)

#     IQR = Q3 - Q1
#     lower = Q1 - 1.5 * IQR
#     upper = Q3 + 1.5 * IQR
#     return df[(df[column] >= lower) & (df[column] <= upper)]


# df = remove_outliers_iqr(df, "SalePrice")
# st.success("Handled succesfully")

"### Visualization"
heat_tab, bar_tab, hist_tab, count_tab, scatter_tab = st.tabs(['Heat Map', 'Bar Plot', 'Histogram', 'Count Plot', 'Scatter Plot'])

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
with scatter_tab:
    draw_plots(sns.scatterplot, True, True)

"### Label encoding categories"
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
for c in cat_col:
    df[c] = encode.fit_transform(df[c])
st.success("Succesfully Encoded features")
df[cat_col.columns]

"### Split data"
x = df.drop("SalePrice", axis=1) 
y = df["SalePrice"]
st.success("Succesfully split into feature and label")

num_col = num_col.drop('SalePrice',axis=1)
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42) 
st.success("Succesfully split into train and test data")

"### Scaling features using minmax"
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
for c in num_col:
    x_train[[c]] = min_max_scaler.fit_transform(x_train[[c]])
    x_test[[c]] = min_max_scaler.fit_transform(x_test[[c]])
st.success("Succesfully scaled **feature** data")
x_train[num_col.columns]

y_train_scaled = min_max_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = min_max_scaler.transform(y_test.values.reshape(-1, 1))
st.success("Succesfully scaled **label** data")
y_train_scaled

"### Model selection and training"
"Using Random Forest Regressor"
from sklearn.ensemble import  RandomForestRegressor
model = RandomForestRegressor()
model.fit(x_train,y_train)
st.success("Succesfully trained model")

"### Model Prediction"
m_pridect = model.predict(x_test)
st.success("Succesfully ran model prediction")

"### Model Evaluation"
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from math import sqrt

mse = mean_squared_error(y_test, m_pridect)
rmse = sqrt(mse)
mae = mean_absolute_error(y_test, m_pridect)
r2 = r2_score(y_test, m_pridect)
report = pd.DataFrame({
    'Metric': ['Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error', 'R2 Score'],
    'Value': [mse, rmse, mae, r2]
})

report
