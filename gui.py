import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import  mean_absolute_error, r2_score, mean_squared_error
from math import sqrt
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")
"# Predicting House price "
# st.title("Predicting House price ")
file = st.file_uploader('Please select a dataset', type=['csv', 'xlsx', 'xls'])
if file is None:
    st.stop()
if file.name.endswith('.csv'):
    df = pd.read_csv(file)
elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
    df = pd.read_excel(file)

"### Loading data"
st.success("Succesfully loaded data")
df

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

def draw_plots(plotfunc, include_num=False, include_cat=False):        
    global df
    cols = st.columns(5)
    n = 0

    num_col = df.select_dtypes('number')
    cat_col = df.select_dtypes('object')
    included_cols = []
    if include_num: included_cols.extend(num_col.columns.ravel())
    if include_cat: included_cols.extend(cat_col.columns.ravel())
    for c in included_cols:
        with cols[n]:
            fig = plt.figure(figsize=(10, 4))
            plotfunc(df[c])
            st.pyplot(fig)
        
        n += 1
        if n >= len(cols): n = 0
    
def display_visualization():
    global df
    num_col = df.select_dtypes('number')
    box_tab, heat_tab, bar_tab, hist_tab, count_tab, scatter_tab = st.tabs(['Box Plot', 'Heat Map', 'Bar Plot', 'Histogram', 'Count Plot', 'Scatter Plot'])

    with box_tab:
        draw_plots(sns.boxplot, True, False)
    with heat_tab:
        fig = plt.figure(figsize=(20,20))
        sns.heatmap(num_col.corr(), annot=True)
        st.pyplot(fig)
    with bar_tab:
        draw_plots(sns.barplot, False, True)
    with hist_tab:
        draw_plots(sns.histplot, True, False)
    with count_tab:
        draw_plots(sns.countplot, False, True)
    with scatter_tab:
        draw_plots(sns.scatterplot, True, True)


"### Missing value handling"
"Percentage of missing values in each column"
df.isnull().sum() / df.shape[0] * 100
items = st.multiselect('Select columns to be dropped', df.columns)
for col in items:
    df = df.drop(col, axis=1)

"### Visualization (Before Preprocessing)"
if st.button("Display"):
    display_visualization()

"### Choosing method for handling missing value"
cat_tab, num_tab = st.tabs(['Categorical', 'Numerical'])
with cat_tab:
    cat_col = df.select_dtypes('object')
    cat_col = cat_col.loc[:, cat_col.isnull().sum() > 0]    
    
    if len(cat_col.columns) == 0:
        st.warning("No missing values in categorical data")
    else:
        st.selectbox('Method for Categorical data', ['Simple Imputer'])
        strategy = st.selectbox('Strategy for simple imputer', ['mode', 'const'])
        if strategy == 'mode':
            imputer = SimpleImputer(strategy='most_frequent')
            cat_col
            df[cat_col.columns] = imputer.fit_transform(df[cat_col.columns])
            df[cat_col.columns]
        else: 
            for col in cat_col.columns:
                value = st.selectbox(f'{col} value', df[col].unique())
                imputer = SimpleImputer(strategy='constant', fill_value=value)
                df[[col]] = imputer.fit_transform(df[[col]].values.reshape(-1, 1))
                
            df[cat_col.columns]
    
with num_tab:
    num_col = df.select_dtypes('number')
    num_col = num_col.loc[:, num_col.isnull().sum() > 0]
    if len(num_col.columns) == 0:
        st.warning("No missing values in numerical data")
    else:
        imputer_type = st.selectbox('Method for Numerical data', ['Simple Imputer', 'KNN Imputer', 'Iterative Imputer'])
        if imputer_type == 'Simple Imputer':
            strategy = st.selectbox('Strategy for simple imputer', ['mean', 'median', 'const', 'mode'])
            if strategy == 'mean':
                imputer = SimpleImputer(strategy='mean')
                df[num_col.columns] = imputer.fit_transform(df[num_col.columns])
            elif strategy == 'median':
                imputer = SimpleImputer(strategy='median')
                df[num_col.columns] = imputer.fit_transform(df[num_col.columns])
            elif strategy == 'const':
                for col in num_col.columns:
                    value = st.number_input(f'{col} value', value=0)
                    imputer = SimpleImputer(strategy='constant', fill_value=value)
                    df[[col]] = imputer.fit_transform(df[[col]].values.reshape(-1, 1))
            elif strategy == 'mode':
                imputer = SimpleImputer(strategy='most_frequent')
                df[num_col.columns] = imputer.fit_transform(df[num_col.columns])
                
        elif imputer_type == 'KNN Imputer':
            n = st.slider('Number of neighbors', min_value=1, max_value=10, value=5)
            imputer = KNNImputer(n_neighbors=n)
            df[num_col.columns] = imputer.fit_transform(df[num_col.columns])
        elif imputer_type == 'Iterative Imputer':
            imputer = IterativeImputer()
            df[num_col.columns] = imputer.fit_transform(df[num_col.columns])
            
        df[num_col.columns]

"Percentage of missing values in each column after imputing"
df.isnull().sum() / df.shape[0] * 100

"### Detect and handle duplicates"
num_of_dup = df.duplicated().sum()
df.drop_duplicates(inplace=True)
st.success(f"Succesfully removed **{num_of_dup}** duplicates")

num_col = df.select_dtypes(include="number")
cat_col = df.select_dtypes(include="object")

def handle_outlier_IQR(col_name):
    global df
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col_name] >= lower_bound) & (df[col_name] <= upper_bound)]

def handle_outlier_Zscore():
    global df
    z_scores = np.abs(stats.zscore(df.select_dtypes('number')))
    df = df[(z_scores < 3).all(axis=1)]
    
"### Detecting and handling outliers"
before_outlier_count = len(df)
method = st.selectbox('Method for handling outlier', ['IQR', 'Z-score'])
if method == 'IQR':
    for c in num_col:
        handle_outlier_IQR(c)
elif method == 'Z-score':
    handle_outlier_Zscore()
        
st.success(f"Succesfully removed **{before_outlier_count - len(df)}** outliers using **{method}** method")

"### encoding categories"
if len(cat_col.columns) == 0:
    st.warning("No categorical data to be encoded")
else:
    encoder_type = st.selectbox('Method for encoding', ['Label Encoder', 'One Hot Encoder'])
    if encoder_type == 'Label Encoder':
        encode = LabelEncoder()
        for c in cat_col:
            df[c] = encode.fit_transform(df[c])
    elif encoder_type == 'One Hot Encoder':
        encode = OneHotEncoder(sparse_output=False, drop='first')
        encoded_features = encode.fit_transform(df[cat_col.columns])
        encoded_df = pd.DataFrame(encoded_features, columns=encode.get_feature_names_out(cat_col.columns))
        df = df.drop(cat_col.columns, axis=1).reset_index(drop=True)
        df = pd.concat([df, encoded_df], join='inner', axis=1)
        cat_col = encoded_df


"### Visualization (After Preprocessing)"
if st.button("Display2", ):
    display_visualization()
    
label_col = st.selectbox('Select label to be predicted', df.columns)
num_col = num_col.drop(label_col,axis=1)
"### Split data"
x = df.drop(label_col, axis=1) 
y = df[label_col]
st.success("Succesfully split into feature and label")

method = st.selectbox('Method for splitting data', ['Train_test_split', 'KFold'])
if method == 'KFold':
    number_of_splits = st.slider('Number of splits', min_value=2, max_value=10, value=2)
    st.success("Succesfully split into **train** and **test** data")
    "### Normalization for data"
    method = st.selectbox('Select scaler type', ['Standard', 'MinMax'])
    if method == 'Standard':
        scaler = StandardScaler()
    elif method == 'MinMax': 
        scaler = MinMaxScaler()

    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        for c in num_col:
            x_train[[c]] = scaler.fit_transform(x_train[[c]])
            x_test[[c]] = scaler.fit_transform(x_test[[c]])
    
    y_train = scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test = scaler.transform(y_test.values.reshape(-1, 1))
    st.success("Succesfully scaled **label** data")
    y_train.T
    
    "### Model selection and training"
    model_type = st.selectbox('Select model to be used', ['RandomForestRegressor', 'DecisionTreeRegressor', 'LinearRegressor'])
    if model_type == 'RandomForestRegressor':
        model = RandomForestRegressor()
    elif model_type == 'DecisionTreeRegressor':
        model = DecisionTreeRegressor()
    elif model_type == 'LinearRegressor':
        model = LinearRegression()
    
    with st.spinner("Training model..."):
        model.fit(x_train, y_train)
    st.success("Succesfully trained model")

    "### Model Prediction"
    with st.spinner("Predicting model..."):
        m_pridect = model.predict(x_test)
    st.success("Succesfully ran model prediction")

    "### Model Evaluation"
    with st.spinner("Evaluating model..."):
        cv = cross_validate(model, x, y, cv=kf, scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'])
        mse = -cv['test_neg_mean_squared_error'].mean()
        rmse = sqrt(mse)
        mae = -cv['test_neg_mean_absolute_error'].mean()
        r2 = cv['test_r2'].mean()
        st.success("Succesfully evaluated model")    
    report = pd.DataFrame({
        'Metric': ['Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error', 'R2 Score'],
        'Value': [mse, rmse, mae, r2]
    })

    report

    
elif method == 'Train_test_split':
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    st.success("Succesfully split into train and test data")
    "### Normalization for data"
    method = st.selectbox('Select scaler type', ['Standard', 'MinMax'])

    if method == 'Standard':
        scaler = StandardScaler()
    elif method == 'MinMax': 
        scaler = MinMaxScaler()

    for c in num_col:
        x_train[[c]] = scaler.fit_transform(x_train[[c]])
        x_test[[c]] = scaler.fit_transform(x_test[[c]])
    st.success("Succesfully scaled **feature** data")
    x_train[num_col.columns]

    y_train = scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test = scaler.transform(y_test.values.reshape(-1, 1))
    st.success("Succesfully scaled **label** data")
    y_train.T

    "### Model selection and training"

    model_type = st.selectbox('Select model to be used', ['RandomForestRegressor', 'DecisionTreeRegressor', 'LinearRegressor'])
    if model_type == 'RandomForestRegressor':
        model = RandomForestRegressor()
    elif model_type == 'DecisionTreeRegressor':
        model = DecisionTreeRegressor()
    elif model_type == 'LinearRegressor':
        model = LinearRegression()
    with st.spinner("Training model..."):
        model.fit(x_train,y_train)
    st.success("Succesfully trained model")

    "### Model Prediction"
    with st.spinner("Predicting model..."):
        m_pridect = model.predict(x_test)
    st.success("Succesfully ran model prediction")

    "### Model Evaluation"
    with st.spinner("Evaluating model..."):
        mse = mean_squared_error(y_test, m_pridect)
        rmse = sqrt(mse)
        mae = mean_absolute_error(y_test, m_pridect)
        r2 = r2_score(y_test, m_pridect)
        report = pd.DataFrame({
            'Metric': ['Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error', 'R2 Score'],
            'Value': [mse, rmse, mae, r2]
        })

        report



