"""

Author: Raghavendra Tapas
Context: Predicting Real Estate Prices in Bengaluru, India
Dataset Source: https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data
Disclaimer: It is to be noted that this code serves as educational and personal project showcase.
It is also important to note that this is code only and no explainations are provided. For explanation see the jupyter notebook.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import matplotlib

# Importing dataset
df1 = pd.read_csv("bengaluru_house_prices.csv")
df1.groupby('area_type')['area_type'].agg('count')

# Removing a set of features that are not significant
df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')

# Removing the null values
df3 = df2.dropna()

# Removing the text from size column
df4 = df3
df4['size'] = df4['size'].apply(lambda x: int(x.split(' ')[0]))

# Converting range data to mean
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

# Function Replacing data range to mean values
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/ 2
    try:
        return float(x)
    except:
        return None

df5 = df4.copy()
df5.total_sqft = df5.total_sqft.apply(convert_sqft_to_num)
df5 = df5[df5.total_sqft.notnull()]

# rename size to BHK(bedroom, hall, kitchen)
df5.rename(columns = {'size':'bhk'}, inplace = True)

# Removing the outliers.
df5['sqft_per_bedroom'] = df5.total_sqft/df5.bhk

# deleting the rows that have sqft_per_bedroom lesser than 300
df6 = df5[~(df5.sqft_per_bedroom < 300)]

# removing the column as it has no further use case.
df6.drop(['sqft_per_bedroom'], axis = 1, inplace = True)

# Reduce number of locations
location_stats = df6['location'].value_counts(ascending=False)
less_than_10 = location_stats[location_stats<=10]
df6.location = df6.location.apply(lambda x: 'other' if x in less_than_10 else x)

# Outlier Removal Using Standard Deviation and Mean
df6['price_per_sqft'] = (df6.price) * 100000/df6.total_sqft

def remove_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_outliers(df6)
df7.shape, df6.shape

# Pearson Co-Relation
sns.heatmap(df7.corr(method='pearson'), annot=True, cmap='coolwarm');

# custom function to plot scatter plot
def plot_scatter(df, input_bhk, input_location, input_color, width, height):
    filtered_results = df[(df.bhk == input_bhk) & (df.location == input_location)]
    matplotlib.rcParams['figure.figsize'] = (width, height)
    plt.scatter(filtered_results.total_sqft, filtered_results.price, color = input_color, label = (f'{input_location}, {input_bhk} BHK'), s=30)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.legend()

# Checking the outliers
plot_scatter(df7, 2, "Rajaji Nagar", 'blue', 10, 6)
plot_scatter(df7, 3, "Rajaji Nagar", 'red', 10, 6)

plot_scatter(df7, 2, "Hebbal", 'blue', 10, 6)
plot_scatter(df7, 3, "Hebbal", 'red', 10, 6)

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df8 = remove_bhk_outliers(df7)

# Plot same scatter chart again to visualize price_per_sqft for 2 BHK and 3 BHK properties
plot_scatter(df7, 3, "Hebbal", 'red', 10, 6)
plot_scatter(df8, 3, "Hebbal", 'blue', 10, 6)

plot_scatter(df7, 3, "Rajaji Nagar", 'red', 10, 6)
plot_scatter(df8, 3, "Rajaji Nagar", 'blue', 10, 6)

# Delete the rows that are no longer needed
df9 = df8.drop(['bhk','price_per_sqft'],axis='columns')

# One Hot Encoding For Location
dummies = pd.get_dummies(df9.location)
df10 = pd.concat([df9,dummies.drop('other',axis='columns')],axis='columns')
df11 = df10.drop('location',axis='columns')

# Creating Model
X = df11.drop(['price'],axis='columns')
y = df11.price

# test and training data split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=10)

# Using linear regression
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)

# K Fold cross validation to measure accuracy of our LinearRegression model
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)

# Grid Search CV to check other models
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

# function to use linear regression, lasso and decision tree on the dataset and evaluate their scores

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)

# Best Model: Linear Regression

# Exporting the Machine learning model
import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)

# Export location and column information to a file
import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))