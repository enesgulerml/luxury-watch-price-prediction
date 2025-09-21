import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

data = pd.read_csv("streamlit_plus/Luxury watch.csv")
df = data.copy()

df.head()

df.isnull().sum()

df["Power Reserve"] = (
    df["Power Reserve"]
    .str.replace("hours", "", regex=False)
    .str.strip()
)

df["Water Resistance"] = (
    df["Water Resistance"]
    .str.replace("meters", "", regex=False)
    .str.strip()
)

df["Power Reserve"] = pd.to_numeric(df["Power Reserve"], errors="coerce")
df["Water Resistance"] = pd.to_numeric(df["Water Resistance"], errors="coerce")

df.head()

## Integer ##
df["Water Resistance"].isnull().sum()

## Float ##
df["Case Diameter (mm)"].isnull().sum()

df["Case Thickness (mm)"].isnull().sum()

df["Band Width (mm)"].isnull().sum()

df["Power Reserve"].isnull().sum()

df[df["Power Reserve"].isnull()]

def fill_power_reserve(row):
    if pd.isna(row['Power Reserve']):
        if row['Movement Type'] == 'Quartz':
            return 0
        elif row['Movement Type'] == 'Eco-Drive':
            return 999
        elif row['Movement Type'] in ['Automatic', 'Manual']:
            return 48
        else:
            return np.nan
    else:
        return row['Power Reserve']

df['Power Reserve'] = df.apply(fill_power_reserve, axis=1)

df[df["Power Reserve"].isnull()]

## Object ##
df.dtypes
df.head()

df.isnull().sum()

df[df["Complications"].isnull()]

df['Complications'] = df['Complications'].fillna('None')

df["Price (USD)"].dtype


df['Price (USD)'] = df['Price (USD)'].str.replace(',', '')

df['Price (USD)'] = df['Price (USD)'].astype(float)

df.head()


group_cols = ['Brand', 'Model', 'Case Material', 'Strap Material', 'Power Reserve']
df['Price_filled'] = df['Price (USD)']

for name, group in df.groupby(group_cols):
    median_val = group['Price_filled'].median()
    if not np.isnan(median_val):
        df.loc[group.index, 'Price_filled'] = df.loc[group.index, 'Price_filled'].fillna(median_val)


for name, group in df.groupby(['Brand', 'Model']):
    median_val = group['Price_filled'].median()
    if not np.isnan(median_val):
        df.loc[group.index, 'Price_filled'] = df.loc[group.index, 'Price_filled'].fillna(median_val)

df['Price_filled'].fillna(df['Price (USD)'].median(), inplace=True)

df['Price (USD)'] = df['Price_filled']
df.drop(columns=['Price_filled'], inplace=True)

df.isnull().sum()

#### Outlier ####
df.dtypes
df.head()

# Water Resistance #
## There are nearly 70 outliers, but I didn't change them anyway. Because, this values can be true.
q1 = df["Water Resistance"].quantile(0.25)
q3 = df["Water Resistance"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[(df['Water Resistance'] < lower_bound) | (df['Water Resistance'] > upper_bound)]

print(outliers)

# Case Diameter (mm) #
## Same with Water Resistance.
q1 = df["Case Diameter (mm)"].quantile(0.25)
q3 = df["Case Diameter (mm)"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[(df['Case Diameter (mm)'] < lower_bound) | (df['Case Diameter (mm)'] > upper_bound)]

print(outliers)

# Case Thickness (mm) #
## No outliers.
q1 = df["Case Thickness (mm)"].quantile(0.25)
q3 = df["Case Thickness (mm)"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[(df['Case Thickness (mm)'] < lower_bound) | (df['Case Thickness (mm)'] > upper_bound)]

print(outliers)

# Band Width (mm) #
## Same with Water Resistance.
q1 = df["Band Width (mm)"].quantile(0.25)
q3 = df["Band Width (mm)"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[(df['Band Width (mm)'] < lower_bound) | (df['Band Width (mm)'] > upper_bound)]

print(outliers)

# Power Reserve #
## There were two values around 999. They were eco-drive model. They have very long battery usage. But I clip it to 120.
q1 = df["Power Reserve"].quantile(0.25)
q3 = df["Power Reserve"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[(df['Power Reserve'] < lower_bound) | (df['Power Reserve'] > upper_bound)]

print(outliers)

df["Power Reserve"] = df["Power Reserve"].clip(upper=120)

# Price (USD) #
q1 = df["Price (USD)"].quantile(0.25)
q3 = df["Price (USD)"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[(df['Price (USD)'] < lower_bound) | (df['Price (USD)'] > upper_bound)]

print(outliers)

df['Price_log'] = np.log1p(df['Price (USD)'])

df['Price (USD)'] = df['Price_log']

df.drop(columns=['Price_log'], inplace=True)

df.head()

## Encoding ##



cat_cols = ['Brand', 'Model', 'Case Material', 'Strap Material',
            'Movement Type', 'Dial Color', 'Crystal Material', 'Complications']


for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

df.head()

df.dtypes

## XGBoost ##

X = df.drop(columns=['Price (USD)'])
y = df['Price (USD)']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


xgb_model.fit(X_train, y_train)


y_pred = xgb_model.predict(X_test)

y_test_usd = np.expm1(y_test)
y_pred_usd = np.expm1(y_pred)


rmse_usd = np.sqrt(mean_squared_error(y_test_usd, y_pred_usd))
print(f"Test RMSE (USD): {rmse_usd:.2f}")


mae = mean_absolute_error(y_test, y_pred)


r2 = r2_score(y_test, y_pred)
n = X_test.shape[0]
p = X_test.shape[1]
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"Test MAE: {mae:.4f}")
print(f"Test R²: {r2:.4f}")
print(f"Test Adjusted R²: {r2_adj:.4f}")


# Hyperparameter #

X = df.drop(columns=['Price (USD)'])
y = df['Price (USD)']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)


param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 5),
    'reg_alpha': uniform(0, 5),
    'reg_lambda': uniform(0, 5)
}


random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_dist,
    n_iter=50,
    scoring='neg_root_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)


random_search.fit(X_train, y_train)


best_params = random_search.best_params_
best_score = -random_search.best_score_
print("Best RMSE (log-scale):", best_score)
print("Best hyperparameters:", best_params)


best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
r2_test = r2_score(y_test, y_pred)
print(f"Test RMSE (log-scale): {rmse_test:.4f}")
print(f"Test R² (log-scale): {r2_test:.4f}")

## LightGBM ##
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


lgb_model = lgb.LGBMRegressor(random_state=42)


param_grid = {
    'n_estimators': [300, 500, 700],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'num_leaves': [20, 31, 50],
    'min_child_samples': [10, 20, 30]
}


grid = GridSearchCV(estimator=lgb_model,
                    param_grid=param_grid,
                    cv=5,
                    scoring='neg_root_mean_squared_error',
                    verbose=-1,
                    n_jobs=-1)

grid.fit(X_train, y_train)


best_lgb_model = grid.best_estimator_
print("Best parameters:", grid.best_params_)


y_pred_log = best_lgb_model.predict(X_test)
y_pred_usd = np.exp(y_pred_log)
y_test_usd = np.exp(y_test)

rmse_usd = np.sqrt(mean_squared_error(y_test_usd, y_pred_usd))
mae_usd = np.mean(np.abs(y_test_usd - y_pred_usd))
r2_usd = r2_score(y_test_usd, y_pred_usd)

print(f"Final Model Test RMSE (USD): {rmse_usd:.2f}")
print(f"Final Model Test MAE (USD): {mae_usd:.2f}")
print(f"Final Model Test R² (USD): {r2_usd:.4f}")

## Streamlit APP ##
import joblib
import lightgbm as lgb


final_model = best_lgb_model

joblib.dump(final_model, "final_lgb_model.pkl")

