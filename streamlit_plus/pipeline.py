import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

try:
    import lightgbm as lgb
except Exception:
    lgb = None

# ---------------------------
# 1. Raw-data cleaning helpers
# ---------------------------
def _clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'Power Reserve' in df.columns:
        df['Power Reserve'] = df['Power Reserve'].astype(str).str.replace('hours', '', regex=False).str.strip()
        df['Power Reserve'] = pd.to_numeric(df['Power Reserve'], errors='coerce')

    if 'Water Resistance' in df.columns:
        df['Water Resistance'] = df['Water Resistance'].astype(str).str.replace('meters', '', regex=False).str.replace('metre', '', regex=False).str.strip()
        df['Water Resistance'] = pd.to_numeric(df['Water Resistance'], errors='coerce')

    if 'Complications' in df.columns:
        df['Complications'] = df['Complications'].fillna('None').astype(str).str.replace(r'\s+', ' ',
                                                                                         regex=True).str.strip()

    if 'Price (USD)' in df.columns:
        df['Price (USD)'] = df['Price (USD)'].astype(str).str.replace(',', '').str.strip()
        df['Price (USD)'] = pd.to_numeric(df['Price (USD)'], errors='coerce')

    return df


def _fill_power_reserve_logic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def fill_power_reserve(row):
        if pd.isna(row['Power Reserve']):
            mt = row.get('Movement Type', None)
            if mt == 'Quartz':
                return 0
            elif mt == 'Eco-Drive':
                return 999
            elif mt in ('Automatic', 'Manual'):
                return 48
            else:
                return np.nan
        else:
            return row['Power Reserve']

    if 'Power Reserve' in df.columns and 'Movement Type' in df.columns:
        df['Power Reserve'] = df.apply(fill_power_reserve, axis=1)
    return df


def _fill_price_by_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Price (USD)' in df.columns:
        return df

    df['Price_filled'] = df['Price (USD)']
    group_cols = ['Brand', 'Model', 'Case Material', 'Strap Material', 'Power Reserve']

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
    return df


def _clip_power_reserve(df: pd.DataFrame, upper=120) -> pd.DataFrame:
    """Clip the Eco-Drive 999 values to a sane upper bound (default 120)."""
    df = df.copy()
    if 'Power Reserve' in df.columns:
        df['Power Reserve'] = df['Power Reserve'].clip(upper=upper)
    return df


# ---------------------------
# 2. Full raw preprocess function (applied before pipeline fitting)
# ---------------------------
def preprocess_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = _clean_string_columns(df)

    df = _fill_power_reserve_logic(df)

    if 'Price (USD)' in df.columns:
        df = _fill_price_by_groups(df)

    df = _clip_power_reserve(df, upper=120)

    num_cols = ['Water Resistance', 'Case Diameter (mm)', 'Case Thickness (mm)',
                'Band Width (mm)', 'Power Reserve', 'Price (USD)']

    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    for c in num_cols:
        if c in df.columns:
            if df[c].isnull().any():
                df[c].fillna(df[c].median(), inplace=True)

    if 'Complications' in df.columns:
        df['Complications'] = df['Complications'].fillna('None').astype(str)

    return df


# ---------------------------
# 3. Preprocessor builder for features
# ---------------------------
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OrdinalEncoder

def build_preprocessor(numeric_features=None, categorical_features=None):
    if numeric_features is None:
        numeric_features = ["Water Resistance", "Case Diameter (mm)", "Case Thickness (mm)",
                            "Band Width (mm)", "Power Reserve"]

    if categorical_features is None:
        categorical_features = ['Brand', 'Model', 'Case Material', 'Strap Material',
                                'Movement Type', 'Dial Color', 'Crystal Material', 'Complications']

    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
    ])

    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numeric_features),
            ('cat', cat_pipeline, categorical_features)
        ],
        remainder='drop',
        sparse_threshold=0
    )

    return preprocessor


# ---------------------------
# 4) Build pipeline with estimator
# ---------------------------
def build_pipeline(estimator=None):
    preprocessor = build_preprocessor()

    if estimator is None:
        if lgb is None:
            raise RuntimeError("lightgbm not available; pass an sklearn-compatible estimator to build_pipeline()")
        estimator = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            random_state=42
        )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('estimator', estimator)
    ])
    return pipeline


# ---------------------------
# 5) Training wrapper
# ---------------------------
def train_pipeline(csv_path,
                   pipeline_save_path="final_pipeline.pkl",
                   target_col='Price (USD)',
                   test_size=0.2,
                   random_state=42,
                   do_gridsearch=False):

    # 1) load
    df_raw = pd.read_csv(csv_path)
    df = preprocess_raw_df(df_raw)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found after preprocessing")

    # 2) prepare X,y (y is log1p)
    y = np.log1p(df[target_col].values)
    X = df.drop(columns=[target_col])

    # 3) train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 4) build pipeline
    pipeline = build_pipeline()

    if do_gridsearch:
        param_grid = {
            'estimator__n_estimators': [200, 500],
            'estimator__learning_rate': [0.01, 0.05],
            'estimator__max_depth': [4, 6],
        }
        grid = GridSearchCV(pipeline, param_grid=param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=-1)
        grid.fit(X_train, y_train)
        pipeline = grid.best_estimator_
    else:
        pipeline.fit(X_train, y_train)

    # 5) predictions on test (log scale)
    y_pred_log = pipeline.predict(X_test)

    # convert back to USD
    y_test_usd = np.expm1(y_test)
    y_pred_usd = np.expm1(y_pred_log)

    # metrics in USD scale
    rmse_usd = np.sqrt(mean_squared_error(y_test_usd, y_pred_usd))
    mae_usd = np.mean(np.abs(y_test_usd - y_pred_usd))
    r2_usd = r2_score(y_test_usd, y_pred_usd)

    print("=== Evaluation on hold-out test (USD scale) ===")
    print(f"RMSE (USD): {rmse_usd:.2f}")
    print(f"MAE  (USD): {mae_usd:.2f}")
    print(f"R2   (USD): {r2_usd:.4f}")

    # 6) save pipeline
    joblib.dump(pipeline, SAVE_PATH)
    print(f"Saved pipeline to: {pipeline_save_path}")

    return pipeline, X_test, y_test_usd, y_pred_usd

# ---------------------------
# 6) Helpers for serving / Streamlit
# ---------------------------
def load_pipeline(path="streamlit_plus/models/final_lgb_model.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline file not found: {path}")
    return joblib.load(path)

def predict_from_raw_inputs(pipeline, input_dict):
    input_df = pd.DataFrame([input_dict])
    input_df = _clean_string_columns(input_df)
    input_df = _fill_power_reserve_logic(input_df)
    input_df = _clip_power_reserve(input_df, upper=120)

    pred_log = pipeline.predict(input_df)
    pred_usd = np.expm1(pred_log)
    return float(pred_usd[0])

# ---------------------------
# 7) Script entrypoint for training
# ---------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_PATH = os.path.join(BASE_DIR, "models", "final_lgb_model.pkl")
    CSV_PATH = os.path.join(BASE_DIR, "Luxury watch.csv")
    print("Training pipeline from:", CSV_PATH)
    pipeline, X_test, y_test_usd, y_pred_usd = train_pipeline(CSV_PATH, pipeline_save_path=SAVE_PATH, do_gridsearch=False)