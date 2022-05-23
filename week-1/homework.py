import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Q1. Downloading the data
jan_df = pd.read_parquet("fhv_tripdata_2021-01.parquet")
print(f"records: {jan_df.shape[0]}")

# Q2. Computing duration and data preparation
jan_df['duration'] = jan_df["dropOff_datetime"] - jan_df["pickup_datetime"]
jan_df.duration = jan_df.duration.apply(lambda td: td.total_seconds() / 60)
print(f"avg trip duration: {jan_df.duration.mean()}")
init_shape = jan_df.shape[0]
jan_df = jan_df[(jan_df.duration >= 1) & (jan_df.duration <= 60)]
print(f"drops: {init_shape - jan_df.shape[0]}")

# Q3. Missing values
print(f"nan perc: {round(jan_df.isna().sum()['PUlocationID'] / jan_df['PUlocationID'].shape[0], 2) * 100} %")
jan_df['PUlocationID'].fillna(-1, inplace=True)

# Q4. One-hot encoding
train_cols = ["PUlocationID", "DOlocationID"]
jan_df[train_cols] = jan_df[train_cols].astype(str)
train_dicts = jan_df[train_cols].to_dict(orient='records')
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)
print(f"Dimensionality: {X_train.shape[1]}")

# Q5. Training a model
target = 'duration'
y_train = jan_df[target].values
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_train)
print(f"RMSE train: {mean_squared_error(y_train, y_pred, squared=False)}")

# Q6. Evaluating the model
feb_df = pd.read_parquet("fhv_tripdata_2021-02.parquet")
feb_df['duration'] = feb_df["dropOff_datetime"] - feb_df["pickup_datetime"]
feb_df.duration = feb_df.duration.apply(lambda td: td.total_seconds() / 60)
feb_df = feb_df[(feb_df.duration >= 1) & (feb_df.duration <= 60)]
feb_df['PUlocationID'].fillna(-1, inplace=True)
feb_df[train_cols] = feb_df[train_cols].astype(str)
val_dicts = feb_df[train_cols].to_dict(orient='records')
X_val = dv.transform(val_dicts)
y_val = feb_df[target].values
y_pred = lr.predict(X_val)

print(f"RMSE val: {mean_squared_error(y_val, y_pred, squared=False)}")

