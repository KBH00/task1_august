import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def preprocess_data(df):
    df = pd.get_dummies(df, columns=['ru_id'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

def extract_labels(df):
    if 'uenomax' in df.columns:
        labels = df[['datetime', 'ru_id', 'uenomax']]
    else:
        labels = None
    return labels

class MyModel:
    def __init__(self):
        self.models = {}

    def train(self, x_train, y_train):
        for base_station in y_train['ru_id'].unique():
            if f"ru_id_{base_station}" in x_train.columns:
                x = x_train[x_train[f"ru_id_{base_station}"] == 1].drop(columns=['datetime', f"ru_id_{base_station}"])
                y = y_train[y_train['ru_id'] == base_station]['uenomax']
                x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x, y, test_size=0.2,
                                                                                          random_state=42)
                model = LinearRegression()
                model.fit(x_train_split, y_train_split)

                y_pred = model.predict(x_val_split)
                rmse = np.sqrt(mean_squared_error(y_val_split, y_pred))
                print(f"Base Station: {base_station}, Validation RMSE: {rmse}")
                self.models[base_station] = model

    def predict(self, x_test):
        y_pred = pd.DataFrame(columns=['datetime', 'ru_id', 'uenomax'])

        for base_station in y_train['ru_id'].unique():
            if f"ru_id_{base_station}" in x_test.columns and base_station in self.models:
                x = x_test[x_test[f"ru_id_{base_station}"] == 1].drop(columns=['datetime', f"ru_id_{base_station}"])
                model = self.models[base_station]
                predictions = model.predict(x)
            else:
                x = x_test.reset_index()
                predictions = [0] * len(x)

            # Use a temporary DataFrame to store the predictions for this base station
            temp_pred = pd.DataFrame({
                'datetime': x['datetime'],
                'ru_id': base_station,
                'uenomax': predictions
            })

            # Only append predictions for datetimes that are not already in y_pred
            y_pred = y_pred.append(temp_pred[~temp_pred['datetime'].isin(y_pred['datetime'])], ignore_index=True)

        return y_pred

train_data = pd.read_csv("./Q1_Data/Q1_train.csv")
test_data = pd.read_csv("./Q1_data/Q1_test.csv")
print(train_data['ru_id'].unique())

x_train = preprocess_data(train_data)
x_test = preprocess_data(test_data)

y_train = extract_labels(train_data)
y_train.loc[:, 'datetime'] = pd.to_datetime(y_train['datetime'])
y_train.set_index('datetime', inplace=True)

x_train.reset_index(inplace=True)

model = MyModel()
model.train(x_train, y_train)

x_test.reset_index(inplace=True)

y_pred = model.predict(x_test)
print(y_pred)
y_pred = y_pred.groupby(['datetime', 'ru_id'])['uenomax'].mean().reset_index()
y_pred = y_pred.pivot(index='datetime', columns='ru_id', values='uenomax').reset_index()
y_pred.columns.name = None
print(y_pred)


def save_predictions(predictions, filename):
    predictions.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")


save_predictions(y_pred, "Q1_submitResult.csv")
