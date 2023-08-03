import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def submitResult(pred):
    try:
        label = pd.read_csv('./Q1_data/Q1_label_sample.csv')
        if pred.columns.equals(label.columns):
            print("Check: Column names and order are the same.")
        else:
            print(
                f"Warning: Column names and order are not the same.\n- Predicted dataframe column names: {pred.columns}\n- Label dataframe column names: {label.columns}")
            return

        if (label['datetime'] == pred['datetime']).all():
            print("Check: 'datetime' order and number of samples match.")
        else:
            print("Warning: 'datetime' of the test set and model prediction do not match.")
            return

        pred.to_csv('./Q1_submitResult.csv', index=False)
        print("Done: The result has been saved as 'Q1_submitResult.csv'.")
    except Exception as e:
        print("Error:", e)

train_data = pd.read_csv('./Q1_data/Q1_train.csv')
test_data = pd.read_csv('./Q1_data/Q1_test.csv')

test_data = test_data.drop(columns=['Unnamed: 0'])

numeric_features = train_data.drop(columns=['uenomax']).select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['ru_id']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

train_data_preprocessed = preprocessor.fit_transform(train_data.drop(columns=['uenomax']))
test_data_preprocessed = preprocessor.transform(test_data)
X_train, X_val, y_train, y_val = train_test_split(train_data_preprocessed, train_data['uenomax'], test_size=0.2, random_state=42)
dt_model = DecisionTreeRegressor(random_state=42)

dt_model.fit(X_train, y_train)
y_val_pred = dt_model.predict(X_val)

mae = mean_absolute_error(y_val, y_val_pred)
print(f"Mean Absolute Error on Validation Set: {mae}")

uenomax_pred = dt_model.predict(test_data_preprocessed)

submission = test_data[['datetime', 'ru_id']].copy()
submission['uenomax'] = uenomax_pred
#submission.to_csv('./Q1_label_sample.csv', index=False)

submission_pivot = submission.pivot(index='datetime', columns='ru_id', values='uenomax').reset_index()
submission_pivot.head()

submitResult(submission_pivot)

#visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_val_pred, alpha=0.3)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=3, color='red')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values for uenomax on Validation Set')
plt.show()
