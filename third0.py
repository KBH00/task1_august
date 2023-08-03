import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error

# Load datasets
train = pd.read_csv('./Q1_data/Q1_train.csv')
test = pd.read_csv('./Q1_data/Q1_test.csv')
label_sample = pd.read_csv('./Q1_data/Q1_label_sample.csv')

# Define numeric and categorical transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Identify numeric and categorical columns
numeric_features = train.drop(columns=['uenomax']).select_dtypes(include=['int64', 'float64']).columns
categorical_features = ['ru_id']

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Preprocess train_data
X_train = train.drop(columns=['uenomax'])
y_train = train['uenomax']
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Train a decision tree model with hyperparameter tuning
model = DecisionTreeRegressor(random_state=42)
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}
cv = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error')
cv.fit(X_train_preprocessed, y_train)

# Print best parameters and score
print(cv.best_params_, -cv.best_score_)

# Preprocess test data
X_test_preprocessed = preprocessor.transform(test)

# Generate predictions
uenomax_pred = cv.predict(X_test_preprocessed)

# Prepare the submission dataframe
submission = test[['datetime', 'ru_id']].copy()
submission['uenomax'] = uenomax_pred
submission_pivot = submission.pivot(index='datetime', columns='ru_id', values='uenomax').reset_index()

# Function to check the submission format
def check_submission_format(pred, label_sample):
    try:
        if pred.columns.equals(label_sample.columns):
            print("Check: Column names and order are the same.")
        else:
            print(
                f"Warning: Column names and order are not the same.\n"
                f"- Predicted dataframe column names: {pred.columns}\n"
                f"- Label dataframe column names: {label_sample.columns}"
            )
            return

        if (label_sample['datetime'] == pred['datetime']).all():
            print("Check: 'datetime' order and number of samples match.")
        else:
            print("Warning: 'datetime' of the test set and model prediction do not match.")
            return
    except Exception as e:
        print("Error:", e)

# Check the submission format
check_submission_format(submission_pivot, label_sample)

# Show the submission dataframe
print(submission_pivot.head())
