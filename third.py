import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

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

train_features = torch.tensor(X_train).float()
train_targets = torch.tensor(y_train.values).float()
val_features = torch.tensor(X_val).float()
val_targets = torch.tensor(y_val.values).float()

train_dataset = TensorDataset(train_features, train_targets)
val_dataset = TensorDataset(val_features, val_targets)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

fcn_model = nn.Sequential(
    nn.Linear(train_features.shape[1], 128),
    nn.ReLU(),
    nn.Linear(128, 256),
    nn.ELU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ELU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(fcn_model.parameters())

def validation_loss(model, val_loader):
    total_val_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            outputs = model(inputs).squeeze()
            loss = loss_fn(outputs, targets)
            total_val_loss += loss.item() * inputs.size(0)
    return total_val_loss / len(val_loader.dataset)

for epoch in range(500):
    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = fcn_model(inputs).squeeze()
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{10} - Validation loss: {validation_loss(fcn_model, val_loader)}")

fcn_mae = validation_loss(fcn_model, val_loader)
fcn_mae

test_features = torch.tensor(test_data_preprocessed).float()
fcn_model.eval()
with torch.no_grad():
    uenomax_pred = fcn_model(test_features).squeeze().numpy()

submission = test_data[['datetime', 'ru_id']].copy()
submission['uenomax'] = uenomax_pred

submission_pivot = submission.pivot(index='datetime', columns='ru_id', values='uenomax').reset_index()

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

        pred.to_csv('./pth/Q1_submitResult.csv', index=False)
        print("Done: The result has been saved as 'Q1_submitResult.csv'.")
    except Exception as e:
        print("Error:", e)

submitResult(submission_pivot)
