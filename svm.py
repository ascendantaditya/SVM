import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    df['AEP_MW'] = pd.to_numeric(df['AEP_MW'], errors='coerce')
    df['AEP_MW'].fillna(df['AEP_MW'].mean(), inplace=True)
    return df

# Train the SVR model
def train_svr_model(df):
    NewDataSet = df.resample('D').mean()
    Training_Set = NewDataSet.iloc[:, 0:1][:-60]
    Training_Set = Training_Set.values
    sc = MinMaxScaler(feature_range=(0, 1))
    Train = sc.fit_transform(Training_Set)
    X_Train, Y_Train = [], []
    for i in range(60, Train.shape[0]):
        X_Train.append(Train[i-60:i])
        Y_Train.append(Train[i])
    X_Train = np.array(X_Train)
    Y_Train = np.array(Y_Train)
    X_Train_flattened = X_Train.reshape(X_Train.shape[0], -1)
    svr = SVR()
    param_grid = {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': [1e-3, 1e-4, 'scale', 'auto']}
    grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    grid_search.fit(X_Train_flattened, Y_Train.ravel())
    best_svr = grid_search.best_estimator_
    return best_svr, sc, NewDataSet

# Predict data for a particular year
def predict_for_year(year, model, scaler, dataset, sequence_length=60):
    year_data = dataset[dataset.index.year == year]
    if len(year_data) < sequence_length:
        raise ValueError(f"Not enough data to create sequences for the year {year}.")
    inputs = year_data.values.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    X_test = [inputs[i-sequence_length:i] for i in range(sequence_length, len(inputs))]
    X_test = np.array(X_test)
    X_test_flattened = X_test.reshape(X_test.shape[0], -1)
    predicted_values = model.predict(X_test_flattened).reshape(-1, 1)
    predicted_values = scaler.inverse_transform(predicted_values)
    return predicted_values, year_data.index[sequence_length:]

# Main function to load data and train model
def main():
    df = load_and_preprocess_data("AEP_HOURLY.csv")
    model, scaler, dataset = train_svr_model(df)
    return model, scaler, dataset

if __name__ == "__main__":
    main()