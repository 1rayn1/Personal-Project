import pandas as pd
import os
import joblib
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# Paths

file_path = 'car data.csv'
model_file = 'sgd_car_model.pkl'
scaler_file = 'scaler.pkl'

def load_and_clean_data(path):
    data = pd.read_csv(path, index_col=0)
    data = data.dropna(subset=['Present_Price', 'Year', 'Selling_Price', 'Kms_Driven'])
    return data

def get_model(train_X, train_y):
    if os.path.exists(model_file):
        print("Loading saved incremental model...")
        model = joblib.load(model_file)
    else:
        print("No saved model found. Creating new incremental model...")
        model = SGDRegressor(
            max_iter=1,
            learning_rate='invscaling',
            eta0=0.01,
            warm_start=True,
            random_state=1
        )
    model.partial_fit(train_X, train_y)
    joblib.dump(model, model_file)
    print("Incremental model saved.")
    return model

def get_scaler(train_X):
    if os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
    else:
        scaler = StandardScaler()
        scaler.fit(train_X)
        joblib.dump(scaler, scaler_file)
    return scaler

def predict_price(model, scaler):
    while True:
        try:
            print("\nEnter car details to predict Present Price (or type 'exit' to quit):")
            year = input("Year: ")
            if year.lower() == 'exit':
                break
            selling_price = input("Selling Price: ")
            kms_driven = input("Kms Driven: ")

            input_data = [year, selling_price, kms_driven]
            if not all(x.replace('.', '', 1).isdigit() for x in input_data):
                print("Please enter valid numeric values.")
                continue

            input_data = [[float(year), float(selling_price), float(kms_driven)]]
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            print(f"Predicted Present Price: ₹{prediction[0]:.2f}")
        except Exception as pred_error:
            print("Prediction error:", pred_error)

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

def missing_value_removal(score_dataset, X_train, X_valid, y_train, y_valid):

    cols_with_missing = [col for col in X_train.columns
                        if X_train[col].isnull().any()]

    reduced_X_train = X_train.drop(cols_with_missing,axis = 1)
    reduced_X_valid = X_valid.drop(cols_with_missing,axis = 1)

    method1 = (score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

    my_imputer = SimpleImputer()

    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns

    method2 = (score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

    if method1 < method2:
        return method1
    elif method2 < method1:
        return method2
    else:
        return "Both methods give the same MAE"

def main():
    data = None
    scaler = None
    model = None
    while True:
        print("\nSelect an action:")
        print("1. Reset data and model")
        print("2. Train and evaluate model")
        print("3. Predict new data")
        print("4. Exit program")
        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            if os.path.exists(model_file):
                os.remove(model_file)
                print("Deleted saved model.")
            if os.path.exists(scaler_file):
                os.remove(scaler_file)
                print("Deleted saved scaler.")
            data = None
            scaler = None
            model = None

        elif choice == "2":
            try:
                data = load_and_clean_data(file_path)
                y = data['Present_Price'].values
                features = ['Year', 'Selling_Price', 'Kms_Driven']
                X = data[features]

                train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

                best_mae = missing_value_removal(score_dataset, train_X, val_X, train_y, val_y)
                print(f"Best MAE from outlier/missing value handling: {best_mae}")

                scaler = get_scaler(train_X.values)
                train_X_scaled = scaler.transform(train_X.values)
                val_X_scaled = scaler.transform(val_X.values)

                model = get_model(train_X_scaled, train_y)
                preds = model.predict(val_X_scaled)
                mae = mean_absolute_error(val_y, preds)
                print(f"MAE after incremental training: {mae:.2f}")
            except Exception as e:
                print("Error during training:", e)

        elif choice == "3":
            if model is None or scaler is None:
                print("Model and scaler not trained yet. Please train the model first (option 2).")
            else:
                predict_price(model, scaler)

        elif choice == "4":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()