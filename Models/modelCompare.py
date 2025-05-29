import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings

# Suppress LightGBM warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# Load and preprocess data
data_dir = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/Test8/'
all_files = glob.glob(os.path.join(data_dir, '*.csv'))

data_list = []
for file in all_files:
    # Extract coordinates from filename, e.g., (1,1).csv -> x=1, y=1
    filename = os.path.basename(file)
    coords = filename.replace('.csv', '').strip('()').split(',')
    x_coord, y_coord = float(coords[0]), float(coords[1])
    
    # Read CSV for this location
    df = pd.read_csv(file)
    
    # Add coordinates to the dataframe
    df['x_coord'] = x_coord
    df['y_coord'] = y_coord
    data_list.append(df)

# Combine data, preserving location-specific structure
all_data = pd.concat(data_list, ignore_index=True)

pivot_data = all_data.groupby(['x_coord', 'y_coord', 'antenna']).agg({
     'rssi': 'mean',
     'phase_angle': 'mean'
 }).unstack()

# Define features and target
feature_columns = [col for col in pivot_data.columns if col not in ['x_coord', 'y_coord']]
X = pivot_data[feature_columns]
y = pivot_data[['x_coord', 'y_coord']]

# Split data (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models and parameter grids
models_params = {
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'param_grid': {
            'n_estimators': [100, 500],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2']
        }
    },
    'GradientBoosting': {
        'model': MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),
        'param_grid': {
            'estimator__n_estimators': [100, 500],
            'estimator__learning_rate': [0.01, 0.05],
            'estimator__max_depth': [3, 5]
        }
    },
    'SVR': {
        'model': MultiOutputRegressor(SVR(kernel='rbf')),
        'param_grid': {
            'estimator__C': [0.1, 1.0, 10.0],
            'estimator__epsilon': [0.01, 0.1]
        }
    },
    'LightGBM': {
        'model': MultiOutputRegressor(LGBMRegressor(random_state=42, verbose=-1)),
        'param_grid': {
            'estimator__n_estimators': [100, 200],
            'estimator__learning_rate': [0.01, 0.05],
            'estimator__max_depth': [3, 5],
            'estimator__num_leaves': [7, 15],
            'estimator__min_data_in_leaf': [5, 10]
        }
    }
}

# Tune models and store results
results = {}
for name, config in models_params.items():
    print(f"Tuning {name}...")
    grid_search = GridSearchCV(
        config['model'],
        config['param_grid'],
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    mse_avg = np.mean(mse)
    results[name] = {
        'MSE_X': mse[0],
        'MSE_Y': mse[1],
        'MSE_Avg': mse_avg,
        'MAE_X': mae[0],
        'MAE_Y': mae[1],
        'Best_Params': grid_search.best_params_
    }
    print(f"{name} Best Parameters: {grid_search.best_params_}")
    print(f"{name} Test Set MSE (x, y): {mse[0]:.2f}, {mse[1]:.2f} | Average MSE: {mse_avg:.2f} | MAE (x, y): {mae[0]:.2f}, {mae[1]:.2f}\n")

# Find best model
best_model = min(results, key=lambda x: results[x]['MSE_Avg'])
print(f"Best Model: {best_model} with Average MSE: {results[best_model]['MSE_Avg']:.2f}")

# Plot actual vs predicted coordinates for best model
best_model_instance = models_params[best_model]['model']
best_model_instance.fit(X_train, y_train)
y_pred_best = best_model_instance.predict(X_test)

plt.figure(figsize=(8, 8))
plt.scatter(y_test['x_coord'], y_test['y_coord'], c='blue', label='Actual', alpha=0.6)
plt.scatter(y_pred_best[:, 0], y_pred_best[:, 1], c='red', label='Predicted', alpha=0.6)
plt.xlabel('X Coordinate (meters)')
plt.ylabel('Y Coordinate (meters)')
plt.title(f'Actual vs Predicted Coordinates ({best_model})')
plt.legend()
plt.grid(True)
plt.savefig('coordinates_plot.png')
plt.close()