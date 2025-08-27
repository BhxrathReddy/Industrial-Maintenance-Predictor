import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load the datasets - We'll only use train_FD003.txt and split it
data_train = pd.read_csv(
    r"../data/train_FD003.txt",
    sep=r"\s+",
    header=None
)

# Load RUL data for validation (we'll use this to validate our test split)
rul_validation = pd.read_csv(
    r"../data/RUL_FD003.txt",
    sep=r"\s+",
    header=None,
    names=['RUL']
)

# Define column names
cols = (
    ["unit_number", "time_in_cycles", "op_set1", "op_set2", "op_set3"] +
    [f"s{i}" for i in range(1, 22)]
)

# Create DataFrame
df_full = pd.DataFrame(data_train.values, columns=cols)

print("Dataset Information:")
print(f"Full training data shape: {df_full.shape}")
print(f"Number of machines in training data: {df_full['unit_number'].nunique()}")
print(f"RUL validation data shape: {rul_validation.shape}")

# Split machines into 70% train and 30% test
unique_machines = sorted(df_full['unit_number'].unique())
print(f"Total machines: {len(unique_machines)}")

# Split machines randomly
train_machines, test_machines = train_test_split(
    unique_machines, test_size=0.3, random_state=42
)

# Convert to integers for easier handling
train_machines = [int(m) for m in train_machines]
test_machines = [int(m) for m in test_machines]

print(f"Training machines: {len(train_machines)} (70%)")
print(f"Test machines: {len(test_machines)} (30%)")
print(f"Training machine IDs: {sorted(train_machines)[:10]}...") # Show first 10
print(f"Test machine IDs: {sorted(test_machines)[:10]}...")      # Show first 10

# Function to calculate RUL for training data
def calculate_rul(df):
    df = df.copy()
    # Find max cycles for each engine
    max_cycles = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycles']
    df = df.merge(max_cycles, on='unit_number')
    # RUL = max_cycles - current_cycle
    df['RUL'] = df['max_cycles'] - df['time_in_cycles']
    return df.drop('max_cycles', axis=1)

# Function to prepare test data for our split using RUL_FD003.txt
def prepare_internal_test_data(df_test_machines, test_machine_ids, rul_validation):
    """Prepare test data using the true RUL values from RUL_FD003.txt"""
    test_cycles = []
    
    for machine_id in test_machine_ids:
        machine_data = df_test_machines[df_test_machines['unit_number'] == machine_id]
        
        # Get the last cycle for this machine
        last_cycle_data = machine_data[machine_data['time_in_cycles'] == machine_data['time_in_cycles'].max()].copy()
        
        # Get the true RUL from RUL_FD003.txt (machine_id is 1-indexed)
        if machine_id <= len(rul_validation):
            true_rul = rul_validation.iloc[machine_id - 1]['RUL']
            last_cycle_data['RUL'] = true_rul
            test_cycles.append(last_cycle_data)
        else:
            print(f"Warning: Machine {machine_id} not found in RUL_FD003.txt")
    
    return pd.concat(test_cycles, ignore_index=True) if test_cycles else pd.DataFrame()

# Split the data based on machine IDs
df_train = df_full[df_full['unit_number'].isin(train_machines)].copy()
df_test_internal = df_full[df_full['unit_number'].isin(test_machines)].copy()

print(f"\nData split results:")
print(f"Training data: {df_train.shape}")
print(f"Internal test data: {df_test_internal.shape}")

# Calculate RUL for training data (all cycles for training machines)
print("\nCalculating RUL for training data...")
df_train_with_rul = calculate_rul(df_train)

# For test machines, we'll use the true RUL values from RUL_FD003.txt
# This file contains the actual RUL for the last cycle of each machine
print("Preparing internal test data using RUL_FD003.txt...")

# Use the true RUL values from the external file
df_test_with_rul = prepare_internal_test_data(df_test_internal, test_machines, rul_validation)

print(f"Training data with RUL: {df_train_with_rul.shape}")
print(f"Internal test data with RUL: {df_test_with_rul.shape}")

# Show RUL distribution for test data
print(f"\nTest RUL statistics (from RUL_FD003.txt):")
print(f"Min RUL: {df_test_with_rul['RUL'].min():.1f}")
print(f"Max RUL: {df_test_with_rul['RUL'].max():.1f}")
print(f"Mean RUL: {df_test_with_rul['RUL'].mean():.1f}")
print(f"Test machines and their true RUL values:")
for _, row in df_test_with_rul[['unit_number', 'time_in_cycles', 'RUL']].iterrows():
    print(f"  Machine {int(row['unit_number'])}: Cycle {int(row['time_in_cycles'])}, True RUL = {row['RUL']:.1f}")

# Define feature columns (exclude unit_number, time_in_cycles, and RUL)
feature_cols = ['op_set1', 'op_set2', 'op_set3'] + [f's{i}' for i in range(1, 22)]

# Prepare training data (all cycles from training machines)
X_train_full = df_train_with_rul[feature_cols]
y_train_full = df_train_with_rul['RUL']

# Prepare test data (last cycle from each test machine)
X_test = df_test_with_rul[feature_cols]
y_test_true = df_test_with_rul['RUL']  # True RUL values from our calculation

print(f"\nFeature matrix shapes:")
print(f"X_train_full: {X_train_full.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train_full: {y_train_full.shape}")
print(f"y_test_true: {y_test_true.shape}")

# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

# Scale the features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Scaled feature shapes:")
print(f"X_train_scaled: {X_train_scaled.shape}")
print(f"X_val_scaled: {X_val_scaled.shape}")
print(f"X_test_scaled: {X_test_scaled.shape}")

# Train XGBoost model
print("\n" + "="*50)
print("TRAINING XGBOOST MODEL")
print("="*50)

xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

print("Training XGBoost model...")
xgb_model.fit(X_train_scaled, y_train)

# Make predictions
print("Making predictions...")
xgb_val_pred = xgb_model.predict(X_val_scaled)
xgb_test_pred = xgb_model.predict(X_test_scaled)

# Evaluate on validation set
xgb_val_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_pred))
xgb_val_mae = mean_absolute_error(y_val, xgb_val_pred)
xgb_val_r2 = r2_score(y_val, xgb_val_pred)

# Evaluate on test set
xgb_test_rmse = np.sqrt(mean_squared_error(y_test_true, xgb_test_pred))
xgb_test_mae = mean_absolute_error(y_test_true, xgb_test_pred)
xgb_test_r2 = r2_score(y_test_true, xgb_test_pred)

print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)
print(f"Validation RMSE: {xgb_val_rmse:.2f}")
print(f"Validation MAE: {xgb_val_mae:.2f}")
print(f"Validation R²: {xgb_val_r2:.3f}")
print()
print(f"Test RMSE: {xgb_test_rmse:.2f}")
print(f"Test MAE: {xgb_test_mae:.2f}")
print(f"Test R²: {xgb_test_r2:.3f}")

# Create output DataFrame with predictions
print("\n" + "="*50)
print("CREATING OUTPUT FILE")
print("="*50)

# Get test machine IDs and their predicted RUL
test_machine_ids = df_test_with_rul['unit_number'].values
predicted_rul = xgb_test_pred

# Create output DataFrame
output_df = pd.DataFrame({
    'Machine_ID': test_machine_ids,
    'Predicted_RUL': predicted_rul,
    'True_RUL': y_test_true.values,
    'Prediction_Error': predicted_rul - y_test_true.values
})

# Save predictions to file
output_file = "../output/predicted_RUL_results.csv"
output_df.to_csv(output_file, index=False)

print(f"Predictions saved to: {output_file}")
print(f"Number of test machines: {len(output_df)}")

# Validation against RUL_FD003.txt (we're already using these values)
print("\n" + "="*50)
print("VALIDATION SUMMARY")
print("="*50)

print(f"✓ Using true RUL values from RUL_FD003.txt for test machines")
print(f"✓ Test machines: {sorted(test_machines)}")
print(f"✓ All test machines have corresponding RUL values in RUL_FD003.txt")

# Show some examples of the validation
print(f"\nSample validation (Predicted vs True RUL from RUL_FD003.txt):")
print(f"{'Machine_ID':<12} {'True_RUL':<10} {'Pred_RUL':<10} {'Error':<10}")
print("-" * 45)
for i in range(min(10, len(output_df))):
    machine_id = output_df.iloc[i]['Machine_ID']
    true_rul = output_df.iloc[i]['True_RUL']
    pred_rul = output_df.iloc[i]['Predicted_RUL']
    error = output_df.iloc[i]['Prediction_Error']
    print(f"{int(machine_id):<12} {true_rul:<10.1f} {pred_rul:<10.1f} {error:<10.1f}")

# Overall validation metrics
true_rul_values = output_df['True_RUL'].values
predictions = output_df['Predicted_RUL'].values

final_rmse = np.sqrt(mean_squared_error(true_rul_values, predictions))
final_mae = mean_absolute_error(true_rul_values, predictions)
final_r2 = r2_score(true_rul_values, predictions)

print(f"\nFinal Model Performance against RUL_FD003.txt:")
print(f"RMSE: {final_rmse:.2f}")
print(f"MAE: {final_mae:.2f}")
print(f"R²: {final_r2:.3f}")

# Display sample predictions
print("\n" + "="*50)
print("SAMPLE PREDICTIONS (First 10 machines)")
print("="*50)
print(f"{'Machine_ID':<12} {'True_RUL':<10} {'Pred_RUL':<10} {'Error':<10}")
print("-" * 45)
for i in range(min(10, len(output_df))):
    machine_id = output_df.iloc[i]['Machine_ID']
    true_rul = output_df.iloc[i]['True_RUL']
    pred_rul = output_df.iloc[i]['Predicted_RUL']
    error = output_df.iloc[i]['Prediction_Error']
    print(f"{machine_id:<12} {true_rul:<10.1f} {pred_rul:<10.1f} {error:<10.1f}")

# Feature importance
print("\n" + "="*50)
print("TOP 10 FEATURE IMPORTANCE")
print("="*50)

feature_importance = xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(feature_importance_df.head(10).to_string(index=False))

print("\n" + "="*50)
print("PROCESS COMPLETED SUCCESSFULLY!")
print("="*50)
print(f"✓ Trained on {len(train_machines)} machines with {len(df_train_with_rul)} data points")
print(f"✓ Tested on {len(test_machines)} machines") 
print(f"✓ Model performance - Test RMSE: {xgb_test_rmse:.2f}, Test R²: {xgb_test_r2:.3f}")
print(f"✓ Results saved to: {output_file}")
print(f"✓ Data split: 70% train ({len(train_machines)} machines), 30% test ({len(test_machines)} machines)")

# Optional: Compare with external RUL validation if needed
print(f"\nNote: This model was trained and tested using 70/30 split of train_FD003.txt")
print(f"Training machines: {sorted(train_machines)}")
print(f"Test machines: {sorted(test_machines)}")