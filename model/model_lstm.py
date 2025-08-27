import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ---------------------------
# STEP 1: LOAD DATASETS
# ---------------------------
train_path = r"C:\Users\bhara\OneDrive\Documents\GitHub\Industrial-Maintenance-Predictor\data\train_FD003.txt"
test_path = r"C:\Users\bhara\OneDrive\Documents\GitHub\Industrial-Maintenance-Predictor\data\test_FD003.txt"
rul_path = r"C:\Users\bhara\OneDrive\Documents\GitHub\Industrial-Maintenance-Predictor\data\RUL_FD003.txt"

data_train = pd.read_csv(train_path, sep="\s+", header=None)
data_test = pd.read_csv(test_path, sep="\s+", header=None)
data_rul = pd.read_csv(rul_path, sep="\s+", header=None)

cols = ["unit_number", "time_in_cycles", "op_set1", "op_set2", "op_set3"] + [f"s{i}" for i in range(1, 22)]
df_train = pd.DataFrame(data_train.values, columns=cols)
df_test = pd.DataFrame(data_test.values, columns=cols)

# ---------------------------
# STEP 2: ADD RUL TO TRAIN DATA
# ---------------------------
def add_rul(df):
    rul = []
    for unit in df['unit_number'].unique():
        unit_df = df[df['unit_number'] == unit]
        max_cycle = unit_df['time_in_cycles'].max()
        rul.extend(max_cycle - unit_df['time_in_cycles'])
    df['RUL'] = rul
    return df

df_train = add_rul(df_train)

# ---------------------------
# STEP 3: SCALE FEATURES
# ---------------------------
feature_cols = ["op_set1", "op_set2", "op_set3"] + [f"s{i}" for i in range(1, 22)]
scaler = MinMaxScaler()
df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
df_test[feature_cols] = scaler.transform(df_test[feature_cols])

# ---------------------------
# STEP 4: GENERATE SEQUENCES FOR LSTM
# ---------------------------
SEQ_LENGTH = 30

def gen_sequences(df, seq_length, feature_cols, label_col=None):
    sequences, labels = [], []
    for unit in df['unit_number'].unique():
        unit_df = df[df['unit_number'] == unit].sort_values('time_in_cycles')
        feature_array = unit_df[feature_cols].values
        rul_array = unit_df[label_col].values if label_col else None
        
        for i in range(len(unit_df) - seq_length + 1):
            sequences.append(feature_array[i:i+seq_length])
            if label_col:
                labels.append(rul_array[i+seq_length-1])
    return np.array(sequences), np.array(labels) if label_col else np.array(sequences)

X_train, y_train = gen_sequences(df_train, SEQ_LENGTH, feature_cols, label_col='RUL')

print("Training Sequences Shape:", X_train.shape)
print("Training Labels Shape:", y_train.shape)

# ---------------------------
# STEP 5: BUILD LSTM MODEL
# ---------------------------
model = Sequential([
    LSTM(128, input_shape=(SEQ_LENGTH, len(feature_cols)), return_sequences=True),
    Dropout(0.2),
    LSTM(64, input_shape=(SEQ_LENGTH, len(feature_cols)), return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()

# ---------------------------
# STEP 6: TRAIN MODEL
# ---------------------------
history = model.fit(X_train, y_train, epochs=70, batch_size=32, validation_split=0.2, verbose=1)

# ---------------------------
# STEP 7: VISUALIZE TRAINING LOSS
# ---------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training & Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()

# ---------------------------
# STEP 8: PREPARE TEST SEQUENCES WITH TRUE RUL
# ---------------------------
def prepare_test_sequences(df_test, rul_df, seq_length, feature_cols):
    test_sequences = []
    true_rul = []
    for i, unit in enumerate(df_test['unit_number'].unique()):
        unit_df = df_test[df_test['unit_number'] == unit].sort_values('time_in_cycles')
        features = unit_df[feature_cols].values
        if len(features) >= seq_length:
            seq = features[-seq_length:]
        else:
            pad = np.zeros((seq_length - len(features), len(feature_cols)))
            seq = np.vstack((pad, features))
        test_sequences.append(seq)
        true_rul.append(rul_df.iloc[i, 0])
    return np.array(test_sequences), np.array(true_rul)

X_test, y_true = prepare_test_sequences(df_test, data_rul, SEQ_LENGTH, feature_cols)

# ---------------------------
# STEP 9: PREDICT & EVALUATE
# ---------------------------
y_pred = model.predict(X_test).flatten()

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"Test RMSE: {rmse:.2f}")

# ---------------------------
# STEP 10: VISUALIZE PREDICTIONS VS ACTUAL
# ---------------------------
plt.figure(figsize=(8,5))
plt.plot(y_true, label='Actual RUL')
plt.plot(y_pred, label='Predicted RUL')
plt.title("Predicted vs Actual RUL")
plt.xlabel("Engine Number")
plt.ylabel("RUL (cycles)")
plt.legend()
plt.show()
