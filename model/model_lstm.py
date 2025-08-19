import pandas as pd
data_train = pd.read_csv(
    r"C:\Users\bhara\OneDrive\Documents\GitHub\Industrial-Maintenance-Predictor\data\train_FD003.txt",
    sep="\s+",
    header=None
)
data_test = pd.read_csv(
    r"C:\Users\bhara\OneDrive\Documents\GitHub\Industrial-Maintenance-Predictor\data\test_FD003.txt",
    sep="\s+",
    header=None
)
cols = (
    ["unit_number", "time_in_cycles", "op_set1", "op_set2", "op_set3"] +
    [f"s{i}" for i in range(1, 22)]
)
df_train=pd.DataFrame(data_train.values, columns=cols)
df_test=pd.DataFrame(data_test.values, columns=cols)

train_df = df_train[df_train['unit_number'] <= 3]
test_df = df_test[df_test['unit_number'] <= 3]
print(train_df.head(10))
print(test_df.head(10))