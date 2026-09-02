from forge.data.sample import load_sample

df = load_sample()
print(f"Loaded sample recording: {df.shape}")
print(df.head(5))
print(df.columns.to_list())

# extend dataframe
sensor_columns = df.columns.difference(["datetime", "anomaly", "changepoint"])
label_columns = df.columns.intersection(["anomaly", "changepoint"])
print(f"Sensor columns: {sensor_columns.to_list()}")
print(f"Label columns: {label_columns.to_list()}")
print(df["anomaly"].value_counts())
