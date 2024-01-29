import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load CSV file using pandas
csv_path1 = 'celestial_data.csv'
df1 = pd.read_csv(csv_path1)

csv_path2 = 'test.csv'
df2=pd.read_csv(csv_path2)

feature_columns=['camcol', 'field', 'rowv', 'colv', 'u', 'g', 'r', 'i', 'z', 'psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z', 'petroRad_u', 'petroRad_g', 'petroRad_r', 'petroRad_i', 'petroRad_z', 'expRad_u', 'expRad_g', 'expRad_r', 'expRad_i', 'expRad_z', 'q_u', 'q_g', 'q_r', 'q_i', 'q_z','ra', 'dec', 'b', 'l', 'type']

df1=df1[feature_columns]
df2=df2[feature_columns]
label_mapping = {'star': 0, 'galaxy': 1}
df1['type'] = df1['type'].map(label_mapping)
df2['type'] = df2['type'].map(label_mapping)

correlation_matrix = df1.corr()
print("Correlation Matrix:")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(correlation_matrix)

columns_already_dropped = set()
columns_to_drop = set()
for i in range(len(feature_columns)):
    if feature_columns[i] in columns_already_dropped:
            # Skip columns that are already marked for dropping
        continue
    if abs(correlation_matrix.iloc[i,33])<0.1:
        col_to_drop=feature_columns[i]
        print("will drop",i,"bc of type")
        columns_to_drop.add(col_to_drop)
        columns_already_dropped.add(col_to_drop)
        continue
    for j in range(i + 1, len(feature_columns)):
        if feature_columns[j]=='psfMag_z':
            continue
        if abs(correlation_matrix.iloc[i, j]) > 0.75:
            print("will drop", j, "bc of ",i)
            col_to_drop = feature_columns[j]
            columns_to_drop.add(col_to_drop)
            columns_already_dropped.add(col_to_drop)

df_processed1 = df1.drop(columns=columns_to_drop)
df_processed2 = df2.drop(columns=columns_to_drop)
print(df_processed1.shape)
print(df_processed1.columns)

type_col=df_processed1['type']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_processed1)
df_processed1 = pd.DataFrame(scaled_data, columns=df_processed1.columns)
df_processed1['type']=type_col

type_col=df_processed2['type']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_processed2)
df_processed2 = pd.DataFrame(scaled_data, columns=df_processed2.columns)
df_processed2['type']=type_col

df1 = df1.dropna()
df2 = df2.dropna()

df_processed1.to_csv('processed_celestial_data.csv', index=False)
df_processed2.to_csv('processed_celestial_test.csv', index=False)