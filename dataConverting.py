import pandas as pd
import h5py as h5 

file_name = "METR-LA.h5"

data = h5.File(file_name, 'r')

with pd.HDFStore(file_name, 'r') as d:
    df = d.get('df')
    df.to_csv("metr-la.csv")

# bikin dulu folder raw_data

path = "raw_data"
df = pd.read_csv('metr-la.csv')
df['Time'] = pd.to_datetime(df['Time'])
col = list(df.columns)

for i in col[1:]:
    data = df[['Time', i]]
    data.to_csv(f'{path}/{i}.csv', index=False)
    print(f'{path}/{i}.csv', 'has been created')