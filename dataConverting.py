import pandas as pd
import h5py as h5 

file_name = "METR-LA.h5"

data = h5.File(file_name, 'r')

print(data.keys())

with pd.HDFStore(file_name, 'r') as d:
    df = d.get('df')
    df.to_csv("metr-la.csv")