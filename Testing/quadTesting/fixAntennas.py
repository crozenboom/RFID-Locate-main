import pandas as pd

filename = "quadz4_train.csv"
df = pd.read_csv(filename)
antenna = "antenna"
if antenna not in df.columns:
    print("COLUMN NOT FOUND. Available columns:")
    print(df.columns)
    exit()

if df[antenna].dtype == object:
    try: 
        df[antenna] = df[antenna].astype(int)
    except ValueError:
        print ("Cannot convert antenna column to integers.")

df[antenna] = df[antenna].apply(lambda x: 3 if x == 2 else 2 if x == 3 else x)
df.to_csv(filename, index=False)
print(f"Ports 2 and 3 swapped. File '{filename}' updated")
