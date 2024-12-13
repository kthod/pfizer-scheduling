import pandas as pd

file_path = r"C:\Users\thodoris\Documents\Pfizer\DE22_20240508.xlsx"  # Replace with the path to your Excel file
df = pd.read_excel(file_path)

# Print column names
print("Column Names:", list(df.columns))

# Iterate through rows line by line
for index, row in df.iterrows():
    print(f"Row {index + 1}:")
    for column_name, value in row.items():
        print(f"  {column_name}: {value}")