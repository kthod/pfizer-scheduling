import pandas as pd

# Load the Excel files
file1 = pd.read_excel(r"C:\Users\thodoris\Documents\Pfizer\DE22_20240508.xlsx")
file2 = pd.read_excel(r"C:\Users\thodoris\Documents\Pfizer\DE22_Alternative_Ressources_20240508_Latest.xlsx")

# Perform a join (e.g., inner join on a common column)
result = pd.merge(file1, file2, left_on=["MATERIAL_KEY", "WORK_CENTER_RESOURCE"], right_on=["MATERIAL_KEY", "WORK_CENTER_RESOURCE"], how="inner")

# Save the result to a new Excel file
result.to_excel(r"C:\Users\thodoris\Documents\Pfizer\result_fail.xlsx", index=False)
