import pandas as pd

file_path = "../data/RV_March2024.xlsx"

xls = pd.ExcelFile(file_path)
print("Sheet names:")
print(xls.sheet_names)

for name in xls.sheet_names:
    print("\n=== Sheet:", name, "===")
    df = pd.read_excel(xls, sheet_name=name, nrows=5)
    print("Shape preview:", pd.read_excel(xls, sheet_name=name).shape)
    print(df.head())
