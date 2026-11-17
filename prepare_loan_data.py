import pandas as pd

url = "https://raw.githubusercontent.com/Architectshwet/Loan-prediction-using-Machine-Learning-and-Python/master/data/train_u6lujuX_CVtuZ9i.csv"

df = pd.read_csv(url)

# معالجة القيم الناقصة
for col in ["Gender", "Married", "Self_Employed"]:
    df[col] = df[col].fillna(df[col].mode()[0])

df["Dependents"] = df["Dependents"].replace("3+", "3")
df["Dependents"] = pd.to_numeric(df["Dependents"])
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].median())

df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(
    df["Loan_Amount_Term"].mode()[0]
)

df["Credit_History"] = df["Credit_History"].fillna(-1)

# تحويل نتيجة القرض إلى أرقام
df["Loan_Status"] = df["Loan_Status"].map({"N": 0, "Y": 1})

# حفظ البيانات
df.to_csv("loan_clean.csv", index=False)

print("حجم البيانات:", df.shape)
print("القيم الناقصة:", df.isnull().sum().sum())
print("تم حفظ loan_clean.csv بنجاح!")