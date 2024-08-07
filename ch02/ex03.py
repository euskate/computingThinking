import pandas as pd

# 데이터 수집 (예시 데이터)
data = pd.DataFrame({
    'Applicant': ['A', 'B', 'C', 'D', 'E'],
    'Income': [50000, 60000, 55000, 70000, 80000],
    'Loan_Amount': [10000, 12000, 11000, 14000, 16000],
    'Approved': [1, 0, 1, 1, 0]
})

# 편향 확인
grouped_data = data.groupby('Approved').mean()
print("Average Loan Amount by Approval Status:")
print(grouped_data)