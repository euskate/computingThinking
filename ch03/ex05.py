import pandas as pd

# AI 데이터 예제
data = pd.DataFrame({
    'User': ['A', 'B', 'C', 'D'],
    'Usage': [10, 15, 5, 20],
    'Ethical_Concerns': ['None', 'Privacy Issue', 'Bias', 'None']
})

# 윤리적 우려 사항 분석
concern_summary = data['Ethical_Concerns'].value_counts()
print("Ethical Concerns Summary:")
print(concern_summary)

# 윤리적 우려가 있는 경우 조치 계획
for index, row in data.iterrows():
    if row['Ethical_Concerns'] != 'None':
        print(f"\nAction Plan for User {row['User']}: Address {row['Ethical_Concerns']}")