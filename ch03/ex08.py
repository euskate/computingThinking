import pandas as pd

# 품질 지표 데이터
data = pd.DataFrame({
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
    'Bug_Count': [10, 8, 6, 4, 2]
})

# 개선 추세 시각화
data['Bug_Count'].plot(kind='line', marker='o')
plt.xlabel('Month')
plt.ylabel('Bug Count')
plt.title('Bug Count Over Time')
plt.grid(True)
plt.show()