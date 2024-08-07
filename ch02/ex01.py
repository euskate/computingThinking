import pandas as pd
import matplotlib.pyplot as plt

# 데이터 수집
data = {
    'Customer': ['A', 'B', 'C', 'D', 'E'],
    'Purchase_Amount': [100, 150, 200, 250, 300],
    'Frequency': [5, 6, 7, 8, 9]
}
df = pd.DataFrame(data)

# 데이터 시각화
plt.figure(figsize=(10, 5))

# 구매 금액 시각화
plt.subplot(1, 2, 1)
plt.bar(df['Customer'], df['Purchase_Amount'])
plt.title('Purchase Amount by Customer')
plt.xlabel('Customer')
plt.ylabel('Amount')

# 구매 빈도 시각화
plt.subplot(1, 2, 2)
plt.bar(df['Customer'], df['Frequency'])
plt.title('Purchase Frequency by Customer')
plt.xlabel('Customer')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()