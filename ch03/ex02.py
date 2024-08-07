import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# 가상의 판매 데이터
data = pd.DataFrame({
    'Month': np.arange(1, 13),
    'Sales': [200, 210, 215, 220, 230, 240, 245, 250, 260, 270, 275, 280]
})

# 독립 변수와 종속 변수
X = data[['Month']]
y = data['Sales']

# 모델 훈련
model = LinearRegression()
model.fit(X, y)

# 향후 6개월 예측
future_months = np.arange(13, 19).reshape(-1, 1)
predictions = model.predict(future_months)

# 예측 결과 시각화
plt.plot(data['Month'], data['Sales'], label='Historical Sales')
plt.plot(np.arange(1, 19), np.concatenate([y, predictions]), '--', label='Predicted Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Sales Prediction')
plt.legend()
plt.show()