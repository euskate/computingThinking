import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 학생 성과 데이터
data = pd.DataFrame({
    'Student': ['A', 'B', 'C', 'D', 'E'],
    'Score': [90, 80, 70, 60, 50]
})

# 데이터 정규화
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['Score']])

# K-평균 군집화
kmeans = KMeans(n_clusters=2, random_state=0).fit(data_scaled)
data['Cluster'] = kmeans.labels_

# 적절한 학습 자료 제공
learning_materials = {0: 'Advanced Topics', 1: 'Basic Topics'}

data['Recommended_Material'] = data['Cluster'].map(learning_materials)
print(data)