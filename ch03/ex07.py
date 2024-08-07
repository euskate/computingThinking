import matplotlib.pyplot as plt

# 제조 공정 데이터 예제
stages = ['Raw Materials', 'Production', 'Quality Check', 'Packaging', 'Distribution']
efficiency = [70, 80, 90, 85, 75]

# 시스템적 접근 시각화
plt.plot(stages, efficiency, marker='o')
plt.xlabel('Manufacturing Stages')
plt.ylabel('Efficiency (%)')
plt.title('Manufacturing Process Efficiency')
plt.grid(True)
plt.show()