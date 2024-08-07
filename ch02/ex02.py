from sklearn.metrics import confusion_matrix, classification_report

# 예측 결과와 실제 결과 (예시)
y_true = [0, 1, 1, 0, 1, 0, 1, 0]
y_pred = [0, 0, 1, 0, 1, 1, 0, 0]

# 혼동 행렬과 분류 리포트
cm = confusion_matrix(y_true, y_pred)
cr = classification_report(y_true, y_pred)

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(cr)