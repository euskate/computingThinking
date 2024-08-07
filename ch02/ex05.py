# 데이터 리터러시 예제
import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Customer': ['A', 'B', 'C', 'D', 'E'],
    'Purchase_Amount': [100, 150, 200, 250, 300],
    'Frequency': [5, 6, 7, 8, 9]
}
df = pd.DataFrame(data)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(df['Customer'], df['Purchase_Amount'])
plt.title('Purchase Amount by Customer')
plt.xlabel('Customer')
plt.ylabel('Amount')

plt.subplot(1, 2, 2)
plt.bar(df['Customer'], df['Frequency'])
plt.title('Purchase Frequency by Customer')
plt.xlabel('Customer')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 비판적 사고 예제
from sklearn.metrics import confusion_matrix, classification_report

y_true = [0, 1, 1, 0, 1, 0, 1, 0]
y_pred = [0, 0, 1, 0, 1, 1, 0, 0]

cm = confusion_matrix(y_true, y_pred)
cr = classification_report(y_true, y_pred)

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(cr)

# 윤리적 사고 예제
import pandas as pd

data = pd.DataFrame({
    'Applicant': ['A', 'B', 'C', 'D', 'E'],
    'Income': [50000, 60000, 55000, 70000, 80000],
    'Loan_Amount': [10000, 12000, 11000, 14000, 16000],
    'Approved': [1, 0, 1, 1, 0]
})

grouped_data = data.groupby('Approved').mean()
print("Average Loan Amount by Approval Status:")
print(grouped_data)

# 창의적 사고 예제
from textblob import TextBlob
import pandas as pd

feedback = pd.DataFrame({
    'Customer': ['A', 'B', 'C', 'D'],
    'Feedback': [
        'I love the new feature, it is fantastic!',
        'The product is okay, but the delivery was slow.',
        'I am not satisfied with the product quality.',
        'Great service and fast delivery!'
    ]
})

def analyze_sentiment(feedback_text):
    analysis = TextBlob(feedback_text)
    return analysis.sentiment.polarity

feedback['Sentiment'] = feedback['Feedback'].apply(analyze_sentiment)
print("Customer Feedback with Sentiment Analysis:")
print(feedback)