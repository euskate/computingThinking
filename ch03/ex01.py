# 고객 피드백을 분석하여 개선점을 도출하는 창의적 방법

import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# 고객 피드백 데이터
feedback_data = pd.DataFrame({
    'Customer': ['A', 'B', 'C', 'D'],
    'Feedback': [
        'The service was excellent but the wait time was long.',
        'I loved the product but the customer support was unresponsive.',
        'The quality was poor but delivery was fast.',
        'Great overall experience but the pricing was high.'
    ]
})

# 감성 분석을 통한 피드백 개선점 도출
def analyze_sentiment(feedback):
    analysis = TextBlob(feedback)
    return analysis.sentiment.polarity

feedback_data['Sentiment'] = feedback_data['Feedback'].apply(analyze_sentiment)

# 피드백 개선점을 시각화
plt.bar(feedback_data['Customer'], feedback_data['Sentiment'])
plt.xlabel('Customer')
plt.ylabel('Sentiment Score')
plt.title('Customer Feedback Sentiment Analysis')
plt.show()