from textblob import TextBlob
import pandas as pd

# 고객 피드백 데이터
feedback = pd.DataFrame({
    'Customer': ['A', 'B', 'C', 'D'],
    'Feedback': [
        'I love the new feature, it is fantastic!',
        'The product is okay, but the delivery was slow.',
        'I am not satisfied with the product quality.',
        'Great service and fast delivery!'
    ]
})

# 감정 분석
def analyze_sentiment(feedback_text):
    analysis = TextBlob(feedback_text)
    return analysis.sentiment.polarity

feedback['Sentiment'] = feedback['Feedback'].apply(analyze_sentiment)
print("Customer Feedback with Sentiment Analysis:")
print(feedback)