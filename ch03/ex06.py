from textblob import TextBlob

# 고객 피드백 데이터
feedbacks = [
    "I am very happy with your service!",
    "The experience was disappointing and frustrating.",
    "I feel okay about the product, but it could be improved.",
    "Great support, but the product didn't meet my expectations."
]

# 감정 분석
def analyze_emotion(feedback):
    analysis = TextBlob(feedback)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

emotions = [analyze_emotion(feedback) for feedback in feedbacks]
print("Customer Feedback Emotions:")
for feedback, emotion in zip(feedbacks, emotions):
    print(f"Feedback: '{feedback}' - Emotion: {emotion}")