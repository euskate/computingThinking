import re

def is_spam(email_content):
    spam_keywords = ['free', 'discount', 'urgent', 'limited', 'buy']
    email_content = email_content.lower()
    
    for keyword in spam_keywords:
        if keyword in email_content:
            return True
    
    return False

# 예제 이메일 내용
email1 = "Congratulations! You have won a FREE gift card. Click here to claim it."
email2 = "Meeting Reminder: Team meeting at 3 PM tomorrow."

print(f"Email 1 is spam: {is_spam(email1)}")
print(f"Email 2 is spam: {is_spam(email2)}")