# 문제 분해 예제
def input_grades():
    grades = {}
    while True:
        name = input("Enter student's name (or 'quit' to stop): ")
        if name == 'quit':
            break
        grade = float(input(f"Enter {name}'s grade: "))
        grades[name] = grade
    return grades

def save_grades(grades, filename):
    with open(filename, 'w') as f:
        for name, grade in grades.items():
            f.write(f"{name},{grade}\n")

def read_grades(filename):
    grades = {}
    with open(filename, 'r') as f:
        for line in f:
            name, grade = line.strip().split(',')
            grades[name] = float(grade)
    return grades

def analyze_grades(grades):
    if not grades:
        return None, None, None
    avg_grade = sum(grades.values()) / len(grades)
    max_grade = max(grades.values())
    min_grade = min(grades.values())
    return avg_grade, max_grade, min_grade

def generate_report(avg_grade, max_grade, min_grade):
    print(f"Average Grade: {avg_grade:.2f}")
    print(f"Highest Grade: {max_grade:.2f}")
    print(f"Lowest Grade: {min_grade:.2f}")

grades = input_grades()
save_grades(grades, 'grades.txt')
grades = read_grades('grades.txt')
avg_grade, max_grade, min_grade = analyze_grades(grades)
generate_report(avg_grade, max_grade, min_grade)

# 패턴 인식 예제
import re

def is_spam(email_content):
    spam_keywords = ['free', 'discount', 'urgent', 'limited', 'buy']
    email_content = email_content.lower()
    
    for keyword in spam_keywords:
        if keyword in email_content:
            return True
    
    return False

email1 = "Congratulations! You have won a FREE gift card. Click here to claim it."
email2 = "Meeting Reminder: Team meeting at 3 PM tomorrow."

print(f"Email 1 is spam: {is_spam(email1)}")
print(f"Email 2 is spam: {is_spam(email2)}")

# 추상화 예제
def calculator():
    print("Welcome to the simple calculator!")
    print("Select operation: +, -, *, /")
    operation = input("Operation: ")

    num1 = float(input("Enter first number: "))
    num2 = float(input("Enter second number: "))

    if operation == '+':
        result = num1 + num2
    elif operation == '-':
        result = num1 - num2
    elif operation == '*':
        result = num1 * num2
    elif operation == '/':
        if num2 != 0:
            result = num1 / num2
        else:
            return "Error: Division by zero!"
    else:
        return "Invalid operation!"

    return f"Result: {result}"

print(calculator())

# 알고리즘 설계 예제
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

arr = [64, 25, 12, 22, 11]
print(f"Original list: {arr}")
sorted_arr = bubble_sort(arr)
print(f"Sorted list: {sorted_arr}")