# 성적 입력
def input_grades():
    grades = {}
    while True:
        name = input("Enter student's name (or 'quit' to stop): ")
        if name == 'quit':
            break
        grade = float(input(f"Enter {name}'s grade: "))
        grades[name] = grade
    return grades

# 성적 저장
def save_grades(grades, filename):
    with open(filename, 'w') as f:
        for name, grade in grades.items():
            f.write(f"{name},{grade}\n")

# 성적 조회
def read_grades(filename):
    grades = {}
    with open(filename, 'r') as f:
        for line in f:
            name, grade = line.strip().split(',')
            grades[name] = float(grade)
    return grades

# 성적 분석
def analyze_grades(grades):
    if not grades:
        return None, None, None
    avg_grade = sum(grades.values()) / len(grades)
    max_grade = max(grades.values())
    min_grade = min(grades.values())
    return avg_grade, max_grade, min_grade

# 보고서 생성
def generate_report(avg_grade, max_grade, min_grade):
    print(f"Average Grade: {avg_grade:.2f}")
    print(f"Highest Grade: {max_grade:.2f}")
    print(f"Lowest Grade: {min_grade:.2f}")

# 메인 프로그램
grades = input_grades()
save_grades(grades, 'grades.txt')
grades = read_grades('grades.txt')
avg_grade, max_grade, min_grade = analyze_grades(grades)
generate_report(avg_grade, max_grade, min_grade)