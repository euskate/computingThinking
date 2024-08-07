# 협력적 사고를 위한 팀원 역할 분담 예제

team_roles = {
    'Alice': 'Researcher',
    'Bob': 'Designer',
    'Charlie': 'Developer',
    'David': 'Tester'
}

print("Team Roles:")
for member, role in team_roles.items():
    print(f"{member}: {role}")

# 역할에 따른 업무 리스트
tasks = {
    'Researcher': ['Market Analysis', 'Competitor Analysis'],
    'Designer': ['UI/UX Design', 'Prototyping'],
    'Developer': ['Coding', 'Feature Implementation'],
    'Tester': ['Testing', 'Bug Reporting']
}

for role, task_list in tasks.items():
    print(f"\nTasks for {role}:")
    for task in task_list:
        print(f"- {task}")