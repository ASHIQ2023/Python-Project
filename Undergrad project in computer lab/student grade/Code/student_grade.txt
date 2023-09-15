import csv

students = {}

def add_student(student_id, name):
    students[student_id] = {"name": name, "grades": {}}

def add_grade(student_id, subject, grade):
    if student_id in students:
        students[student_id]["grades"][subject] = float(grade)
    else:
        print("Student not found.")

def calculate_average(student_id):
    if student_id in students:
        grades = students[student_id]["grades"].values()
        average = sum(grades) / len(grades)
        print("Average grade for {}: {:.2f}".format(name, average))
    else:
        print("Student not found.")

def grade_summary(student_id):
    if student_id in students:
        print(f"Grade Summary for {students[student_id]['name']}:")
        for subject, grade in students[student_id]["grades"].items():
            print("{}: {}".format(subject, grade))
    else:
        print("Student not found.")

def find_highest_and_lowest_grades():
    all_grades = [grade for student in students.values() for grade in student["grades"].values()]
    if all_grades:
        highest_grade = max(all_grades)
        lowest_grade = min(all_grades)
        print("Highest Grade: {:.2f}".format(highest_grade))
        print("Lowest Grade: {:.2f}".format(lowest_grade))
    else:
        print("No grades found.")

def save_data_to_csv(filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Student ID', 'Name', 'Subject', 'Grade'])
        for student_id, student_data in students.items():
            for subject, grade in student_data["grades"].items():
                writer.writerow([student_id, student_data["name"], subject, grade])

def load_data_from_csv(filename):
    students.clear()
    try:
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                student_id, name, subject, grade = row
                if student_id not in students:
                    add_student(student_id, name)
                add_grade(student_id, subject, grade)
    except FileNotFoundError:
        print("File not found.")

while True:
    print("\nStudent Grade Tracker")
    print("1. Add Student")
    print("2. Add Grade")
    print("3. Calculate Average")
    print("4. Grade Summary")
    print("5. Highest and Lowest Grades")
    print("6. Search for a Student")
    print("7. Save Data to CSV")
    print("8. Load Data from CSV")
    print("9. Quit")

    choice = input("Enter your choice: ")

    if choice == "1":
        student_id = input("Enter student ID: ")
        name = input("Enter student name: ")
        add_student(student_id, name)
    elif choice == "2":
        student_id = input("Enter student ID: ")
        subject = input("Enter subject: ")
        grade = input("Enter grade: ")
        add_grade(student_id, subject, grade)
    elif choice == "3":
        student_id = input("Enter student ID: ")
        calculate_average(student_id)
    elif choice == "4":
        student_id = input("Enter student ID: ")
        grade_summary(student_id)
    elif choice == "5":
        find_highest_and_lowest_grades()
    elif choice == "6":
        student_id = input("Enter student ID: ")
        grade_summary(student_id)
    elif choice == "7":
        filename = input("Enter CSV filename to save: ")
        save_data_to_csv(filename)
    elif choice == "8":
        filename = input("Enter CSV filename to load: ")
        load_data_from_csv(filename)
    elif choice == "9":
        break
    else:
        print("Invalid choice. Please try again.")

print("You choose to exit!")