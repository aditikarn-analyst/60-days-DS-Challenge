def analyze_marks(student_marks):

    marks = list(student_marks.values())

    total_marks = sum(marks)
    average_marks = total_marks / len(marks)
    highest_marks = max(marks)
    lowest_marks = min(marks)
    top_student = max(student_marks, key=student_marks.get)

    return total_marks, average_marks, highest_marks, lowest_marks, top_student


def main():

    print("=" * 60)
    print("        STUDENT MARKS DATA ANALYSIS SYSTEM")
    print("=" * 60)

    student_marks = {}
    n = int(input("Enter number of students: "))
    for i in range(n):
        name = input(f"Enter name of student {i+1}: ")
        marks = float(input(f"Enter marks of {name}: "))
        student_marks[name] = marks

    total, average, highest, lowest, topper = analyze_marks(student_marks)
    print("\n" + "-" * 60)
    print("               ANALYSIS RESULTS")
    print("-" * 60)

    for student, marks in student_marks.items():
        print(f"{student:<20}: {marks}")

    print("-" * 60)
    print(f"{'Total Marks':<20}: {total}")
    print(f"{'Average Marks':<20}: {average:.2f}")
    print(f"{'Highest Marks':<20}: {highest}")
    print(f"{'Lowest Marks':<20}: {lowest}")
    print(f"{'Top Student':<20}: {topper}")

    print("=" * 60)
    print("Analysis completed successfully.")


if __name__ == "__main__":
    main()