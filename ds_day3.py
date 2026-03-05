def evaluate_result(marks):
    if marks >= 40:
        return "PASS"
    else:
        return "FAIL"


def main():

    print("=" * 60)
    print("STUDENT RESULT EVALUATION SYSTEM")
    print("=" * 60)

    student_name = input("Enter Student Name: ")
    marks = float(input("Enter Student Marks: "))

    result = evaluate_result(marks)

    print("\n" + "-" * 60)
    print("RESULT SUMMARY")
    print("-" * 60)

    print(f"{'Student Name':<20}: {student_name}")
    print(f"{'Marks Obtained':<20}: {marks}")
    print(f"{'Final Result':<20}: {result}")

    print("-" * 60)

    if result == "PASS":
        print("Status: Student has successfully passed the examination.")
    else:
        print("Status: Student did not meet the passing criteria.")

    print("=" * 60)


if __name__ == "__main__":
    main()