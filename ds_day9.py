import numpy as np

def main():

    print("=" * 60)
    print("           MATRIX OPERATIONS USING NUMPY")
    print("=" * 60)

    matrix_A = np.array([[2, 4],
                         [6, 8]])

    matrix_B = np.array([[1, 3],
                         [5, 7]])

    addition_result = matrix_A + matrix_B

    multiplication_result = np.dot(matrix_A, matrix_B)

    print("\nMatrix A:")
    print(matrix_A)

    print("\nMatrix B:")
    print(matrix_B)

    print("\n" + "-" * 60)
    print("                MATRIX ADDITION")
    print("-" * 60)
    print(addition_result)

    print("\n" + "-" * 60)
    print("            MATRIX MULTIPLICATION")
    print("-" * 60)
    print(multiplication_result)

    print("\n" + "=" * 60)
    print("Matrix operations completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()