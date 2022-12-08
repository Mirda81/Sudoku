def next_box(quiz):
    for row in range(9):
        for col in range(9):
            if quiz[row][col] == 0:
                return (row, col)
    return False


# Function to fill in the possible values by evaluating rows collumns and smaller cells

def possible(quiz, row, col, n):
    # global quiz
    for i in range(0, 9):
        if quiz[row][i] == n and row != i:
            return False
    for i in range(0, 9):
        if quiz[i][col] == n and col != i:
            return False

    row0 = (row) // 3
    col0 = (col) // 3
    for i in range(row0 * 3, row0 * 3 + 3):
        for j in range(col0 * 3, col0 * 3 + 3):
            if quiz[i][j] == n and (i, j) != (row, col):
                return False
    return True


# Recursion function to loop over untill a valid answer is found.

def solve(quiz):
    val = next_box(quiz)
    if val is False:
        return True
    else:
        row, col = val
        for n in range(1, 10):  # n is the possible solution
            if possible(quiz, row, col, n):
                quiz[row][col] = n
                if solve(quiz):
                    return True
                else:
                    quiz[row][col] = 0
        return


def Solved(quiz):
    for row in range(9):
        if row % 3 == 0 and row != 0:
            print("....................")

        for col in range(9):
            if col % 3 == 0 and col != 0:
                print("|", end=" ")

            if col == 8:
                print(quiz[row][col])
            else:
                print(str(quiz[row][col]) + " ", end="")
