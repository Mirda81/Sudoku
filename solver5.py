import numpy as np
import time as t

def solve_sudoku(board):
    start = t.time()
    # Funkce, která vrátí seznam možných hodnot pro políčko na zadaných řádku a sloupci
    def possible_values(row, col):
        used_values = set()
        # Zjistěme, které hodnoty jsou již použity v daném řádku
        for i in range(9):
            if board[row][i] != 0:
                used_values.add(board[row][i])
        # Zjistěme, které hodnoty jsou již použity v daném sloupci
        for i in range(9):
            if board[i][col] != 0:
                used_values.add(board[i][col])
        # Zjistěme, které hodnoty jsou již použity v daném bloku
        start_row = row // 3 * 3
        start_col = col // 3 * 3
        for i in range(3):
            for j in range(3):
                if board[start_row + i][start_col + j] != 0:
                    used_values.add(board[start_row + i][start_col + j])
        # Vraťte seznam možných hodnot, které nejsou již použité v řádku, sloupci nebo bloku
        return [i for i in range(1, 10) if i not in used_values]

    # Funkce rekurzivně procházející sudoku a zkoušející možné hodnoty pro každé políčko
    def solve(board):
        for row in range(9):
            for col in range(9):
                # Pokud je políčko prázdné, zkoušejte možné hodnoty
                if board[row][col] == 0:
                    for value in possible_values(row, col):
                        board[row][col] = value
                        if solve(board):
                            return True
                        board[row][col] = 0
                    return False
        return True

    # Spusťte rekurzivní funkci pro vyřešení sudoku
    solve(board)

    return board, "Solved in %.4fs" % (t.time() - start)

grid = np.array([[9, 0, 0, 8, 5, 0, 0, 0, 0],
 [0, 0, 0, 0, 4, 9, 3, 5, 7],
 [0, 0, 0, 6, 0, 0, 4, 0, 8],
 [1, 2, 0, 0, 8, 0, 6, 0, 0],
 [3, 5, 9, 0, 0, 0, 8, 7, 4],
 [0, 0, 6, 0, 7, 0, 0, 1, 2],
 [7, 0, 4, 0, 0, 8, 0, 0, 0],
 [5, 3, 1, 4, 9, 0, 0, 0, 0],
 [0, 0, 0, 0, 3, 1, 0, 0, 9]])

solved_grid = solve_sudoku(grid)

print(solved_grid)