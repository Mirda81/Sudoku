import numpy as np
import math


def get_b_r_c(matrix, x, y):
    row_full = matrix[x, :]
    col_full = matrix[:, y]
    box_full = matrix[math.floor(x / 3) * 3: math.floor(x / 3) * 3 + 3,
               math.floor(y / 3) * 3: math.floor(y / 3) * 3 + 3]
    box_full = list(box_full.flatten())
    return box_full, row_full, col_full


def check_poss(matrix, x, y):
    box, row, col = get_b_r_c(matrix, x, y)
    full = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    row_full = [cislo for cislo in row if cislo != 0]
    col_full = [cislo for cislo in col if cislo != 0]
    box = [cislo for cislo in box if cislo != 0]

    denied_nums = row_full + col_full + box
    possible_nums = [x for x in full if x not in denied_nums]

    return possible_nums


def get_poss_matrix(matrix):
    matice_moznosti = matrix.copy()
    for x in range(9):
        for y in range(9):
            if matrix[x, y] == 0:
                moznosti = check_poss(matrix, x, y)
                matice_moznosti[x, y] = moznosti
    return matice_moznosti

def unpack_moznosti(box):
 result = []
 for cell in range(9):
  hodnota = box[cell]
  if not isinstance(hodnota, int):
   result.extend(box[cell])
 return  result

# matice = np.array([[0, 0, 0, 0, 0, 0, 0, 6, 0],
#                    [2, 8, 0, 0, 0, 0, 0, 0, 4],
#                    [0, 0, 7, 0, 0, 5, 8, 0, 0],
#                    [5, 0, 0, 3, 4, 0, 0, 2, 0],
#                    [4, 0, 0, 5, 0, 1, 0, 0, 8],
#                    [0, 1, 0, 0, 7, 6, 0, 0, 3],
#                    [0, 0, 5, 1, 0, 0, 2, 0, 0],
#                    [3, 0, 0, 0, 0, 0, 0, 8, 1],
#                    [0, 9, 0, 0, 0, 0, 0, 0, 0]], dtype=object)


# matice = np.array([[2,0,5,0,0, 7, 0, 0, 6],
#  [4, 0, 0, 9, 6, 0, 0, 0, 0],
#  [0, 0, 0, 0, 8, 0, 0, 4, 5],
#  [9, 8, 0, 0, 7, 4, 0, 0, 0],
#  [5, 7, 0, 8, 0, 2, 0, 6, 9],
#  [0, 0, 0, 6, 3, 0, 0, 5, 7],
#  [7, 5, 0, 0, 2, 0, 0, 0, 0],
#  [0, 6, 0, 0, 5, 1, 0, 0, 2],
#  [3, 0, 0, 4, 0, 0, 5, 0, 8]], dtype=object)

# matice = np.array([[8,0,0,0,1, 0, 0, 0, 9],
#  [0, 5, 0, 8, 0, 7, 0, 1, 0],
#  [0, 0, 4, 0, 9, 0, 7, 0, 0],
#  [0, 6, 0, 7, 0, 1, 0, 2, 0],
#  [5, 0, 8, 0, 6, 0, 1, 0, 7],
#  [0, 1, 0, 5, 0, 2, 0, 9, 0],
#  [0, 0, 7, 0, 4, 0, 6, 0, 0],
#  [0, 8, 0, 3, 0, 9, 0, 4, 0],
#  [3, 0, 0, 0, 5, 0, 0, 0, 8]], dtype=object)
# print(matice[3:6,6:])
# print(matice_moznosti[3:6,6:])
def solve_sudoku(matice):
    matice = matice.astype(object)
    pocitadlo = 0
    while matice.size - np.count_nonzero(matice) >0:
        matice_moznosti = get_poss_matrix(matice)
        for row in range(9):
            for col in range(9):
                if matice[row,col] == 0:
                    moznosti = matice_moznosti[row,col]

                    box_moznosti, row_moznosti, col_moznosti = get_b_r_c(matice_moznosti,row,col)
                    box2 = unpack_moznosti(box_moznosti)
                    col2 = unpack_moznosti(col_moznosti)
                    row2 = unpack_moznosti(row_moznosti)
                    for moznost in moznosti:
                     if box2.count(moznost) == 1 or col2.count(moznost) == 1 or row2.count(moznost) == 1:
                      matice[row, col] = moznost
                      matice_moznosti = get_poss_matrix(matice)
        vyreseno_po = np.count_nonzero(matice)
        # if vyreseno_po == vyreseno:
        #     matice[0,0] = 1
        if pocitadlo == 1000:
            break
        else:
            pocitadlo+=1
    print(matice)
    return matice





# box_moznosti, row_moznosti, col_moznosti = get_b_r_c(matice_moznosti,3,6)
# box2 = unpack_moznosti(box_moznosti)
# box2.count(1)
# print(box2)
# print(box2.count(1))