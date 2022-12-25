import numpy as np
# matice2 = np.array([[0, 0, 0, 0, 0, 0, 0, 6, 0],
#                    [2, 8, 0, 0, 0, 0, 0, 0, 4],
#                    [0, 0, 7, 0, 0, 5, 8, 0, 0],
#                    [5, 0, 0, 3, 4, 0, 0, 2, 0],
#                    [4, 0, 0, 5, 0, 1, 0, 0, 8],
#                    [0, 1, 0, 0, 7, 6, 0, 0, 3],
#                    [0, 0, 5, 1, 0, 0, 2, 0, 0],
#                    [3, 0, 0, 0, 0, 0, 0, 8, 1],
#                    [0, 9, 0, 0, 0, 0, 0, 0, 0]], dtype=object)
#
matice8 = np.array([[9, 0, 0, 8, 5, 0, 0, 0, 0],
 [0, 0, 0, 0, 4, 9, 3, 5, 7],
 [0, 0, 0, 6, 0, 0, 4, 0, 8],
 [1, 2, 0, 0, 8, 0, 6, 0, 0],
 [3, 5, 9, 0, 0, 0, 8, 7, 4],
 [0, 0, 6, 0, 7, 0, 0, 1, 2],
 [7, 0, 4, 0, 0, 8, 0, 0, 0],
 [5, 3, 1, 4, 9, 0, 0, 0, 0],
 [0, 0, 0, 0, 3, 1, 0, 0, 9]],dtype=object)
# # matice2 = np.array([[2,0,5,0,0, 7, 0, 0, 6],
# #  [4, 0, 0, 9, 6, 0, 0, 0, 0],
# #  [0, 0, 0, 0, 8, 0, 0, 4, 5],
#  [9, 8, 0, 0, 7, 4, 0, 0, 0],
#  [5, 7, 0, 8, 0, 2, 0, 6, 9],
#  [0, 0, 0, 6, 3, 0, 0, 5, 7],
#  [7, 5, 0, 0, 2, 0, 0, 0, 0],
#  [0, 6, 0, 0, 5, 1, 0, 0, 2],
#  [3, 0, 0, 4, 0, 0, 5, 0, 8]], dtype=object)
#
# # matice = np.array([[8,0,0,0,1, 0, 0, 0, 9],
# #  [0, 5, 0, 8, 0, 7, 0, 1, 0],
# #  [0, 0, 4, 0, 9, 0, 7, 0, 0],
# #  [0, 6, 0, 7, 0, 1, 0, 2, 0],
# #  [5, 0, 8, 0, 6, 0, 1, 0, 7],
# #  [0, 1, 0, 5, 0, 2, 0, 9, 0],
# #  [0, 0, 7, 0, 4, 0, 6, 0, 0],
# #  [0, 8, 0, 3, 0, 9, 0, 4, 0],
# #  [3, 0, 0, 0, 5, 0, 0, 0, 8]], dtype=object)

def poss_matrix(matrix):
    full = list(range(1,10))
    mapa = lambda x: x if int(x) > 0 else full
    flat = matrix.flatten()
    flat = map(mapa, matrix.flatten())
    result = np.array(list(flat),dtype='object').reshape((9,9))
    return result

def elimination_by_numbers(matrix_poss):
    for row in range(0,9):
        for col in range(0,9):
            hodnota = matrix_poss[row,col]
            if isinstance(hodnota,int):
                matrix_poss = eliminate_row(matrix_poss, row,hodnota)
                matrix_poss = eliminate_col(matrix_poss, col, hodnota)
                matrix_poss = eliminate_box(matrix_poss, row,col,hodnota)

    return matrix_poss
def eliminate_row(matrix_poss, r, number):
    for col in range(0,9):
        try:
            value_b = matrix_poss[r, col].copy()
            value_b.remove(number)
            matrix_poss[r, col] = value_b
        except:
            pass
    return matrix_poss

def eliminate_col(matrix_poss, c, number):
    for row in range(0,9):
        try:
            value_b = matrix_poss[row, c].copy()
            value_b.remove(number)
            matrix_poss[row, c] = value_b
        except:
            pass
    return matrix_poss

def eliminate_box(matrix_poss, r,c, number):
    r0 = (r//3)*3
    c0 = (c//3)*3
    box_flatten = matrix_poss[r0:r0+3,c0:c0+3].flatten().copy()
    for i in range(0,9):
        try:
            value_b = box_flatten[i].copy()
            value_b.remove(number)
            box_flatten[i] = value_b
        except:
            pass

    matrix_poss[r0:r0 + 3, c0:c0 + 3] = np.array(box_flatten).reshape(3,3)

    return matrix_poss

def get_row_poss(matrix_poss, r0,c0,c1):
    possibilities = matrix_poss[r0,c0:c1]
    list_poss = [m_list for m_list in possibilities if isinstance(m_list, list)]
    result = np.hstack(list_poss)
    return  result

def get_col_poss(matrix_poss, r0,r1, c0):
    possibilities = matrix_poss[r0:r1,c0]
    list_poss = [m_list for m_list in possibilities if isinstance(m_list, list)]
    result = np.hstack(list_poss)
    return result

def get_box_poss(matrix_poss, r0, c0):
    possibilities = matrix_poss[r0:r0+3,c0:c0+3].flatten()
    list_poss = [m_list for m_list in possibilities if isinstance(m_list, list)]
    result = np.hstack(list_poss)
    return result

def fill_solved(matrix_poss,matice):
    for row in range(0,9):
        for col in range(0,9):
            moznosti = matrix_poss[row,col]
            if isinstance(moznosti, list):
                radek_moznosti = get_row_poss(matrix_poss, row,0,9)
                sloupec_moznosti = get_col_poss(matrix_poss, 0,9,col)
                box_moznosti = get_box_poss(matrix_poss, (row//3)*3,(col//3)*3)
                if len(moznosti) == 1:
                    hodnota = moznosti[0]
                    matrix_poss[row,col] = hodnota
                    matice[row, col] = hodnota
                    eliminate_row(matrix_poss, row, hodnota)
                    eliminate_col(matrix_poss, col, hodnota)
                    eliminate_box(matrix_poss, row, col, hodnota)

                else:
                    for moznost in moznosti:
                        if list(radek_moznosti).count(moznost) == 1 or list(sloupec_moznosti).count(moznost) == 1 or list(box_moznosti).count(moznost) == 1:
                            matrix_poss[row, col] = moznost
                            matice[row, col] = moznost
                            eliminate_row(matrix_poss, row, moznost)
                            eliminate_col(matrix_poss, col, moznost)
                            eliminate_box(matrix_poss, row, col, moznost)

    return matrix_poss,matice



def solve(matice):
    matice = matice.astype(object)
    matrix_poss = poss_matrix(matice)
    matrix_poss = elimination_by_numbers(matrix_poss)
    pocitadlo = 0
    while matice.size - np.count_nonzero(matice) > 0:
        matrix_poss, matice = fill_solved(matrix_poss,matice)
        pocitadlo+=1
        if pocitadlo > 100:
            break
    return matice
# # fl = matice2.flatten()
# # #
# # #
# # # mapa = lambda x: x**2 if int(x) > 0 else x
# # # result = map(mapa, fl)
print(solve(matice8))
