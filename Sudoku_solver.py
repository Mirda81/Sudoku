import numpy as np

matice8 = np.array([[0, 0, 0, 0, 0, 0, 0, 6, 0],
                    [2, 8, 0, 0, 0, 0, 0, 0, 4],
                    [0, 0, 7, 0, 0, 5, 8, 0, 0],
                    [5, 0, 0, 3, 4, 0, 0, 2, 0],
                    [4, 0, 0, 5, 0, 1, 0, 0, 8],
                    [0, 1, 0, 0, 7, 6, 0, 0, 3],
                    [0, 0, 5, 1, 0, 0, 2, 0, 0],
                    [3, 0, 0, 0, 0, 0, 0, 8, 1],
                    [0, 9, 0, 0, 0, 0, 0, 0, 0]], dtype=object)


#
# matice8 = np.array([[9, 0, 0, 8, 5, 0, 0, 0, 0],
#  [0, 0, 0, 0, 4, 9, 3, 5, 7],
#  [0, 0, 0, 6, 0, 0, 4, 0, 8],
#  [1, 2, 0, 0, 8, 0, 6, 0, 0],
#  [3, 5, 9, 0, 0, 0, 8, 7, 4],
#  [0, 0, 6, 0, 7, 0, 0, 1, 2],
#  [7, 0, 4, 0, 0, 8, 0, 0, 0],
#  [5, 3, 1, 4, 9, 0, 0, 0, 0],
#  [0, 0, 0, 0, 3, 1, 0, 0, 9]],dtype=object)
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
    full = list(range(1, 10))
    mapa = lambda x: x if int(x) > 0 else full
    flat = map(mapa, matrix.flatten())
    result = np.array(list(flat), dtype='object').reshape((9, 9))
    return result


def initial_eliminates(matrix_poss):
    full_range = list(range(0, 9))
    for row in full_range:
        for col in full_range:
            hodnota = matrix_poss[row, col]
            if isinstance(hodnota, int):
                r0 = (row // 3) * 3
                c0 = (col // 3) * 3
                matrix_poss = eliminate_row(matrix_poss, row, full_range, hodnota)
                matrix_poss = eliminate_col(matrix_poss, full_range, col, hodnota)
                matrix_poss = eliminate_box(matrix_poss, r0, r0 + 3, c0, c0 + 3, hodnota)

    return matrix_poss


def eliminate_row(matrix_poss, r, c, number):
    row = matrix_poss[r,c].copy()
    for i,col in enumerate(row):
        try:
            value_b = col.copy()
            value_b.remove(number)
            row[i] = value_b
        except:
            pass
    matrix_poss[r, c] = row
    return matrix_poss


def eliminate_col(matrix_poss, r,c, number):
    col = matrix_poss[r,c].copy()
    for i,row in enumerate(col):
        try:
            value_b = row.copy()
            value_b.remove(number)
            col[i] = value_b
        except:
            pass
    matrix_poss[r, c] = col
    return matrix_poss


def eliminate_box(matrix_poss, r0, r1, c0, c1, number):
    box_flatten = matrix_poss[r0:r1, c0:c1].flatten().copy()
    for i in range(0, len(box_flatten)):
        try:
            value_b = box_flatten[i].copy()
            value_b.remove(number)
            box_flatten[i] = value_b
        except:
            pass

    matrix_poss[r0:r0 + 3, c0:c0 + 3] = np.array(box_flatten).reshape(3, 3)

    return matrix_poss


def unpack_possibilities(possibilities):
    list_poss = [m_list for m_list in possibilities if isinstance(m_list, list)]
    if len(list_poss) ==0:
        return []
    else:
        result = np.hstack(list_poss)
        return result


def pointing_pairs(matrix_poss):

    for first_dim in range(0, 9, 3):
        for second_dim in range(0, 9):
            full_range = list(range(0, 9))
            # řádky po 3
            indexes = list(np.array([0, 1, 2]))
            if second_dim % 3 == 0 or second_dim == 0:
                full_box_rows = matrix_poss[first_dim:first_dim + 3, (second_dim // 3) * 3:(second_dim // 3) * 3 + 3]
                full_box_colls= matrix_poss[second_dim:second_dim + 3, (first_dim // 3) * 3:(first_dim // 3) * 3 + 3]

            reseny_sloupec = indexes.pop(second_dim - (second_dim // 3) * 3)
            reseny_moznosti_row = unpack_possibilities(full_box_rows[0:3, reseny_sloupec])
            zbytek_moznosti_row = unpack_possibilities(full_box_rows[0:3, indexes].flatten())

            reseny_moznosti_col = unpack_possibilities(full_box_colls[reseny_sloupec,0:3])
            zbytek_moznosti_col = unpack_possibilities(full_box_colls[indexes, 0:3].flatten())

            reseny_moznosti_row = list(set([x for x in reseny_moznosti_row if list(reseny_moznosti_row).count(x) > 1]))
            reseny_moznosti_col = list(set([x for x in reseny_moznosti_col if list(reseny_moznosti_col).count(x) > 1]))
            if len(reseny_moznosti_row) > 0:
                for cislo in reseny_moznosti_row:
                    if cislo not in zbytek_moznosti_row:
                        prohledat_radky = list(set(full_range) - set(range(first_dim,first_dim+3)))
                        matrix_poss = eliminate_col(matrix_poss,prohledat_radky,second_dim,cislo)

            if len(reseny_moznosti_col) > 0:
                for cislo_col in reseny_moznosti_col:
                    if cislo_col not in zbytek_moznosti_col:
                        prohledat_radky = list(set(full_range) - set(range(first_dim,first_dim+3)))
                        matrix_poss = eliminate_row(matrix_poss,second_dim,prohledat_radky, cislo_col)

    return matrix_poss


def fill_solved(matrix_poss, matice):
    full_range=(list(range(0,9)))
    for row in range(0, 9):
        for col in range(0, 9):
            moznosti = matrix_poss[row, col]
            if isinstance(moznosti, list):
                r0 = (row // 3) * 3
                c0 = (col // 3) * 3
                radek_moznosti = unpack_possibilities(matrix_poss[row, 0:9])
                sloupec_moznosti = unpack_possibilities(matrix_poss[0:9, col])
                box_moznosti = unpack_possibilities(matrix_poss[r0:r0 + 3, c0:c0 + 3].flatten())
                if len(moznosti) == 1:
                    hodnota = moznosti[0]
                    matrix_poss[row, col] = hodnota
                    matice[row, col] = hodnota
                    eliminate_found(matrix_poss, row, col, hodnota)
                else:
                    for moznost in moznosti:
                        if list(radek_moznosti).count(moznost) == 1 or list(sloupec_moznosti).count(
                                moznost) == 1 or list(box_moznosti).count(moznost) == 1:
                            matrix_poss[row, col] = moznost
                            matice[row, col] = moznost
                            eliminate_found(matrix_poss, row,col,moznost)

    return matrix_poss, matice

def eliminate_found(matrix_poss,row,col,hodnota):
    r0 = (row // 3) * 3
    c0 = (col // 3) * 3
    full_range=(list(range(0,9)))
    eliminate_row(matrix_poss, row, full_range, hodnota)
    eliminate_col(matrix_poss, full_range, col, hodnota)
    eliminate_box(matrix_poss, r0, r0 + 3, c0, c0 + 3, hodnota)
    return matrix_poss


def solve(matice):
    matice = matice.astype(object)
    matrix_poss = poss_matrix(matice)
    matrix_poss = initial_eliminates(matrix_poss)
    pocitadlo = 0
    while matice.size - np.count_nonzero(matice) > 0:
        # matrix_poss = pointing_pairs(matrix_poss)
        matrix_poss, matice = fill_solved(matrix_poss, matice)

        pocitadlo += 1
        if pocitadlo > 100:
            matrix_poss_orig = matrix_poss.copy()
            break

    return matice





# # fl = matice2.flatten()
# # #
# # #
# # # mapa = lambda x: x**2 if int(x) > 0 else x
# # # result = map(mapa, fl)
print(solve(matice8))
