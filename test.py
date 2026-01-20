# nxn矩阵，3x3， [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
# 回字形遍历
matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]

def roundsk(mat):
    res = []

    m, n = len(mat), len(mat[0])

    i = j = 0

    while len(res)<m*n:
        res.append(mat[i][j])
        mat[i][j] = -1

        print(res, i, j)

        # to right, i=0或上方=-1，右方没到顶
        if (i==0 or mat[i-1][j]==-1) and j<n-1 and mat[i][j+1]!=-1:
            j += 1
            continue
        # to down
        if (j==n-1 or mat[i][j+1]==-1) and i<m-1 and mat[i+1][j]!=-1:
            i += 1
            continue
        # to left
        if (i==m-1 or mat[i+1][j]==-1) and j>=1 and mat[i][j-1]!=-1:
            j -= 1
            continue
        # to up
        if (j==0 or mat[i][j-1]==-1) and i>=1 and mat[i-1][j]!=-1:
            i -= 1
            continue

    return res


print(roundsk(matrix))