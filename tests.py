import numpy as np
N = 500
with open("tests_result.txt", 'w') as file:
    for i in range(5):
        print(i)
        with open("../parapells/matrix/" + str(N) + "/A.txt", 'r') as aFile:
            A = np.loadtxt(aFile)

        with open("../parapells/matrix/" + str(N) + "/B.txt", 'r') as bFile:
            B = np.loadtxt(bFile)

        matr = np.dot(A, B)
        with open("../parapells/matrix/" + str(N) + "/C.txt", 'r') as cFile:
            C = np.loadtxt(cFile)
        result = np.array_equal(matr, C)

    
        file.write(str(N) + ' ' + str(result) + '\n')
        N += 100
print("complete!")