from numpy import *

A = mat(random.rand(4,4))

myEye = A*A.I
myEye = myEye - eye(4)

print(myEye)