# Call c function from python
# To create c library,
# gcc -shared -o libhello.so -fPIC hello.c

import ctypes

chello = ctypes.CDLL("./libhello.so")

chello.hello()