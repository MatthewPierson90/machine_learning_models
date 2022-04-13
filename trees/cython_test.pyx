import cython as c

def func1(n: c.int) -> c.long:
    to_return:c.long= 1
    k: c.long = 1
    while k < n:
        to_return *= k
        k += 1
    return to_return
