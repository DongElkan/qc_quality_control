"""
This module provides tools to generate the sets for calculating
multinomial coefficients, and permute an array.

"""
import numba as nb
import numpy as np


@nb.njit(["float64[:,:](float64[:])", "int64[:,:](int64[:])"])
def permutation(x) -> np.ndarray:
    """
    Permutes array using plain changes.

    Args:
        x: Array

    Returns:
        array: All permutations of the array.

    """
    n = x.size
    c = np.zeros(n, dtype=nb.int64)

    # number of all permutations
    nt = 1
    for i in range(1, n + 1):
        nt *= i

    p = np.zeros((nt, n), dtype=x.dtype)
    for i in range(n):
        p[0][i] = x[i]

    j = 1
    k = 0
    while j < n:
        if c[j] < j:
            if np.mod(j, 2) == 0:
                jl = 0
            else:
                jl = c[j]

            # swap
            xl = x[jl]
            x[jl] = x[j]
            x[j] = xl

            k += 1
            for i in range(n):
                p[k][i] = x[i]

            c[j] += 1
            j = 1
        else:
            c[j] = 0
            j += 1

    return p


def partitions(n, m):
    """
    Finds all partitions of n having m elements.

    Args:
         n: Int for partition.
         m: No. of isotopes.

    Returns:
        list: List of partitions.

    References:
        DE Knuth. Algorithm H, Combinatorial Algorithms. In
        The Art of Computer Programming. Volume 4A, Part 1.

    """
    c = []
    m1 = m + 1
    a = [0, n] + [0] * (m - 1) + [-1]
    while 1:
        c.append(a[1: m1])
        while a[2] < a[1] - 1:
            a[1] -= 1
            a[2] += 1
            c.append(a[1: m1])

        j = 3
        s = a[1] + a[2] - 1
        while a[j] >= a[1] - 1:
            s += a[j]
            j += 1
        if j > m:
            break

        x = a[j] + 1
        a[j] = x
        j -= 1
        while j > 1:
            a[j] = x
            s -= x
            j -= 1
            a[1] = s

    return c


# if __name__ == "__main__":
    # cx = partitions(20, 6)
    # cc = []
    # for ck in cx:
    #     for p in permutations(ck):
    #         cc.append(p)
    # print(len(cc))
    # x = list(range(10))
    # print(len(list(permutations(x))))
    # xx = np.arange(6).astype(np.int64)
    # t = plain_change(xx)
    # print(t.shape)
    # for tk in t:
    #     print(tk)
