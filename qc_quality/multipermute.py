"""
This module provides tools to generate the sets for calculating
multinomial coefficients.

The codes for permutation of multiset is copied from multipermute
(https://github.com/ekg/multipermute).
@copyright to Erik Garrison <erik.garrison@bc.edu> 2010

Modified by Nai-ping Dong (naipdong@polyu.edu.hk)

References:
A Williams. Loopless Generation of Multiset Permutations using a
Constant Number of Variables by Prefix Shifts. SODA '09: Proceedings of
the twentieth annual ACM-SIAM symposium on Discrete algorithms. 2009,
987–996.

"""
from typing import Union, Optional, Any
import dataclasses


@dataclasses.dataclass
class ListElement:
    value: Union[int, float]
    nxt: Optional[Any]

    def nth(self, n: int):
        o = self
        i = 0
        while i < n and o.nxt is not None:
            o = o.nxt
            i += 1
        return o


def init(multiset):
    multiset.sort()
    h = ListElement(multiset[0], None)
    for item in multiset[1:]:
        h = ListElement(item, h)
    return h, h.nth(len(multiset) - 2), h.nth(len(multiset) - 1)


def visit(h):
    """Converts our bespoke linked list to a python list."""
    o = h
    v = []
    while o is not None:
        v.append(o.value)
        o = o.nxt
    return v


def permutations(multiset):
    """Generator providing all multiset permutations of a multiset."""
    perms = []
    h, i, j = init(multiset)
    perms.append(visit(h))
    while j.nxt is not None or j.value < h.value:
        if j.nxt is not None and i.value >= j.nxt.value:
            s = j
        else:
            s = i
        t = s.nxt
        s.nxt = t.nxt
        t.nxt = h
        if t.value < h.value:
            i = t
        j = i.nxt
        h = t
        perms.append(visit(h))

    return perms


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
#     cx = partitions(20, 6)
#     cc = []
#     for ck in cx:
#         for p in permutations(ck):
#             cc.append(p)
#     print(len(cc))
    # x = list(range(10))
    # print(len(list(permutations(x))))
