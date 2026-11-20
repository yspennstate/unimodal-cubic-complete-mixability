# Complete mixability of unimodal cubic densities at the boundary mean

Code for the paper "Complete mixability of unimodal cubic densities at the boundary mean" (Y. Shmalo).

The paper shows that every unimodal probability density on [0,1] that is a polynomial of degree at most three and has mean 1/3 is 3-completely mixable, and gives a strictly positive unimodal quartic density with mean 1/3 that is not.

The paper source and PDF are in `paper/`.

`check.py` needs SymPy. It checks in exact rational arithmetic:

- the normal form of a cubic density with mean 1/3, and the integrals, means and derivatives of A, B and H;
- the marginal of the coupling with density 8(xy+xz+yz) on the triangle;
- the decomposition g = (3v/4)A + 6wB + (u/4)H and the identity g(0) - 4g(1) = u/3 + 4v/3 + 7w;
- every value used for the quartic examples p0 and p and the quadratic r.

The output of `python check.py` is in `check_output.txt`. It runs in a few seconds.
