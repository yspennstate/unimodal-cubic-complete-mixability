"""Exact checks for the paper on unimodal cubic densities with mean 1/3.

Run: python check.py   (needs sympy)
"""
import sympy as sp

x, y, t, C, D = sp.symbols('x y t C D')


def mass_mean(f):
    return sp.integrate(f, (x, 0, 1)), sp.integrate(x * f, (x, 0, 1))


out = []

# The general density of degree <= 3 with mass 1 and mean 1/3.
f = 2 + C / 6 + D / 5 + (-2 - C - sp.Rational(9, 10) * D) * x + C * x**2 + D * x**3
m0, m1 = mass_mean(f)
out.append(('general cubic: mass, mean', sp.simplify(m0), sp.simplify(m1)))

A = sp.Rational(16, 9) - 4 * x**2 + sp.Rational(20, 9) * x**3
B = sp.Rational(11, 6) - 5 * x**2 + sp.Rational(10, 3) * x**3
H = sp.Rational(4, 3) * (1 - x)**2 * (1 + 5 * x)
for name, g in (('A', A), ('B', B), ('H', H)):
    out.append((name + ': mass, mean, derivative', *mass_mean(g), sp.factor(sp.diff(g, x))))

# The coupling for H: density 8(xy+xz+yz) on the triangle, z = 1-x-y.
z = 1 - x - y
q = 8 * (x * y + x * z + y * z)
out.append(('q: total mass', sp.integrate(sp.integrate(q, (y, 0, 1 - x)), (x, 0, 1))))
out.append(('q: x-marginal minus H', sp.simplify(sp.integrate(q, (y, 0, 1 - x)) - H)))

# The decomposition f = (3v/4)A + 6wB + (u/4)H.
u = sp.diff(f, x).subs(x, 0)
v = -sp.diff(f, x).subs(x, 1)
w = f.subs(x, 1)
out.append(('decomposition residual', sp.expand(sp.Rational(3, 4) * v * A + 6 * w * B + u / 4 * H - f)))
out.append(('weights sum minus 1', sp.expand(sp.Rational(3, 4) * v + 6 * w + u / 4 - 1)))

# The quartic counterexample.
p0 = x / 101 * (2360 - 8070 * x + 9360 * x**2 - 3645 * x**3)
out.append(('p0: mass, mean', *mass_mean(p0)))
out.append(('p0 derivative', sp.factor(sp.diff(p0, x))))
out.append(('p0(0), p0(1)', p0.subs(x, 0), p0.subs(x, 1)))
p = sp.Rational(99, 100) * p0 + sp.Rational(1, 100) * 2 * (1 - x)
out.append(('p: mass, mean', *mass_mean(p)))
out.append(('p(0), p(1), p\'(0), p\'(2/9)', p.subs(x, 0), p.subs(x, 1),
            sp.diff(p, x).subs(x, 0), sp.diff(p, x).subs(x, sp.Rational(2, 9))))
out.append(('real roots of p\' in [0,1]', [r for r in sp.real_roots(sp.Poly(sp.diff(p, x), x))
                                          if 0 <= r <= 1]))
out.append(('min of p on [0,1] (critical points and ends)',
            min([p.subs(x, 0), p.subs(x, 1)] + [p.subs(x, r) for r in sp.real_roots(sp.Poly(sp.diff(p, x), x))
                                                 if 0 <= r <= 1])))


# Inequality (4) F(t) >= 2(1 - F(1-t)) at t = 1/200 for p, and (5) for r = 3 - 8x + 6x^2.
def cdf(g, s):
    return sp.integrate(g, (x, 0, s))


s = sp.Rational(1, 200)
out.append(('p: F(t) - 2(1-F(1-t)) at t=1/200', cdf(p, s) - 2 * (1 - cdf(p, 1 - s))))
r = 3 - 8 * x + 6 * x**2
out.append(('r: mass, mean', *mass_mean(r)))
gap = sp.integrate((t - x) * r, (x, 0, t)) - sp.integrate((x - (1 - 2 * t)) * r, (x, 1 - 2 * t, 1))
out.append(('r: E(t-X)_+ - E(X-(1-2t))_+', sp.factor(gap)))
# The L^1 case of Wang's norm condition at n = 3, mu = 1/3: 2E(t-X)_+ - E(X-(1-2t))_+ >= 0 on (0,1/4).
wang = sp.factor(2 * sp.integrate((t - x) * r, (x, 0, t)) - sp.integrate((x - (1 - 2 * t)) * r, (x, 1 - 2 * t, 1)))
out.append(('r: 2E(t-X)_+ - E(X-(1-2t))_+', wang,
            'real roots in (0,1/4):', [z for z in sp.real_roots(sp.Poly(sp.expand(wang / t**2), t)) if 0 < z < sp.Rational(1, 4)]))
# Endpoint identity: g(0) - 4 g(1) in terms of u, v, w.
out.append(('g(0) - 4g(1) - (u/3 + 4v/3 + 7w)', sp.expand(f.subs(x, 0) - 4 * w - (u / 3 + 4 * v / 3 + 7 * w))))

# Section 6: for a density of degree <= 3 with mass 1 and mean mu, u/4 + 3v/4 + 6w = 30 mu - 9.
mu, a, b = sp.symbols('mu a b')
g = a + b * x + C * x**2 + D * x**3
sol = sp.solve([sp.integrate(g, (x, 0, 1)) - 1, sp.integrate(x * g, (x, 0, 1)) - mu], [a, b], dict=True)[0]
g = g.subs(sol)
ug, vg, wg = sp.diff(g, x).subs(x, 0), -sp.diff(g, x).subs(x, 1), g.subs(x, 1)
out.append(('u/4 + 3v/4 + 6w - (30 mu - 9), mean mu', sp.expand(ug / 4 + sp.Rational(3, 4) * vg + 6 * wg - (30 * mu - 9))))
e = 2 - 6 * x**2 + 4 * x**3
out.append(('2 - 6x^2 + 4x^3: mass, mean, u, v, w', *mass_mean(e), sp.diff(e, x).subs(x, 0),
            -sp.diff(e, x).subs(x, 1), e.subs(x, 1)))

for row in out:
    print(' | '.join(str(c) for c in row))
