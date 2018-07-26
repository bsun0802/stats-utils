import numpy as np
from scipy.special import psi, polygamma
from scipy import optimize, stats


def psi_inv(c, max_iter=50):
    """ The inverse of Psi(x), i.e. the derivative of the logrithm of gamma function.
        Implementation: the psi_inverse(c) is the root of psi(x) - c = 0,root-finding by Newton's method. The choice of initial point was described in Appendix c of Minka, T. P. (2003). Estimating a Dirichlet distribution.
        Annals of Physics, 2000(8), 1-13. http://doi.org/10.1007/s00256-007-0299-1 for details.
    """
    x0 = np.exp(c) + 0.5 if c >= -2.22 else 1 / (psi(1) - c)
    # for i in range(max_iter):
    #     x0 = x0 - (psi(x0) - c) / (polygamma(1, x0))
    # return x0
    # same result as above, but use scipy API.

    def f(x): return psi(x) - c
    return (optimize.newton(f, x0, fprime=lambda x: polygamma(1, x),
                            fprime2=lambda x: polygamma(2, x), maxiter=max_iter))


def beta_mle(y, p0=1, a=0, b=1):
    """ Maximum Likelihood Estimation for the Beta Distribution. R. J. BECKMAN and G. L. TIETJEN
        J. Smtisr. Comprrt. Simul., 1978, Vol. 7, pp. 253-25.
        https://www.tandfonline.com/doi/pdf/10.1080/00949657808810232
        f ~ Beta(p, q, a, b): f(x) = gamma(p+q) * (x-a)^p-1 * (b-x)^q-1 / (gamma(p) * gamma(q))
    """
    n = y.size
    G1 = np.prod(np.power((y - a) / (b - a), 1 / n))
    G2 = np.prod(np.power((b - y) / (b - a), 1 / n))

    assert G1 + G2 < 1  # G1 + G2 > 1 is impossible, proved by
    # the order of operator not incur numerical error in this project.
    # _G1 = np.power(np.prod((y - a) / (b - a)), 1 / n)
    # _G2 = np.power(np.prod((b - y) / (b - a)), 1 / n)
    # assert np.isclose(G1, _G1)
    # assert np.isclose(G2, _G2)

    # assert np.isclose(np.log(G2) - np.log(G1), np.log(G2 / G1))

    def f(p):
        return psi(p) - np.log(G1) - psi(p + psi_inv(np.log(G2) - np.log(G1) + psi(p)))

    # p_hat = optimize.newton(f, x0=1) the same result as brentq
    p_hat = optimize.brentq(f, a=.005, b=6)
    q_hat = psi_inv(psi(p_hat) + np.log(G2) - np.log(G1))

    return p_hat, q_hat


if __name__ == '__main__':
    # test psi_inv, result: correct.
    # x = np.array([0.00000001, 0.001, 0.01, 0.1, 1, 2, 4, 6, 100, 10000000])
    # print([psi_inv(psi(i)) for i in x])

    # test beta mle, result: correct, pretty much stable.
    # y = np.random.beta(1, 1234, size=50)
    # p, q = beta_mle(y)
    # print(p, q)
    with open("huber_beta_est", "w") as out:
        out.write("\t".join(["GENE", "beta_loc", "beta_scale", "strongest.p.beta"]) + "\n")
        with open("huber_strongest_raw_p", "r") as f:
            f.readline()
            for line in f:
                rec = line.split(",")
                gene_name = rec[0]
                assert len(rec[2:]) == 50
                y = np.array(rec[2:], dtype=float)
                p, q = beta_mle(y)
                corrected = stats.beta.cdf(float(rec[1]), p, q)
                out.write(f"{gene_name}\t{p}\t{q}\t{corrected}" + "\n")
