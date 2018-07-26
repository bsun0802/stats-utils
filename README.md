# stats-utils
This module contains utils functions(pure python) for statistical modeling and numerical inference.

## Beta-distribution MLE
`beta_mle.py` implements
 * ![psi_inv](https://latex.codecogs.com/gif.latex?%5Cpsi%5E%7B-1%7D%28x%29).  
   - The inverse of Psi(x), where Psi(x) is the derivative of the logrithm of gamma function.  
   - The inverse of Psi(x) is the root of:  ![psi_inv2](https://latex.codecogs.com/gif.latex?%5Cpsi%28x%29-c%26%3D0),
 which is solved with Newton's method.
   - The choice of initial guess was described in _Appendix c of Minka, T. P. (2003). Estimating a Dirichlet distribution.
        Annals of Physics_, 2000(8), 1-13. https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf
 * beta MLE estimator for location and scale parameters.
   - ![beta_pdf](https://latex.codecogs.com/gif.latex?X%20%5Csim%20%5Cbeta%28p%2C%20q%29%2C%20f%28x%29%20%3D%20%5Cfrac%7B%5Cgamma%7B%28p&plus;q%29x%5E%7Bp-1%7D%281-x%29%5E%7Bq%7D%7D%7D%7B%5Cgamma%7B%28p%29%7D&plus;%5Cgamma%7B%28q%29%7D%7D)
   - The implementation is based on _Maximum Likelihood Estimation for the Beta Distribution._ R. J. BECKMAN and G. L. TIETJEN
        J. Smtisr. Comprrt. Simul., 1978, Vol. 7, pp. 253-25. https://www.tandfonline.com/doi/pdf/10.1080/00949657808810232
