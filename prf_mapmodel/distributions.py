# -*- coding: utf-8 -*-
################################################################################
# prf_mapmodel/util.py
# Utilities fo the prf_mapmodel library.

"""
The `prf_mapmodel.distributions` package contains functions that implement the
log-probability functions for various distributions that can be used in
optimization with the `prf_mapmodel` library.
"""

#===============================================================================
# Constants / Globals

from torch import log, tensor
from numpy import pi
normal_offset = -0.5 * log(tensor(2.0 * pi))
del log, tensor, pi


#===============================================================================
# The log-probability functions.

def normal_logpdf(t, width=1, center=0):
    '''The log of the probability density function of the normal distribution.

    `normal_logpdf(t)` returns the log-probability of the normal 
    distribution: `-(t^2 + log(2 pi)) / 2`.

    `normal_logpdf(t, w)` is equivalent to `normal_logpdf(t)` with a
    standard deviation parameter of `w`; equal to
    `noremal_logpdf(t/w) - log(w)`.

    `normal_logpdf(t, w, center)` is equivalent to 
    `normal_logpdf(t - center, w)`.
    '''
    from torch import as_tensor, log
    t = (as_tensor(t) - center)
    width = as_tensor(width)
    t /= width
    return -0.5 * t**2 + normal_offset - log(width)
def cauchy_logpdf(t, width=1, center=0):
    '''The log of the probability density function of the Cauchy distribution.

    `cauchy_logpdf(t)` returns the log-probability of the Cauchy
    distribution: `-log(pi (1 + t^2))`.

    `cauchy_logpdf(t, w)` is equivalent to `cauchy_logpdf(t)` with a width
    argument of `w`; equal to: `-log(pi w (1 + (t/w)^2))`.

    `cauchy_logpdf(t, w, center)` is equivalent to
    `cauchy_logpdf(t - center, w)`.
    '''
    from numpy import pi
    from torch import as_tensor, log
    t = (as_tensor(t) - center)
    t /= width
    return -log(pi * width * (t**2 + 1))
def hcauchy_logpdf(t, width=1):
    '''The log of the probability density func. of the half-Cauchy distribution.

    `hcauchy_logpdf(t)` returns the log-probability of the half-Cauchy
    distribution: `-log(pi/2 (1 + t^2))`.

    `hcauchy_logpdf(t, w)` is equivalent to `hcauchy_logpdf(t)` with a width
    argument of `w`; equal to: `-log(pi/2 w (1 + (t/w)^2))`.
      
    Note that although the half-Cauchy distribution is only defined on the
    positive real numbers, this function will return a symmetric set of values
    for the negative real numbers.
    '''
    from numpy import pi
    from torch import as_tensor, log
    t = as_tensor(t) / width
    return -log(pi/2 * width * (t**2 + 1))
def laplace_logpdf(t, width=1, center=0):
    '''The log of the probability density function of the Laplace distribution.

    `laplace_logpdf(t)` returns the log-probability of the Laplace
    distribution: `-(|t| + log(2))`.

    `laplace_logpdf(t, w)` is equivalent to `laplace_logpdf(t)` with a scale
    parameter of `1/w`; equal to `-(|t|/w + log(2 w))`.

    `laplace_logpdf(t, w, center)` is equivalent to
    `laplace_logpdf(t - center, w)`.
    '''
    from torch import as_tensor, log, abs
    t = (as_tensor(t) - center)
    width = as_tensor(width)
    t /= width
    return -abs(t)/width - log(2*width)
def exp_logpdf(t, width=1, center=0):
    '''The log of the probability density func. of the exponential distribution.

    `exp_logpdf(t)` returns the log-probability of the exponential
    distribution: `-t` for `t >= 0` else negative infinity.

    `exp_logpdf(t, w)` is equivalent to `exp_logpdf(t)` with a width
    argument of `1/w`; equal to: `-(t/w + log(w))` if `t > 0` else negative
    infinity.

    `exp_logpdf(t, w, center)` is equivalent to `exp_logpdf(t - center, w)`.
    '''
    from numpy import inf
    from torch import as_tensor, log
    t = (as_tensor(t) - center)
    width = as_tensor(width)
    t /= width
    return branch(t > 0, -t/width - log(width), -inf)
def generr_logpdf(q, t, width=1, center=0):
    '''The log of the probability density function of the generalized error distribution.

    `generr_logpdf(q, t)` returns the log-probability of the generalized error
    distribution: `-|t|^q + log(q/2)`.

    `generr_logpdf(q, t, w)` is equivalent to `generr_logpdf(q, t)` with a
    width argument of `w`; equal to: `-|t/w|^q + log(q/(2 w gamma(1/q)))`.

    `generr_logpdf(q, t, w, c)` is equivalent to
    `generr_logpdf(q, t - c, w)`.
    '''
    from torch import as_tensor, log, abs, lgamma
    t = (as_tensor(t) - center)
    t /= width
    return -abs(t)**q + log(0.5 * q / width) - lgamma(1/q)
def gumbel_logpdf(t, lw=1, rw=1, center=0):
    '''The log of the probability density function of the Gumbel distribution.

    `gumbel_logpdf(t)` returns the log-probability of the standard Gumbel
    distribution: `-(t + exp(-t))`.

    `gumbel_logpdf(t, lw)` uses the `lw` for the width of the left-hand side of
    the Gumbel distribution; i.e.: `-(t + exp(-t)/lw)`.

    `gumbel_logpdf(t, lw, rw)` uses the `lw` and `rw` for the widths of the
    left-hand and right-hand side of the Gumbel distribution, respectively;
    i.e.: `-(t/lr + exp(-t)/lw)`.
      
    Note that this is not a typical Gumbel distribution definition, but it is a
    very similar distribution nonetheless. Whereas the Gumbel PDF is usually
    defined as being proportional to `exp(-((t-t0)/w + exp(-(t-t0)/w)))`, this
    particular parameterization allows for the linear part of the exponential
    `B(T) = t` to be scaled separately from the exponential part of the
    exponential `A(t) = exp(-t)`. In a traditional Gumbel distribution,
    the `PDF(t) = exp(-[A([t-t0]/w) + B([t-t0]/w)])`. In the parameterization
    here, `PDF(t) = exp(-[A([t-t0]/a) + B([t-t0]/b)])`.
    '''
    from torch import as_tensor, log, exp, lgamma
    t = (as_tensor(t) - center)
    lw = as_tensor(lw)
    rw = as_tensor(rw)
    const = lw*log(rw/lw) + rw*(log(lw) + lgamma(lw/rw))
    return (lw*exp(-t/lw) + t + const) / -rw
