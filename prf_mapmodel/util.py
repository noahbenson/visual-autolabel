# -*- coding: utf-8 -*-
################################################################################
# prf_mapmodel/util.py
# Utilities fo the prf_mapmodel library.

"""
The `prf_mapmodel.util` package contains utilities for use in and with the
`prf_mapmodel` library.
"""

def safesqrt(u, inplace=False):
    '''Calculates square root only for non-negative real values.

    `safesqrt(u)` is equivalent to `torch.sqrt(u)` except that it only operates
    on values that are greater than 0.

    Parameters
    ----------
    u : tensor
        The PyTorch tensor whose square root is to be taken.

    Returns
    -------
    tensor
        `sqrt(u)` for `u >= 0`, otherwise `u`.
    '''
    import torch
    if not inplace:
        u = u.clone()
    ii = u > 0
    u[ii] = torch.sqrt(u[ii])
    return u
def lbeta(a,b):
    '''Returns the log of the Beta function.

    `lbeta(a,b)` returns the log of the Beta function `beta(a,b)`. This is
    equivalent to `lgamma(a) + lgamma(b) - lgamma(a+b)` using the `lgamma`
    function from PyTorch.
    '''
    import torch
    return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a+b)
def beta_dist(mu, scale):
    '''Returns the PyTorch `Beta` object for the given mean and scale.

    `beta_dist(mu, scale)` returns the pytorch beta-distribution object(s) for
    the given mean `mu` and `scale`. The `mu` parameter must be between 0 and 1,
    and the `scale` parameter may be any real number.
      
    The traditional beta distribution uses parameters `a` and `b`. The
    reparameterizatoin of `mu` and `scale` here is as follows:
     * `a = (mu * (2 - b) - 1) / (mu - 1)`
     * `b = (2 - 1/mu) + exp(scale)`
    '''
    from torch.distributions import Beta
    b = torch.exp(scale)
    a = (mu * (b - 2) + 1) / (1 - mu)
    return Beta(a, b)
def beta_log_prob(mu, scale, x):
    '''Returns the log probability density function of a Beta distribution.

    `beta_log_prob(mu, scale, x)` returns the log probability density of the
    beta distribution parameterized using the mean `mu` (which must be between 0
    and 1) and `scale` (which may be any real number) at the value `x`.
      
    The traditional beta distribution uses parameters `a` and `b`. The
    reparameterizatoin of `mu` and `scale` here is as follows:
     * `a = (mu * (2 - b) - 1) / (mu - 1)`
     * `b = (2 - 1/mu) + exp(scale)`
    '''
    dist = beta_dist(mu, scale)
    return dist.log_prob(x)
def beta_pdf(mu, scale, x):
    '''Returns the probability density function of a Beta distribution.

    `beta_pdf(mu, scale, x)` returns the probability density function of the
    beta distribution parrameterized using the mean `mu` (which must be between
    0 and 1) and `scale` (which may be any real number) at the value `x`.
    
    The traditional beta distribution uses parameters `a` and `b`. The
    reparameterizatoin of mu and scale here is as follows:
     * `a = (mu * (2 - b) - 1) / (mu - 1)`
     * `b = (2 - 1/mu) + exp(scale)`
    '''
    import torch
    return torch.exp(beta_log_prob(mu, scale, x))
def gsigmoid(x, min=-1, max=1, method='logistic', inplace=False):
    """Returns any of several sigmoid functions.

    The optional parameter `method` allows one to specify the function used for
    rescaling `x`. This may be one of the following:
     * `'atan'`: the arc-tangent function
     * `'erf'`: the error-function
     * `'logistic'`: the sigmoid / logistic function
     * `'alg1'`: order-1 algebraic function (`y = x / (1 + abs(x))`)
     * `'alg2`': order-2 algebraic function (`y = x / sqrt(1 + x**2)`)
    
    Paramters
    ---------
    x : tensor
        The PyTorch tensor that is to be converted to a sigmoid.
    min : real, optional
        The minimum value `x` can be rescaled into (default: -1).
    max : real, optional
        The maximum value `x` can be rescaled into (default: 1).
    method : str, optional
        A string designating the sigmoid function to use for rescaling. See the
        table above for a list of methods.
    inplace : boolean, optional
        Whether to perform the limiting in-place or not (default: `False`).

    Returns
    -------
    tensor
        A PyTorch tensor of the sigmoid function of `x`.
    """
    import torch
    x = torch.as_tensor(x)
    u = x if inplace else torch.empty_like(x)
    method = method.lower()
    if method in ('atan', 'arctan', 'tan', 'tangent', 'arctangent'):
        u[:] = 0.5 + torch.atan(x)/np.pi
    elif method in ('erf', 'error', 'normal'):
        u[:] = 0.5 + torch.erf(x)/2
    elif method in ('sigmoid', 'logistic', 'sig'):
        u[:] = torch.sigmoid(x)
    elif method in ('alg1', 'algebraic1'):
        u[:] = x / (1 + torch.abs(x))
    elif method in ('alg2', 'algebraic2'):
        u[:] = x / torch.sqrt(1 + x**2)
    else:
        raise ValueError(f"misunderstood gsigmoid methood: {method}")
    return min + (max - min) * u
def glogit(p, min=-1, max=1, method='logistic', inplace=False):
    """Returns any of several logit (inverse-sigmoid) functions.

    The optional parameter `method` allows one to specify the function used for
    transforming `p`. The inversee of one of the following functions is used:
     * `'atan'` or `'tan'`: the arctangent function
     * `'erf'`: the error-function
     * `'logistic'`: the sigmoid / logistic function
     * `'alg1'`: order-1 algebraic function (`y = x / (1 + abs(x))`)
     * `'alg2`': order-2 algebraic function (`y = x / sqrt(1 + x**2)`)
    
    Paramters
    ---------
    param : tensor
        The PyTorch tensor that is to be 
    min : real, optional
        The minimum value param can be rescaled into (default: -1).
    max : real, optional
        The maximum value param can be rescaled into (default: 1).
    center : real, optional
        The center of the sigmoid function (default: 0).
    method : str, optional
        A string designating the sigmoid function to use for rescaling. See the
        table above for a list of methods.
    inplace : boolean, optional
        Whether to perform the limiting in-place or not (default: `False`).

    Returns
    -------
    tensor
        A PyTorch tensor of the sigmoid function of `x`.
    """
    import torch
    p = torch.as_tensor(p)
    u = p if inplace else p.clone()
    if min != 0:
        u -= min
        max -= min
    if max != 1:
        u /= max
    method = method.lower()
    if method in ('atan', 'arctan'):
        torch.tan((u - 0.5) * np.pi, out=u)
    elif method in ('erf', 'error', 'normal'):
        torch.erfinv((u - 0.5) * 2, out=u)
    elif method in ('sigmoid', 'logistic', 'sig'):
        torch.logit(u, out=u)
    elif method in ('alg1', 'algebraic1'):
        absu = torch.abs(u)
        u[:] = torch.sign(u) * absu / (1 - absu)
    elif method in ('alg2', 'algebraic2'):
        u[:] = torch.sign(u) * torch.abs(u) / torch.sqrt(1 - u**2)
    else:
        raise ValueError(f"misunderstood glogit methood: {method}")
    return u
def triarea(a,b,c):
    '''Returns the area of the triangle with sides of length `a`, `b`, and `c`.

    `triarea(a,b,c)` returns the area of the triangle whose sides have lengths
    `a`, `b`, and `c`.

    Parameters
    ----------
    a : tensor
        The length of the first side of the triangle.
    b : tensor
        The length of the second side of the triangle.
    c : tensor
        The length of the third side of the triangle.

    Returns
    -------
    real
        The surface area of the triangle with sides `a`, `b`, and `c`.
    '''
    hp = 0.5*(a + b + c)
    return hp * (hp - a) * (hp - b) * (hp - c)
def branch(iftensor, thentensor, elsetensor=None):
    """Returns a tensor of element-wise `if` evaluations.

    `branch(q, t, e)` returns, elementwise for the given tensors, 
    `t if q else e`.

    `branch(q, t)` or `branch(q, t, None)` is equivalent to `branch(q, t, 0)`.
    
    The output tensor will always have the same shape as q. The values for t
    and e may be constants or tensors the same shape as q.

    This function should be safe to use in optimization, i.e., with gradient
    calculatioins.
    """
    import torch
    q = torch.as_tensor(iftensor)
    t = torch.tensor(0.0) if thentensor is None else torch.as_tensor(thentensor)
    e = torch.tensor(0.0) if elsetensor is None else torch.as_tensor(elsetensor)
    x = t if e is None else e
    if x is None: raise ValueError('branch: both then and else cannot be None')
    r = torch.zeros(q.shape, dtype=x.dtype)
    if t is not None:
        if t.shape == (): r[q] = t
        else:             r[q] = t[q]
    if e is not None:
        q = ~q
        if e.shape == (): r[q] = e
        else:             r[q] = e[q]
    return r
def zinv(x, inplace=False):
    """Returns `1/x` if x is not equal to 0; otherwise returns 0.

    `zinv(x)` returns 0 if `x == 0` and `1/x` otherwise. This is done in a way
    that is safe for torch gradients; i.e., the gradient for any element of `x`
    that is equal to 0 will also be 0.
    """
    x = torch.as_tensor(x)
    ii = (x != 0)
    rr = x if inplace else torch.zeros_like(x)
    rr[ii] = 1 / x[ii]
    return rr
