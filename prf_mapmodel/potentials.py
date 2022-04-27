# -*- coding: utf-8 -*-
################################################################################
# prf_mapmodel/util.py
# Utilities fo the prf_mapmodel library.

"""
The `prf_mapmodel.potentials` package contains functions that implement various
potential wells and steps for use in optimization by the `prf_mapmodel` library.
"""


#===============================================================================
# Constants / Globals

# We import this just for the constant declarations, then delete it to keep the
# namespace clean.
import numpy as np

#-------------------------------------------------------------------------------
# Step-function constants

standard_well_area = 0.682689
sin_step_width0 = 2.0 * np.arcsin(standard_well_area) / np.pi
cauchy_step_width0 = np.tan(np.pi/2 * standard_well_area)
normal_step_width0 = 1.0/np.sqrt(2.0)
logistic_step_width0 = np.log(-(standard_well_area+1) / (standard_well_area-1))

#-------------------------------------------------------------------------------
# Well-function constants.

normal_hwhm = np.sqrt(np.log(4.0))
sin_well_width0 = 0.5 / normal_hwhm
normal_well_width0 = 1.0
cauchy_well_width0 = 1.0 / normal_hwhm
logistic_well_width0 = np.log(3 + 2*np.sqrt(2.0)) / (2 * normal_hwhm)

# Clean the namespace.
del np


#===============================================================================
# The Potential Functions.

#-------------------------------------------------------------------------------
# The Step Functions.
# Step functions are sigmoid (or sigmoid-like) functions that make a step
# from a min value to a max value around some center value.

def sin_step(t, width=1, min=-1, max=1, center=0):
    '''A sine step function whose derivative is 0 outside of a finite range.

    `sin_step(t)` returns a sine-based step such that:
      * for `t < -w0`, `sin_step(t) = 0`; 
      * for `t > w0`, `sin_step(t) = 1`;
      * otherwise `sin_step(t) = 1/2 (1 + sin(pi/2 w0 t))`.
    
    `sin_step(t, width)` uses the given width; equivalent to
    `sin_step(t/width)`.

    `sin_step(t, width, min, max)` uses the given min and max values instead of
    0 and 1.

    `sin_step(t, width, min, max, center)` centers the distribution at the given
    center value.
      
    The value `w0` is based on the value `s = 0.682689`: approximately the
    fraction of the normal distribution within 1 standard deviation. The actual
    value of `w0` is `2 * arcsin(s) / pi`; this particular value aligns the
    `sin_step` with the `normal_step` in that it ensures that ~68% of the
    distribution for which `sin_step` is the CDF is within `-1` to `+1`.
    
    The `sin_well` is analogous to the `sin_step`; though the two are not
    actually representative of the same distribution, as is the case with other
    `_well` and `_step` functions.
    '''
    from torch import as_tensor, sin
    t = as_tensor(t)
    if center != 0: t = t - center
    t = t * (sin_step_width0 / width)
    step = 0.5*(1 + sin(np.pi/2*t))
    step[t >  1] = 1
    step[t < -1] = 0
    return min + (max - min)*step
def cauchy_step(t, width=1, min=0, max=1, center=0):
    '''A step function based on the CDF of the Cauchy distribution.

    `cauchy_step(t)` returns `(pi + atan(pi/2 w0 t)) / (2 pi)`.

    `cauchy_step(t, width)` uses the given width parameter; equivalent to
    `cauchy_step(t/width)`.

    `cauchy_step(t, width, min, max)` use the given min and max values;
    equivalent to `cauchy_step(t, width) * (max - min) + min`.
      
    The value `w0` is based on the value `s = 0.682689`: approximately the fraction
    of the normal distribution within 1 standard deviation. The actual value of
    `w0` is `tan(pi/2 s)`; this particular value aligns the `cachy_step` with the
    `normal_step` in that it ensures that ~68% of the distribution for which
    `cauchy_step` is the CDF is within `-1` to `+1`.
    
    The `cauchy_well` is related to the `cauchy_step` in that the well uses the
    CDF of the Cauchy distribution while the step uses the PDF.
    '''
    from numpy import pi
    from torch import as_tensor, atan
    t = as_tensor(t) - center
    const = cauchy_step_width0 / width
    u = 0.5 + atan(t * const) / pi
    return u*(max - min) + min
def normal_step(t, width=1, min=0, max=1, center=0):
    '''A step function based on the CDF of the normal distribution.

    `normal_step(t)` returns `(1 + erf(t/w0))/2`.
    
    `normal_step(t, width)` uses the given width parameter; equivalent to
    `normal_step(t/width)`.

    `normal_step(t, width, min, max)` use the given min and max values, 
    equivalent to `normal_step(t, width) * (max - min) + min`.
    
    The value `w0` is based on the value `s = 0.682689`: approximately the fraction
    of the normal distribution within 1 standard deviation. The actual value of
    `w0` is `1/sqrt(2)`; this particular value ensures that 68% of the distribution
    for which normal_step is the CDF (i.e., a normal distribution) is within `-1` to
    `+1`.
    
    The `normal_step` is related to the `normal_well` in that they are composed
    using the CDF and the PDF of the normal distribution, respectively..
    '''
    from torch import as_tensor, erf
    t = as_tensor(t) - center
    t *= normal_step_width0 / width
    u = 0.5 + 0.5*erf(t)
    return u * (max - min) + min
def logistic_step(t, width=1, min=0, max=1, center=0):
    '''A step function based on the CDF of the logistic distrivbution.

    `logistic_step(t)` returns `1/(1 + exp(-w0 t)) = 1/2 * (1 + tanh(w0 t/2))`.

    `logistic_step(t, width)` uses the given width parameter; equivalent to
    `logistic_step(t/width)`

    `logistic_step(t, width, min, max)` use the given min and max values;
    equivalent to `logistic_step(t, width) * (max - min) + min`.
    
    The value `w0` is based on the value `s = 0.682689`: approximately the
    fraction of the normal distribution within 1 standard deviation. The actual
    value of `w0` is `log(-(s+1)/(s-1))`; this particular value aligns the
    `logistic_step` with the `normal_step` in that it ensures that ~68% of the
    distribution for which `logistic_step` is the CDF is within `-1` to `+1`.
    
    The `logistic_step` is related to the logistic_well in that they are
    composed of the CDF and PDF of the logistic distribution, respectively.
    '''
    from torch import as_tensor, exp
    t = as_tensor(t) - center
    t *= logistic_step_width0 / width
    u = 1/(1 + exp(-t))
    return u * (max - min) + min

#-------------------------------------------------------------------------------
# The Well Functions.
# Well functions are functions that form single dips, like the negative of most
# probability density functions (PDFs).

def sin_well(t, width=1, min=0, max=1, center=0):
    '''A well function based on a sine-wave.

    `sin_well(t)` returns a sine-based well function such that:
      * for `t < -1/w0`, `sin_well(t) = 1`;
      * for `-1/w0 <= t < 1/w0`, `sin_well(t) = 1/2 (1 - cos(pi/2 w0 t))`;
      * for `1/20 <= t`, `sin_well(t) = 1`.

    `sin_well(t, width)` uses the given width; equivalent to
    `sin_well(width*t)`; the default value is 1.

    `sin_well(t, width, min, max)` is equivalent to `min + (max - min) *
    sin_well(t, width)`.
       
    The value `w0` is chosen to make the sine-well approximately similar to a
    normal (Gassian) well in that the half-maximum point of both wells will be
    at the value `log(sqrt(4))`, which is where the half-way point occurs for a
    normal distribution. For a `sin_well`, this value of `w0` is `1 / (2
    log(sqrt(4)))`.
       
    `sin_well(t)` is equivalent to
    `sin_well(t, width=1, min=0, max=1, center=0)`.
    '''
    from numpy import pi
    from torch import as_tensor, cos
    t = as_tensor(t) - center
    t *= sin_well_width0 / width
    well = 0.5 - cos(pi * t)/2
    well[(t < -1) | (t >= 1)] = 1
    return well * (max - min) + min
def normal_well(t, width=1, min=0, max=1, center=0, base=0):
    '''A well function based on the PDF of the normal distribution.

    `normal_well(t)` returns `1 - exp(-(t^2)/2)`.

    `normal_well(t, width)` is equivalent to `normal_well(t / width)`.

    `normal_well(t, width, min, max)` is equivalent to
    `normal_well(t, width) * (max - min) + min`.
    '''
    from torch import as_tensor, exp
    t = (as_tensor(t) - center)
    t *= normal_well_width0 / width
    u = 1 - exp(-0.5 * t**2)
    return u * (max - min) + min
def cauchy_well(t, width=1, min=0, max=1, center=0):
    '''A well function based on the PDF of the Cauchy distribution.

    `caucht_well(t)` returns `1 - 1/(1 + (w0 t)^2)`.

    `cauchy_well(t, width)` is equivalent to `cauchy_well(t / width)`.

    `cauchy_well(t, width, min, max)` is equivalent to
    `cauchy_well(t, width) * (max - min) + min`.
      
    The value `w0` is chosen to make the cauchy-well approximately similar to a
    normal (Gaussian) well in that the half-maximum point of both wells will be
    at the value `log(sqrt(4))`, which is where the half-way point occurs for a
    normal distribution. For a `cauchy_well`, this value of `w0` is `1 /
    log(sqrt(4))`.
       
    `cauchy_well(t)` is equivalent to
    `cauchy_well(t, width=1, min=0, max=1, center=0)`.
    '''
    from torch import as_tensor
    t = (as_tensor(t) - center)
    t *= cauchy_well_width0 / width
    u = 1 - 1/(1 + t**2)
    return min + (max - min) * u
def logistic_well(t, width=1, min=0, max=1, center=0):
    '''A well function baseed on the PDF of the logistic distribution.

    `logistic_well(t)` returns `1 - 4 exp(-t)/(1 + exp(-t))`.

    `logistic_well(t, width)` is equivalent to `logistic_well(t / width)`.

    `logistic_well(t, width, min, max)` is equivalent to
    `logistic_well(t, width) * (max - min) + min`.
      
    The value `w0` is chosen to make the logistic-well approximately similar to
    a normal (Gaussian) well in that the half-maximum point of both wells will
    be at the value `log(sqrt(4))`, which is where the half-way point occurs for
    a normal distribution. For a `logistic_well`, this value of `w0` is
    `log(3 + 2 sqrt(2)) / (2 log(sqt(4)))`.
       
    `logistic_well(t)` is equivalent to
    `logistic_well(t, width=1, min=0, max=1, center=0)`.
    '''
    from torch import as_tensor, exp
    t = (as_tensor(t) - center)
    t *= logistic_well_width0 / width
    et = exp(-t)
    u = 1 - 4*et/(1 + et)**2
    return u * (max - min) + min
