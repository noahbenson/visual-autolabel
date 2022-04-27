# -*- coding: utf-8 -*-
################################################################################
# prf_mapmodel/cmag.py
# The cortical magnification module.

"""The cortical magnification module of the `prf_mapmodel` package.

The model of cortical magnification used here combines Horton and Hoyt's (1991)
model of cortical magnification in terms of eccentricity with a model of
tangential magnification asymmetry.

* `hva` is the horizontal-vertical asymmetry, which is the fraction of the mean
  cortical magnification at the horizontal and vertical meridians that the
  vertical meridian alone contributes to--usually this is about 0.5.
* `vma` is the vertical meridian asymmetry, which is the fraction of the mean
  cortical magnification at the vertical meridians that the upper vertical
  meridian alone contributes to--usually this is about 0.5.

Given these values, we can calculate the cortical magnification, in terms of the
rescaled polar angle ($\theta_r$), which represents the polar angle in
traditional radians (increasing counter-clockwise with the origin at the
positive x-axis) but has been rescaled such that +π represents π+`ui` and
-π represnts -π-`li`.

$$
\gamma(\mathbf{\theta_r}; \mathbf{hva}, \mathbf{vma}) = 
   1 + \frac{1}{2}\left(\mathbf{hva} \cos(2 \mathbf{\theta_r}) 
     - \mathbf{vma} \, \hbox{sgn}(\sin(\mathbf{\theta_r})) 
       \sin(\mathbf{\theta_r})^2 \right)
$$

With Horton and Hoyt's (1991) equation, this gives us the following
magnification in terms of corrected polar angle and eccentricity:

$$
m(\mathbf{\theta_r}, \mathbf{\rho}; c_1, c_2, \hbox{hva}, \hbox{vma}) =
   \gamma(\mathbf{\theta_r}; \hbox{hva}, \hbox{vma}) 
   \left( \frac{c_1}{c_2 + \mathbf{\rho}} \right)^2
$$

The integral of this equation over half of the visual field (i.e., the area of
the visual area) out to $M$ degrees of is:

$$ \hat{a}(M) = \int_{\theta_r=-\pi/2}^{\theta_r=pi/2} 
      \int_{\rho=0}^{\rho=M} \rho \, m(\theta_r, \rho) \, d\rho\,d\theta_r $$
$$ \hat{a}(M) = c_1^2 \pi \left(\log\left(\frac{c_2 + M}{c_2}\right)
      - \frac{M}{c_2 + M}\right) $$

This is, in fact, the integral for the Horton and Hoyt (1991) equation alone,
indicating that the tangential multiplier does not change the overall cortical
magnification of an individual visual area. Assuming that we know that the
actual (measued) surface area of the visual-area from 0 to `M` degrees of
eccentricity is $a_M$, we can fix the parameter $c_1$ to force the equation to
have a fixed total surface area:

$$ c_1 = \sqrt{\frac{a_M}{\pi \left(\log\left(\frac{c_2 + M}{c_2}\right)
         - \frac{M}{c_2 + M}\right)}} $$

The variable $M$ above is generally named as `max_eccen` in the code. It
represents is the maximum eccentricity value assumed to be included in V1, $M$
(generally kept as 90°).
"""


#===============================================================================
# Constants / Globals

#-------------------------------------------------------------------------------
# The Horton and Hoyt (1991) constants.
# Using the equation: m[mm^2/deg^2] = (c1[mm] / (0.75[deg] + c2[deg]))^2

HH1991_c1 = 17.3
HH1991_c2 = 0.75

#-------------------------------------------------------------------------------
# The Benson et al. (2021) VMA/HVA constants.
# Note that these are rough estimates of the asymmetry, not precise values.

Benson2021_VMA = 0.5
Benson2021_HVA = 0.5

#-------------------------------------------------------------------------------
# The Gibaldi et al. (2021) ipsilateral-invasion estimates.
# These are also approximate.

Gibaldi2021_upper_ipsi = 0.09
Gibaldi2021_lower_ipsi = 0.17


#===============================================================================
# The cortical magnification model functions.

def cmmdl_angle_multiplier(theta,
                           ui=Gibaldi2021_upper_ipsi,
                           li=Gibaldi2021_lower_ipsi, 
                           hva=Benson2021_HVA,
                           vma=Benson2021_VMA):
    '''Returns a multiplier for the cortical magnification based polar angle.

    `cmmdl_angle_multiplier(theta)` returns the angular cortical magnification
    multiplier based on the polar angle `theta`, which is measured in
    counter-clockwise degrees from the right HM.

    Parameters
    ----------
    theta : tensor
        The polar angle (or angles) in radians of counter-clockwise rotation
        from the positive x-axis.
    ui : real, optional
        The upper visual field ipsilateral representation, in radians. Negative
        values indicate an under-shoot of the upper vertical meridian. The
        default value is `Gibaldi2021_upper_ipsi` (0.09).
    li : real, optional
        The lower visual field ipsilateral representation, in radians. Negative
        values indicate an under-shoot of the lower vertical meridian. The
        default value is `Gibaldi2021_lower_ipsi` (0.17).
    hva : real, optional
        The Horizontal-Vertical Asymmetry (HVA); this is the ratio of the
        cortical magnification at the vertical meridian to the cortical
        magnification at the horizontal meridian. The default value is
        `Benson2021_HVA` (0.5).
    vma : real, optional
        The Vertical Meridian Asymmetry (VMA); this is the ratio of the
        cortical magnification at the upper meridian to the cortical
        magnification at the vertical meridian. The default value is
        `Benson2021_VMA` (0.5).

    Returns
    -------
    real
        The multiplier for the cortical magnification at the given polar agnle.
    '''
    from numpy import pi
    from torch import zeros, sin, cos, sign
    hpi = pi/2
    th = zeros(theta.shape)
    gt = theta > 0
    lt = ~gt
    th[gt] = theta[gt] / (hpi + ui) * hpi
    th[lt] = theta[lt] / (hpi + li) * hpi
    hvpart = hva * cos(2 * th)
    thsin  = sin(th)
    ulpart = vma * torch.sign(thsin) * thsin**2
    return 1.0 + 0.5*(hvpart - ulpart)
def cmmdl_hhcmag(eccen,
                 c1=HH91_c1, 
                 c2=HH91_c2):
    '''Returns the Horton & Hoyt prediction of linear cortical magnification.

    `cmmdl_hhcmag(eccen)` returns the linear radial cortical magnification using
    the Horton and Hoyt (1991) formula, `m = c1 / (c2 + eccen)`, where `c1` is
    in mm and `c2` and `eccen` are in degrees of the visual field.

    Parameters
    ----------
    eccen : real
        The eccentricity, in degrees of the visual field.
    c1 : real, optional
        The parameteer `c1` from the Horton and Hoyt equation (see above), in
        mm. The default is the value reported in the paper, `17.3` mm.
    c2 : real, optional
        The parameteer `c2` from the Horton and Hoyt equation (see above), in
        degrees of the visual field. The default is the value reported in the
        paper, `0.75` deg.

    Returns
    -------
    real
        The linear cortical magnification on mm/deg, equal to
        `c1 / (c2 + eccen)`.
    '''
    return c1 / (c2 + eccen)
def cmmdl_hhcmag2(eccen,
                 c1=HH91_c1, 
                 c2=HH91_c2):
    '''Returns the Horton & Hoyt prediction of areal cortical magnification.

    `cmmdl_hhcmag2(eccen)` returns the areal radial cortical magnification using
    the Horton and Hoyt (1991) formula, `M = (c1 / (c2 + eccen))**2`, where `c1`
    is in mm and `c2` and `eccen` are in degrees of the visual field.

    Parameters
    ----------
    eccen : real
        The eccentricity, in degrees of the visual field.
    c1 : real, optional
        The parameteer `c1` from the Horton and Hoyt equation (see above), in
        mm. The default is the value reported in the paper, `17.3` mm.
    c2 : real, optional
        The parameteer `c2` from the Horton and Hoyt equation (see above), in
        degrees of the visual field. The default is the value reported in the
        paper, `0.75` deg.

    Returns
    -------
    real
        The areal cortical magnification on square-mm/square-deg, equal to
        `(c1 / (c2 + eccen))**2`.
    '''
    return cmmdl_hhcmag(eccen)**2
def cmmdl_cmag2(theta, eccen,
                c1=HH91_c1, 
                c2=HH91_c2,
                ui=Gibaldi2021_upper_ipsi,
                li=Gibaldi2021_lower_ipsi, 
                hva=Benson2021_HVA,
                vma=Benson2021_VMA):
    '''Returns the areal cortical magnification of a point in the visual field.

    `cmmdl_cmag2(theta, eccen)` returns the areal cortical magnification
    prediction for the given polar angle and eccentricity using both the Horton
    and Hoyt (1991) equation as well as the angular cortical magnification
    multiplier.

    Parameters
    ----------
    theta : tensor
        The polar angle (or angles) in radians of counter-clockwise rotation
        from the positive x-axis.
    eccen : real
        The eccentricity, in degrees of the visual field.
    c1 : real, optional
        The parameteer `c1` from the Horton and Hoyt equation (see above), in
        mm. The default is the value reported in the paper, `17.3` mm.
    c2 : real, optional
        The parameteer `c2` from the Horton and Hoyt equation (see above), in
        degrees of the visual field. The default is the value reported in the
        paper, `0.75` deg.
    ui : real, optional
        The upper visual field ipsilateral representation, in radians. Negative
        values indicate an under-shoot of the upper vertical meridian. The
        default value is `Gibaldi2021_upper_ipsi` (0.09).
    li : real, optional
        The lower visual field ipsilateral representation, in radians. Negative
        values indicate an under-shoot of the lower vertical meridian. The
        default value is `Gibaldi2021_lower_ipsi` (0.17).
    hva : real, optional
        The Horizontal-Vertical Asymmetry (HVA); this is the ratio of the
        cortical magnification at the vertical meridian to the cortical
        magnification at the horizontal meridian. The default value is
        `Benson2021_HVA` (0.5).
    vma : real, optional
        The Vertical Meridian Asymmetry (VMA); this is the ratio of the
        cortical magnification at the upper meridian to the cortical
        magnification at the vertical meridian. The default value is
        `Benson2021_VMA` (0.5).

    Returns
    -------
    real
        The predicted areal cortical magnification in square-mm/square-deg.
    '''
    rmag = cmmdl_hhcmag2(eccen, c1=c1, c2=c2)
    tmlt = cmmdl_angle_multiplier(theta, ui=ui, li=li, hva=hva, vma=vma)
    return rmag * tmlt
def cmmdl_c1(area, max_eccen, c2=HH92_c2):
    '''Returns the c1 parameter for a given surfaec area.

    `cmmdl_c1(area, max_eccen)` returns the value of `c1` in the cortical
    magnification model of Horton and Hoyt (1991) that is appropriate for a
    visual area with the given surface-area, limited to the given `max_eccen`
    central degrees.

    Parameters
    ----------
    area : non-negative real
        The surface area of the visual area limited to `max_eccen` degrees.
    max_eccecn : non-negative real
        The maximum eccentricity, in degrees of the visual field, that is
        included in `area`.
    c2 : real, optional
        The parameteer `c2` from the Horton and Hoyt equation (see above), in
        degrees of the visual field. The default is the value reported in the
        paper, `0.75` deg.

    Returns
    -------
    real
        The appropriate parameter `c1` for the given surface area, limited to
        the given maximum eccentricity, and the given `c2` parameter.
    '''
    from numpy import pi
    from torch import log, sqrt
    c2_maxecc = c2 + max_eccen
    den = pi * (log(c2_maxecc / c2) - max_eccen / c2_maxecc)
    return sqrt(area / den)
