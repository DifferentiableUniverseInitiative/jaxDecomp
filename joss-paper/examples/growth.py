import jax.numpy as np
from jax_cosmo.background import *
from jax_cosmo.scipy.interpolate import interp
from jax_cosmo.scipy.ode import odeint

# Taken from jaxpm github/DifferentiableUniverseInitiative/JaxPM


def E(cosmo, a):
  r"""Scale factor dependent factor E(a) in the Hubble
    parameter.
    Parameters
    ----------
    a : array_like
        Scale factor
    Returns
    -------
    E : ndarray, or float if input scalar
        Square of the scaling of the Hubble constant as a function of
        scale factor
    Notes
    -----
    The Hubble parameter at scale factor `a` is given by
    :math:`H^2(a) = E^2(a) H_o^2` where :math:`E^2` is obtained through
    Friedman's Equation (see :cite:`2005:Percival`) :
    .. math::
        E^2(a) = \Omega_m a^{-3} + \Omega_k a^{-2} + \Omega_{de} a^{f(a)}
    where :math:`f(a)` is the Dark Energy evolution parameter computed
    by :py:meth:`.f_de`.
    """
  return np.power(Esqr(cosmo, a), 0.5)


def df_de(cosmo, a, epsilon=1e-5):
  r"""Derivative of the evolution parameter for the Dark Energy density
    f(a) with respect to the scale factor.
    Parameters
    ----------
    cosmo: Cosmology
    Cosmological parameters structure
    a : array_like
    Scale factor
    epsilon: float value
    Small number to make sure we are not dividing by 0 and avoid a singularity
    Returns
    -------
    df(a)/da : ndarray, or float if input scalar
    Derivative of the evolution parameter for the Dark Energy density
    with respect to the scale factor.
    Notes
    -----
    The expression for :math:`\frac{df(a)}{da}` is:
    .. math::
    \frac{df}{da}(a) = =\frac{3w_a \left( \ln(a-\epsilon)-
    \frac{a-1}{a-\epsilon}\right)}{\ln^2(a-\epsilon)}
    """
  return (3 * cosmo.wa * (np.log(a - epsilon) - (a - 1) / (a - epsilon)) /
          np.power(np.log(a - epsilon), 2))


def dEa(cosmo, a):
  r"""Derivative of the scale factor dependent factor E(a) in the Hubble
    parameter.
    Parameters
    ----------
    a : array_like
        Scale factor
    Returns
    -------
    dE(a)/da : ndarray, or float if input scalar
        Derivative of the scale factor dependent factor in the Hubble
      parameter with respect to the scale factor.
    Notes
    -----
    The expression for :math:`\frac{dE}{da}` is:
    .. math::
        \frac{dE(a)}{da}=\frac{-3a^{-4}\Omega_{0m}
        -2a^{-3}\Omega_{0k}
        +f'_{de}\Omega_{0de}a^{f_{de}(a)}}{2E(a)}
    Notes
    -----
    The Hubble parameter at scale factor `a` is given by
    :math:`H^2(a) = E^2(a) H_o^2` where :math:`E^2` is obtained through
    Friedman's Equation (see :cite:`2005:Percival`) :
    .. math::
        E^2(a) = \Omega_m a^{-3} + \Omega_k a^{-2} + \Omega_{de} a^{f(a)}
    where :math:`f(a)` is the Dark Energy evolution parameter computed
    by :py:meth:`.f_de`.
    """
  return (0.5 *
          (-3 * cosmo.Omega_m * np.power(a, -4) -
           2 * cosmo.Omega_k * np.power(a, -3) +
           df_de(cosmo, a) * cosmo.Omega_de * np.power(a, f_de(cosmo, a))) /
          np.power(Esqr(cosmo, a), 0.5))


def growth_factor(cosmo, a):
  """Compute linear growth factor D(a) at a given scale factor,
    normalized such that D(a=1) = 1.

    Parameters
    ----------
    cosmo: `Cosmology`
      Cosmology object

    a: array_like
      Scale factor

    Returns
    -------
    D:  ndarray, or float if input scalar
        Growth factor computed at requested scale factor

    Notes
    -----
    The growth computation will depend on the cosmology parametrization, for
    instance if the $\gamma$ parameter is defined, the growth will be computed
    assuming the $f = \Omega^\gamma$ growth rate, otherwise the usual ODE for
    growth will be solved.
    """
  if cosmo._flags["gamma_growth"]:
    print(f"here")
    return _growth_factor_gamma(cosmo, a)
  else:
    print(f"THERE")
    return _growth_factor_ODE(cosmo, a)


def growth_factor_second(cosmo, a):
  """Compute second order growth factor D2(a) at a given scale factor,
    normalized such that D(a=1) = 1.

    Parameters
    ----------
    cosmo: `Cosmology`
      Cosmology object

    a: array_like
      Scale factor

    Returns
    -------
    D2:  ndarray, or float if input scalar
        Growth factor computed at requested scale factor

    Notes
    -----
    The growth computation will depend on the cosmology parametrization,
    as for the linear growth. Currently the second order growth
    factor is not implemented with $\gamma$ parameter.
    """
  if cosmo._flags["gamma_growth"]:
    raise NotImplementedError(
        "Gamma growth rate is not implemented for second order growth!")
    return None
  else:
    return _growth_factor_second_ODE(cosmo, a)


def growth_rate(cosmo, a):
  """Compute growth rate dD/dlna at a given scale factor.

    Parameters
    ----------
    cosmo: `Cosmology`
      Cosmology object

    a: array_like
      Scale factor

    Returns
    -------
    f:  ndarray, or float if input scalar
        Growth rate computed at requested scale factor

    Notes
    -----
    The growth computation will depend on the cosmology parametrization, for
    instance if the $\gamma$ parameter is defined, the growth will be computed
    assuming the $f = \Omega^\gamma$ growth rate, otherwise the usual ODE for
    growth will be solved.

    The LCDM approximation to the growth rate :math:`f_{\gamma}(a)` is given by:

    .. math::

        f_{\gamma}(a) = \Omega_m^{\gamma} (a)

     with :math: `\gamma` in LCDM, given approximately by:
     .. math::

        \gamma = 0.55

    see :cite:`2019:Euclid Preparation VII, eqn.32`
    """
  if cosmo._flags["gamma_growth"]:
    return _growth_rate_gamma(cosmo, a)
  else:
    return _growth_rate_ODE(cosmo, a)


def growth_rate_second(cosmo, a):
  """Compute second order growth rate dD2/dlna at a given scale factor.

    Parameters
    ----------
    cosmo: `Cosmology`
      Cosmology object

    a: array_like
      Scale factor

    Returns
    -------
    f2:  ndarray, or float if input scalar
        Second order growth rate computed at requested scale factor

    Notes
    -----
    The growth computation will depend on the cosmology parametrization,
    as for the linear growth rate. Currently the second order growth
    rate is not implemented with $\gamma$ parameter.
    """
  if cosmo._flags["gamma_growth"]:
    raise NotImplementedError(
        "Gamma growth factor is not implemented for second order growth!")
    return None
  else:
    return _growth_rate_second_ODE(cosmo, a)


def _growth_factor_ODE(cosmo, a, log10_amin=-3, steps=128, eps=1e-4):
  """Compute linear growth factor D(a) at a given scale factor,
    normalised such that D(a=1) = 1.

    Parameters
    ----------
    a: array_like
      Scale factor

    amin: float
      Mininum scale factor, default 1e-3

    Returns
    -------
    D:  ndarray, or float if input scalar
        Growth factor computed at requested scale factor
    """
  # Check if growth has already been computed
  if not "background.growth_factor" in cosmo._workspace.keys():
    # Compute tabulated array
    atab = np.logspace(log10_amin, 0.0, steps)

    def D_derivs(y, x):
      q = (2.0 - 0.5 * (Omega_m_a(cosmo, x) +
                        (1.0 + 3.0 * w(cosmo, x)) * Omega_de_a(cosmo, x))) / x
      r = 1.5 * Omega_m_a(cosmo, x) / x / x

      g1, g2 = y[0]
      f1, f2 = y[1]
      dy1da = [f1, -q * f1 + r * g1]
      dy2da = [f2, -q * f2 + r * g2 - r * g1**2]
      return np.array([[dy1da[0], dy2da[0]], [dy1da[1], dy2da[1]]])

    y0 = np.array([[atab[0], -3.0 / 7 * atab[0]**2], [1.0, -6.0 / 7 * atab[0]]])
    y = odeint(D_derivs, y0, atab)

    # compute second order derivatives growth
    dyda2 = D_derivs(np.transpose(y, (1, 2, 0)), atab)
    dyda2 = np.transpose(dyda2, (2, 0, 1))

    # Normalize results
    y1 = y[:, 0, 0]
    gtab = y1 / y1[-1]
    y2 = y[:, 0, 1]
    g2tab = y2 / y2[-1]
    # To transform from dD/da to dlnD/dlna: dlnD/dlna = a / D dD/da
    ftab = y[:, 1, 0] / y1[-1] * atab / gtab
    f2tab = y[:, 1, 1] / y2[-1] * atab / g2tab
    # Similarly for second order derivatives
    # Note: these factors are not accessible as parent functions yet
    # since it is unclear what to refer to them with.
    htab = dyda2[:, 1, 0] / y1[-1] * atab / gtab
    h2tab = dyda2[:, 1, 1] / y2[-1] * atab / g2tab

    cache = {
        "a": atab,
        "g": gtab,
        "f": ftab,
        "h": htab,
        "g2": g2tab,
        "f2": f2tab,
        "h2": h2tab,
    }
    cosmo._workspace["background.growth_factor"] = cache
  else:
    cache = cosmo._workspace["background.growth_factor"]
  return np.clip(interp(a, cache["a"], cache["g"]), 0.0, 1.0)


def _growth_rate_ODE(cosmo, a):
  """Compute growth rate dD/dlna at a given scale factor by solving the linear
    growth ODE.

    Parameters
    ----------
    cosmo: `Cosmology`
      Cosmology object

    a: array_like
      Scale factor

    Returns
    -------
    f:  ndarray, or float if input scalar
        Growth rate computed at requested scale factor
    """
  # Check if growth has already been computed, if not, compute it
  if not "background.growth_factor" in cosmo._workspace.keys():
    _growth_factor_ODE(cosmo, np.atleast_1d(1.0))
  cache = cosmo._workspace["background.growth_factor"]
  return interp(a, cache["a"], cache["f"])


def _growth_factor_second_ODE(cosmo, a):
  """Compute second order growth factor D2(a) at a given scale factor,
    normalised such that D(a=1) = 1.

    Parameters
    ----------
    a: array_like
      Scale factor

    amin: float
      Mininum scale factor, default 1e-3

    Returns
    -------
    D2:  ndarray, or float if input scalar
        Second order growth factor computed at requested scale factor
    """
  # Check if growth has already been computed, if not, compute it
  if not "background.growth_factor" in cosmo._workspace.keys():
    _growth_factor_ODE(cosmo, np.atleast_1d(1.0))
  cache = cosmo._workspace["background.growth_factor"]
  return interp(a, cache["a"], cache["g2"])


def _growth_rate_ODE(cosmo, a):
  """Compute growth rate dD/dlna at a given scale factor by solving the linear
    growth ODE.

    Parameters
    ----------
    cosmo: `Cosmology`
      Cosmology object

    a: array_like
      Scale factor

    Returns
    -------
    f:  ndarray, or float if input scalar
        Second order growth rate computed at requested scale factor
    """
  # Check if growth has already been computed, if not, compute it
  if not "background.growth_factor" in cosmo._workspace.keys():
    _growth_factor_ODE(cosmo, np.atleast_1d(1.0))
  cache = cosmo._workspace["background.growth_factor"]
  return interp(a, cache["a"], cache["f"])


def _growth_rate_second_ODE(cosmo, a):
  """Compute second order growth rate dD2/dlna at a given scale factor by solving the linear
    growth ODE.

    Parameters
    ----------
    cosmo: `Cosmology`
      Cosmology object

    a: array_like
      Scale factor

    Returns
    -------
    f2:  ndarray, or float if input scalar
        Second order growth rate computed at requested scale factor
    """
  # Check if growth has already been computed, if not, compute it
  if not "background.growth_factor" in cosmo._workspace.keys():
    _growth_factor_ODE(cosmo, np.atleast_1d(1.0))
  cache = cosmo._workspace["background.growth_factor"]
  return interp(a, cache["a"], cache["f2"])


def _growth_factor_gamma(cosmo, a, log10_amin=-3, steps=128):
  r"""Computes growth factor by integrating the growth rate provided by the
    \gamma parametrization. Normalized such that D( a=1) =1

    Parameters
    ----------
    a: array_like
      Scale factor

    amin: float
      Mininum scale factor, default 1e-3

    Returns
    -------
    D:  ndarray, or float if input scalar
        Growth factor computed at requested scale factor

    """
  # Check if growth has already been computed, if not, compute it
  if not "background.growth_factor" in cosmo._workspace.keys():
    # Compute tabulated array
    atab = np.logspace(log10_amin, 0.0, steps)

    def integrand(y, loga):
      xa = np.exp(loga)
      return _growth_rate_gamma(cosmo, xa)

    gtab = np.exp(odeint(integrand, np.log(atab[0]), np.log(atab)))
    gtab = gtab / gtab[-1]  # Normalize to a=1.
    cache = {"a": atab, "g": gtab}
    cosmo._workspace["background.growth_factor"] = cache
  else:
    cache = cosmo._workspace["background.growth_factor"]
  return np.clip(interp(a, cache["a"], cache["g"]), 0.0, 1.0)


def _growth_rate_gamma(cosmo, a):
  r"""Growth rate approximation at scale factor `a`.

    Parameters
    ----------
    cosmo: `Cosmology`
        Cosmology object

    a : array_like
        Scale factor

    Returns
    -------
    f_gamma : ndarray, or float if input scalar
        Growth rate approximation at the requested scale factor

    Notes
    -----
    The LCDM approximation to the growth rate :math:`f_{\gamma}(a)` is given by:

    .. math::

        f_{\gamma}(a) = \Omega_m^{\gamma} (a)

     with :math: `\gamma` in LCDM, given approximately by:
     .. math::

        \gamma = 0.55

    see :cite:`2019:Euclid Preparation VII, eqn.32`
    """
  return Omega_m_a(cosmo, a)**cosmo.gamma


def Gf(cosmo, a):
  r"""
    FastPM growth factor function

    Parameters
    ----------
    cosmo: dict
      Cosmology dictionary.

    a : array_like
       Scale factor.

    Returns
    -------
    Scalar float Tensor : FastPM growth factor function.

    Notes
    -----

    The expression for :math:`Gf(a)` is:

    .. math::
        Gf(a)=D'_{1norm}*a**3*E(a)
    """
  f1 = growth_rate(cosmo, a)
  g1 = growth_factor(cosmo, a)
  D1f = f1 * g1 / a
  return D1f * np.power(a, 3) * np.power(Esqr(cosmo, a), 0.5)


def Gf2(cosmo, a):
  r""" FastPM second order growth factor function

    Parameters
    ----------
    cosmo: dict
      Cosmology dictionary.

    a : array_like
       Scale factor.

    Returns
    -------
    Scalar float Tensor : FastPM second order growth factor function.

    Notes
    -----

    The expression for :math:`Gf_2(a)` is:

    .. math::
        Gf_2(a)=D'_{2norm}*a**3*E(a)
    """
  f2 = growth_rate_second(cosmo, a)
  g2 = growth_factor_second(cosmo, a)
  D2f = f2 * g2 / a
  return D2f * np.power(a, 3) * np.power(Esqr(cosmo, a), 0.5)


def dGfa(cosmo, a):
  r""" Derivative of Gf against a

    Parameters
    ----------
    cosmo: dict
       Cosmology dictionary.

    a : array_like
       Scale factor.

    Returns
    -------
    Scalar float Tensor : the derivative of Gf against a.

    Notes
    -----

    The expression for :math:`gf(a)` is:

    .. math::
        gf(a)=\frac{dGF}{da}= D^{''}_1 * a ** 3 *E(a) +D'_{1norm}*a ** 3 * E'(a)
                +   3 * a ** 2 * E(a)*D'_{1norm}

    """
  f1 = growth_rate(cosmo, a)
  g1 = growth_factor(cosmo, a)
  D1f = f1 * g1 / a
  cache = cosmo._workspace['background.growth_factor']
  f1p = cache['h'] / cache['a'] * cache['g']
  f1p = interp(np.log(a), np.log(cache['a']), f1p)
  Ea = E(cosmo, a)
  return (f1p * a**3 * Ea + D1f * a**3 * dEa(cosmo, a) + 3 * a**2 * Ea * D1f)


def dGf2a(cosmo, a):
  r""" Derivative of Gf2 against a

    Parameters
    ----------
    cosmo: dict
      Cosmology dictionary.

    a : array_like
       Scale factor.

    Returns
    -------
    Scalar float Tensor : the derivative of Gf2 against a.

    Notes
    -----

    The expression for :math:`gf2(a)` is:

    .. math::
        gf_2(a)=\frac{dGF_2}{da}= D^{''}_2 * a ** 3 *E(a) +D'_{2norm}*a ** 3 * E'(a)
                +   3 * a ** 2 * E(a)*D'_{2norm}

    """
  f2 = growth_rate_second(cosmo, a)
  g2 = growth_factor_second(cosmo, a)
  D2f = f2 * g2 / a
  cache = cosmo._workspace['background.growth_factor']
  f2p = cache['h2'] / cache['a'] * cache['g2']
  f2p = interp(np.log(a), np.log(cache['a']), f2p)
  E = E(cosmo, a)
  return (f2p * a**3 * E + D2f * a**3 * dEa(cosmo, a) + 3 * a**2 * E * D2f)
