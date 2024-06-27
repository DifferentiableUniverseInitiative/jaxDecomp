import jax.numpy as np
from jax_cosmo.background import *
from jax_cosmo.scipy.interpolate import interp
from jax_cosmo.scipy.ode import odeint


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
    return _growth_factor_gamma(cosmo, a)
  else:
    return _growth_factor_ODE(cosmo, a)
