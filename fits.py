import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

def box_gaussian(x, A, x0, w, sigma, c):
    s = np.sqrt(2) * sigma
    return 0.5 * A * (erf((x - x0 + w/2) / s) - erf((x - x0 - w/2) / s)) + c

def edge(x, A, x0, sigma, c):
    return 0.5 * A * (1 + erf((x - x0) / (np.sqrt(2) * sigma))) + c


def curve_fit_xerr(f, x, y, yerr, xerr, n_iter=2, **kwargs):
    """curve_fit with errors on both axes (effective variance method).

    sigma_eff^2 = yerr^2 + (f'(x) * xerr)^2, with the slope f' from the previous fit. Starts with
    only yerr, then refits n_iter times. kwargs go to curve_fit (p0, bounds, maxfev, ...).
    Returns popt, pcov like curve_fit (absolute errors).
    """
    x = np.asarray(x, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    h = 1e-3 * max(np.ptp(x), 1e-12)
    sigma = yerr
    for _ in range(n_iter + 1):
        popt, pcov = curve_fit(f, x, y, sigma=sigma, absolute_sigma=True, **kwargs)
        slope = (f(x + h, *popt) - f(x - h, *popt)) / (2 * h)
        sigma = np.sqrt(yerr**2 + (slope * xerr) ** 2)
        kwargs["p0"] = popt
    return popt, pcov


def beam_width(z, w0, z0, zr):
    """Gaussian beam: width w0 at the focus z0, Rayleigh length zr. w(z) = w0 sqrt(1 + ((z-z0)/zr)^2)"""
    return w0 * np.sqrt(1 + ((z - z0) / zr) ** 2)


def fit_beam(z, sig, err, zerr, k=3, maxfev=10000):
    """Fit beam_width to the widths sig(z) with errors on both axes and one round of outlier rejection.

    Returns popt = (w0, z0, zr) and its covariance. Points further than k robust sigma from the first
    fit are left out of the second.
    """
    z, sig, err = (np.asarray(a, dtype=float) for a in (z, sig, err))
    span = max(np.ptp(z), 1e-9)
    i = np.argmin(sig)
    p0 = [sig[i], z[i], span / 2]
    bounds = ([0, z.min(), 1e-3 * span], [np.inf, z.max(), 100 * span])
    popt, _ = curve_fit_xerr(beam_width, z, sig, err, zerr, p0=p0, bounds=bounds, maxfev=maxfev)
    res = sig - beam_width(z, *popt)
    mad = np.median(np.abs(res - np.median(res)))
    keep = np.abs(res) < k * 1.4826 * mad if mad > 0 else np.ones(z.size, dtype=bool)
    if keep.sum() < 4:   # 3 parameters, do not throw away what the fit needs
        keep = np.ones(z.size, dtype=bool)
    return curve_fit_xerr(beam_width, z[keep], sig[keep], err[keep], zerr, p0=popt, bounds=bounds, maxfev=maxfev)


def polyfit_xerr(x, y, yerr, xerr, deg=2, n_iter=2):
    """Weighted polyfit with errors on both axes (effective variance method), returns coefs, cov."""
    x = np.asarray(x, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    sigma = yerr
    for _ in range(n_iter + 1):
        coefs, cov = np.polyfit(x, y, deg, w=1 / sigma, cov="unscaled")
        sigma = np.sqrt(yerr**2 + (np.polyval(np.polyder(coefs), x) * xerr) ** 2)
    return coefs, cov
