import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf

def box_gaussian(x, A, x0, w, sigma, c):
    s = np.sqrt(2) * sigma
    return 0.5 * A * (erf((x - x0 + w/2) / s) - erf((x - x0 - w/2) / s)) + c

def gaussian(x, A, x0, sigma, c):
    return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + c

FIT_MODELS = ("box_gaussian", "gaussian")
SIGMA_INDEX = {"box_gaussian": 3, "gaussian": 2}   # position of sigma in the parameters of each model


def edge(x, A, x0, sigma, c):
    return 0.5 * A * (1 + erf((x - x0) / (np.sqrt(2) * sigma))) + c

def linear(x, k, m):
    return k*x + m

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


def fit_peak(model, x, y, yerr, xerr, pitch, base=0.0, bounded=True, maxfev=10000):
    """Fit a peak with "box_gaussian" (A, x0, w, sigma, c) or "gaussian" (A, x0, sigma, c).

    Returns curve, popt, pcov, sigma, sigma_err. curve(xf) evaluates the fit, and sigma is the
    Gaussian sigma of either model (for the box-gaussian the width w is not included).
    base is the assumed baseline for the start values. Raises like curve_fit if the fit fails.
    """
    if model not in FIT_MODELS:
        raise ValueError(f"Unknown fit_model {model!r}, choose from {FIT_MODELS}")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.broadcast_to(np.asarray(yerr, dtype=float), y.shape)
    # if model == "box_gaussian":
    #     keep = spike_mask(x, y)
    #     x, y, yerr = x[keep], y[keep], yerr[keep]
    argmax = np.argmax(y)
    peak = y[argmax]
    above = x[y > base + (peak - base) / 2]
    w = above.max() - above.min() if above.size > 1 else pitch / 2
    step = np.diff(x).min() if x.size > 1 else pitch / 10
    if model == "box_gaussian":
        f = box_gaussian
        p0 = [peak - base, x[argmax], w, max(w / 10, step), base]
        lo = [0, x.min(), 0, 1e-4, -np.inf]
        hi = [np.inf, x.max(), np.ptp(x), np.ptp(x), np.inf]
    else:
        f = gaussian
        p0 = [peak - base, x[argmax], max(w / 2.355, step), base]   # w is about the FWHM
        lo = [0, x.min(), 1e-4, -np.inf]
        hi = [np.inf, x.max(), np.ptp(x), np.inf]
    kwargs = dict(p0=p0, maxfev=maxfev)
    if bounded:
        kwargs["bounds"] = (lo, hi)
    popt, pcov = curve_fit_xerr(f, x, y, yerr, xerr, **kwargs)
    i = SIGMA_INDEX[model]
    return (lambda xf: f(xf, *popt)), popt, pcov, abs(popt[i]), np.sqrt(pcov[i, i])


def beam_width(z, w0, z0, zr):
    """Gaussian beam: width w0 at the focus z0, Rayleigh length zr. w(z) = w0 sqrt(1 + ((z-z0)/zr)^2)"""
    return w0 * np.sqrt(1 + ((z - z0) / zr) ** 2)


def fit_beam(z, sig, err, zerr, k=3, maxfev=10000):
    """Fit beam_width to the widths sig(z) with errors on both axes and one round of outlier rejection.

    Returns popt = (w0, z0, zr) and its covariance. Points further than k robust sigma from the first
    fit are left out of the second.
    """
    z, sig, err = (np.asarray(a, dtype=float) for a in (z, sig, err))
    if z.size < 6:
        raise ValueError(f"too few points for a beam fit ({z.size})")
    span = max(np.ptp(z), 1e-9)
    step = np.median(np.diff(np.sort(z)))
    zr_min = max(step, 1e-3 * span)
    i = np.argmin(sig)
    p0 = [sig[i], z[i], max(span / 2, 2 * zr_min)]
    bounds = ([0, z.min(), zr_min], [np.inf, z.max(), 100 * span])
    popt, _ = curve_fit_xerr(beam_width, z, sig, err, zerr, p0=p0, bounds=bounds, maxfev=maxfev)
    res = sig - beam_width(z, *popt)
    mad = np.median(np.abs(res - np.median(res)))
    keep = np.abs(res) < k * 1.4826 * mad if mad > 0 else np.ones(z.size, dtype=bool)
    if keep.sum() < 6:   # 3 parameters, do not throw away what the fit needs
        keep = np.ones(z.size, dtype=bool)
    zk = z[keep]
    popt, pcov = curve_fit_xerr(beam_width, zk, sig[keep], err[keep], zerr, p0=popt, bounds=bounds, maxfev=maxfev)
    w0, z0, zr = popt

    if not np.all(np.isfinite(pcov)):
        raise ValueError(f"covariance is not finite, params {popt}")
    return popt, pcov


def polyfit_xerr(x, y, yerr, xerr, deg=2, n_iter=2):
    """Weighted polyfit with errors on both axes (effective variance method), returns coefs, cov."""
    x = np.asarray(x, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    sigma = yerr
    for _ in range(n_iter + 1):
        coefs, cov = np.polyfit(x, y, deg, w=1 / sigma, cov="unscaled")
        sigma = np.sqrt(yerr**2 + (np.polyval(np.polyder(coefs), x) * xerr) ** 2)
    return coefs, cov


def spike_mask(x: np.ndarray, y: np.ndarray):
    """Removes spikes larger than half of the max-min difference"""
    diff = y.max() - y.min()
    step = np.abs(np.diff(y, prepend=y[0]))

    return np.abs(step) < diff / 2