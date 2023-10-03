import numpy as np
from astropy import units as u
from numpy.random import normal
from astropy.cosmology import z_at_value
from astropy.cosmology import Planck18 as Cosmo

from venv.helpers import wave_to_dv, gaussian, optical_depth
from venv.igm_prop import get_bubbles, calculate_taus
from venv.igm_prop import calculate_taus_i

wave_em = np.linspace(1213, 1219., 100) * u.Angstrom


def delta_v_func(
        muv,
        z=7
):
    """
    Function returns velocity offset as a function of UV magnitude

    :param muv: float,
        UV magnitude of the galaxy.
    :param z: float,
        redshift of the galaxy.

    :return delta_v: float
        velocity offset of a given galaxy.
    """
    if muv >= -20.0 - 0.26 * z:
        gamma = -0.3
    else:
        gamma = -0.7
    return 0.32 * gamma * (muv + 20.0 + 0.26 * z) + 2.34


def get_js(
        muv=-18,
        z=7,
        n_iter=1):
    """
    Function returns Lyman-alpha shape profiles

    :param muv: float,
        UV magnitude for which we're calculating the shape.
    :param z: float,
        redshift of interest.
    :param n_iter: integer,
        number of iterations of Lyman shape to get.

    :return j_s: numpy.array of shape (N_iter, n_wav);
        array of profiles for a number of iterations and wavelengths.
    :return delta_vs : numpy.array of shape (N_iter);
        array of velocity offsets for a number of iterations.
    """
    n_wav = 100

    wave_em = np.linspace(1213, 1219., n_wav) * u.Angstrom
    wv_off = wave_to_dv(wave_em)
    delta_vs = np.zeros(n_iter)
    j_s = np.zeros((n_iter, n_wav))
    delta_v_mean = delta_v_func(muv, z)

    for i in range(n_iter):
        delta_vs[i] = 10**normal(delta_v_mean, 0.24)
        j_s[i, :] = gaussian(wv_off.value, delta_vs[i], delta_vs[i])

    return j_s, delta_vs


def get_mock_data(
        n_gal=10,
        z_start=7.5,
        r_bubble=5,
        xb=0.0,
        yb=0.0,
        zb=0.0,
):
    """

        Generate mock data
        Input:
        N_gal : integer,
            number of galaxies for which to create mock data.
        z_start : float,
            redshift of the center of the bubble.
        R_bubble : float,
            radius of the center ionized bubble.

        Returns:
        tau_data: numpy.array;
            List of transmissivities that represent the mock data.
        xs: numpy.array;
            x-coordinates of the mock galaxies.
        ys: numpy.array;
            y-coordinates of the mock galaxies.
        zs: numpy.array;
            z-coordinates of the mock galaxies. This is the LoS direction,
            negative values represent galaxies closer to us w.r.t. the center
            of the bubble.
    """

    xs = np.random.uniform(-10, 10, size=n_gal)  # limits could also be variables
    ys = np.random.uniform(-10, 10, size=n_gal)
    zs = np.random.uniform(-10, 10, size=n_gal)
    tau_data = np.zeros((n_gal, len(wave_em)))
    for i in range(n_gal):
        red_s = z_at_value(
            Cosmo.comoving_distance,
            Cosmo.comoving_distance(z_start) + zs[i] * u.Mpc
        )
        if (xs[i] - xb) ** 2 + (ys[i] - yb) ** 2 + (
                zs[i] - zb) ** 2 < r_bubble ** 2:
            dist = zs[i] - zb + np.sqrt(
                r_bubble ** 2 - (xs[i] - xb) ** 2 - (ys[i] - yb) ** 2)
            z_end_bub = z_at_value(
                Cosmo.comoving_distance,
                Cosmo.comoving_distance(red_s) - dist * u.Mpc
            )
        else:
            z_end_bub = red_s
        x_b, y_b, z_b, r_bubs = get_bubbles(7.5, 0.8, 50)
        tau = calculate_taus(x_b, y_b, z_b, r_bubs, red_s, z_end_bub, n_iter=1)
        tau_data[i, :] = tau[0]
    return tau_data, xs, ys, zs, x_b, y_b, z_b, r_bubs

