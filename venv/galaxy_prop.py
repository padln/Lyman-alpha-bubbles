import numpy as np
from astropy import units as u
from numpy.random import normal
from astropy.cosmology import z_at_value
from astropy.cosmology import Planck18 as Cosmo

from scipy.stats import gaussian_kde

from venv.helpers import wave_to_dv, gaussian, optical_depth
from venv.igm_prop import get_bubbles, calculate_taus
from venv.igm_prop import calculate_taus_i

from venv.data.EndSta import get_ENDSTA_gals

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
        n_iter=1,
        include_muv_unc=False,
):
    """
    Function returns Lyman-alpha shape profiles

    :param muv: float,
        UV magnitude for which we're calculating the shape.
    :param z: float,
        redshift of interest.
    :param n_iter: integer,
        number of iterations of Lyman shape to get.
    :param include_muv_unc: boolean,
        whether to include the scatter in Muv.

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

    if include_muv_unc and hasattr(muv, '__len__'):
        muv = np.array([np.random.normal(i, 0.1) for i in muv])
    elif include_muv_unc and not hasattr(muv, '__len__'):
        muv = np.random.normal(muv, 0.1)

    if hasattr(muv, '__len__'):
        delta_v_mean = np.array([delta_v_func(i,z) for i in muv])
    else:
        delta_v_mean = delta_v_func(muv, z)

    for i in range(n_iter):
        if hasattr(muv, '__len__'):
            delta_vs[i] = 10**normal(delta_v_mean[i], 0.24)
        else:
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
        dist=10,
        ENDSTA_data=False,
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
        xb : float,
            x-position of the center of the main bubble.
        yb : float,
            y-position of the center of the main bubble.
        zb : float,
            z-position of the center of the main bubble.
        dist : float,
            distance up to which samples of galaxies are taken.
        ENDSTA_data : False
            whether to use data from Endsley & Stark 2021.

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
        x_b: numpy.array; 
            x-positions of the outside bubbles.
        y_b: numpy.array;
            y-positions of the outside bubbles.
        z_b: numpy.array;
            z-positions of the outside bubbles.
        r_bubs: numpy.array;
            radii of outside bubbles.
    """
    if ENDSTA_data:
        xs,ys,zs = get_ENDSTA_gals()
    else:
        xs = np.random.uniform(-dist, dist, size=n_gal)
        ys = np.random.uniform(-dist, dist, size=n_gal)
        zs = np.random.uniform(-dist, dist, size=n_gal)
    tau_data = np.zeros((n_gal, len(wave_em)))
    x_b, y_b, z_b, r_bubs = get_bubbles(0.65, 300, mock=True)
    #print(x_b,y_b,z_b,r_bubs)
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
            dist = 0
            z_end_bub = red_s
        tau = calculate_taus_i(
            x_b,
            y_b, 
            z_b,
            r_bubs,
            red_s,
            z_end_bub,
            n_iter=1,
            x_pos = xs[i],
            y_pos = ys[i],
            dist = dist
        )
        tau_data[i, :] = tau[0]
    #print(x_b,y_b,z_b,r_bubs)
    #assert 1==0
    return tau_data, xs, ys, zs, x_b, y_b, z_b, r_bubs

def calculate_EW_factor(Muv, beta, mean=False):
    """
    Function calculates the luminosity factor that is necessary to calculate
    equivalent width

    :param Muv: float
        UV magnitude of a given galaxy.
    :param beta: float
        beta slope of SED.
    :param mean: boolean
        whether to just take the mean from the distribution. Default is False

    :return: EW_fac: float
        equivalent width that, when multiplied with transmission, gives EW.
    """

    Muv_thesan = np.load('/home/inikolic/projects/Lyalpha_bubbles/code/Lyman-alpha-bubbles/venv/data/Muv_THESAN.npy')
    La_thesan = np.load('/home/inikolic/projects/Lyalpha_bubbles/code/Lyman-alpha-bubbles/venv/data/Lya_THESAN.npy')
    if hasattr(Muv, '__len__'):
         La_sample = np.zeros((len(Muv)))
         for i,(Muvi, beta_i) in enumerate(zip(Muv, beta)):
            
            Las = La_thesan[abs(Muv_thesan - Muvi) < 0.1]  # magnitude uncertainty
            #print(Las, Muv_thesan[abs(Muv_thesan - Muvi) < 0.1], Muvi, flush=True)
            if mean:
                La_sample[i] = np.mean(Las) * 3.846 * 1e33
            else:
                gk = gaussian_kde(Las * 3.846 * 1e33)
                La_sample[i] = gk.resample(1)[0][0]# * 3.846 * 1e33
    else:
        Las = La_thesan[abs(Muv_thesan - Muv) < 0.1] #magnitude uncertainty
        #print(Las, Muv_thesan[abs(Muv_thesan - Muv) < 0.1], Muv, flush=True)
        gk = gaussian_kde(Las)
        if mean:
            La_sample = np.mean(Las) * 3.846 * 1e33
        else:
            gk = gaussian_kde(Las * 3.846 * 1e33)
            La_sample = gk.resample(1)# * 3.846 * 1e33

    L_UV_mean = 10**(-0.4*(Muv-51.6))
    #print(La_sample, L_UV_mean, flush=True)
    C_const = 2.47 * 1e15 * u.Hz / 1216 / u.Angstrom * (1500 / 1216) ** (-beta-2)
    #print(C_const, flush=True)
    return La_sample / C_const.value / L_UV_mean
