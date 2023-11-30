import numpy as np
from numpy.linalg import LinAlgError
import argparse

from scipy import integrate
from scipy.stats import gaussian_kde

from astropy.cosmology import z_at_value
from astropy import units as u
from astropy.cosmology import Planck18 as Cosmo

import itertools
from joblib import Parallel, delayed

from venv.galaxy_prop import get_js, get_mock_data, calculate_EW_factor
from venv.galaxy_prop import get_muv
from venv.igm_prop import get_bubbles
from venv.igm_prop import calculate_taus_i, get_xH


wave_em = np.linspace(1213, 1219., 100) * u.Angstrom


def _get_likelihood(
        ndex,
        xb,
        yb,
        zb,
        rb,
        xs,
        ys,
        zs,
        tau_data,
        n_iter_bub,
        redshift=7.5,
        muv=None,
        include_muv_unc=False,
        beta_data=None,
        use_EW=False,
        xH_unc=False,
):
    """

    :param ndex: integer;
        index of the calculation
    :param xb: float;
        x-coordinate of the bubble
    :param yb: float;
        y-coordinate of the bubble
    :param zb: float;
        z-coordinate of the bubble
    :param rb: float;
        Radius of the bubble.
    :param xs: list;
        x-coordinates of galaxies;
    :param ys: list;
        y-coordinates of galaxies;
    :param zs: list;
        z-coordinates of galaxies;
    :param tau_data: list;
        transmissivities of galaxies.
    :param n_iter_bub: integer;
        how many times to iterate over the bubbles.
    :param include_muv_unc: boolean;
        whether to include the uncertainty in the Muv.

    :return:

    Note: to be used only in the sampler
    """
    likelihood = 1.0
    taus_tot = []
    # For these parameters let's iterate over galaxies
    if beta_data is None:
        beta_data = np.zeros(len(xs))
    for xg, yg, zg, muvi, beti in zip(xs, ys, zs, muv, beta_data):
        taus_now = []
        red_s = z_at_value(
            Cosmo.comoving_distance,
            Cosmo.comoving_distance(redshift) + zg * u.Mpc
        )
        if ((xg - xb) ** 2 + (yg - yb) ** 2
                + (zg - zb) ** 2 < rb ** 2):
            dist = zg - zb + np.sqrt(
                rb ** 2 - (xg - xb) ** 2 - (yg - yb) ** 2
            )
            z_end_bub = z_at_value(
                Cosmo.comoving_distance,
                Cosmo.comoving_distance(red_s) - dist * u.Mpc
            )
        else:
            z_end_bub = red_s
            dist = 0
        for n in range(n_iter_bub):
            j_s = get_js(muv=muvi, n_iter=50, include_muv_unc=include_muv_unc)
            if xH_unc:
                x_H = get_xH(redshift)  # using the central redshift.
            else:
                x_H=0.65
            x_outs, y_outs, z_outs, r_bubs = get_bubbles(
                x_H,
                300
            )
            tau_now_i = calculate_taus_i(
                x_outs,
                y_outs,
                z_outs,
                r_bubs,
                red_s,
                z_end_bub,
                n_iter=50,
                dist=dist,
            )
            eit_l = np.exp(-np.array(tau_now_i))
            res = np.trapz(
                eit_l * j_s[0] / integrate.trapz(
                    j_s[0][0],
                    wave_em.value),
                wave_em.value
            )
            #EW_data = calculate_EW_factor(muvi, beti) * np.array(tau_data_I)

            if np.all(np.array(res) < 10000):
                if use_EW:
                    taus_now.extend(
                        (calculate_EW_factor(muvi, beti) * res).tolist()
                    )
                else:
                    taus_now.extend(np.array(res).tolist()) # just to be sure
            else:
                print("smth wrong", res, flush=True )

        taus_tot.append(np.array(taus_now).flatten())
    taus_tot_b = []
    #print(np.shape(taus_tot_b), np.shape(tau_data), flush=True)

    try:
        taus_tot_b = []
        for li in taus_tot:
            if np.all(np.array(li) < 10000.0): # maybe unnecessary
                taus_tot_b.append(li)
#        print(np.shape(taus_tot_b), np.shape(tau_data), flush=True)

        for ind_data, tau_line in enumerate(np.array(taus_tot_b)):
            tau_kde = gaussian_kde((np.array(tau_line)))
            if use_EW:
                if tau_data[ind_data] < 3:
                    likelihood *= tau_kde.integrate_box(0, 3)
                else:
                    likelihood *= tau_kde.evaluate((tau_data[ind_data]))
            else:
                if tau_data[ind_data] < 1e-3:
                    likelihood *= tau_kde.integrate_box(0, 1e-3)
                else:
                    likelihood *= tau_kde.evaluate((tau_data[ind_data]))
        # print(
        #     np.array(taus_tot),
        #     np.array(tau_data),
        #     np.shape(taus_tot),
        #     np.shape(tau_data),
        #     tau_kde.evaluate(tau_data),
        #     "This is what evaluate does for this params",
        #     xb, yb, zb, rb  , flush=True
        # )
    except (LinAlgError, ValueError, TypeError):
        likelihood *= 0
        print("OOps there was valu error, let's see why:", flush=True)
        print(tau_data, flush=True)
        print(taus_tot_b, flush=True)
        raise TypeError
    if hasattr(likelihood, '__len__'):
        return ndex, np.product(likelihood)
    else:
        return ndex, likelihood


def sample_bubbles_grid(
        tau_data,
        xs,
        ys,
        zs,
        n_iter_bub=100,
        n_grid=10,
        redshift=7.5,
        muv=None,
        include_muv_unc=False,
        beta_data=None,
        use_EW=False,
        xH_unc=False,
):
    """
    The function returns the grid of likelihood values for given input
    parameters.

    :param tau_data: list of floats;
        List of transmission that represent our data.
    :param xs: list of floats;
        List of x-positions that represent our data.
    :param ys: list of floats,
        List of y-positions that represent our data.
    :param zs: list of floats,
        List of z-positions that represent our data. LoS axis.
    :param n_iter_bub: integer,
        number of outside bubble configurations per iteration.
    :param n_grid: integer,
        number of grid points.
    :param muv: muv,
        UV magnitude data
    :param include_muv_unc: boolean,
        whether to include muv uncertainty.
    :param beta_data: float,
        beta data.
    :param use_EW: boolean
        whether to use EW or transmissions directly.

    :return likelihood_grid: np.array of shape (N_grid, N_grid, N_grid, N_grid);
        likelihoods for the data on a grid defined above.
    """

    # first specify a range for bubble size and bubble position
    r_min = 5  # small bubble
    #r_max = 37  # bubble not bigger than the actual size of the box
    r_max = 35
    r_grid = np.linspace(r_min, r_max, n_grid)

    x_min = -15.5
    x_max = 15.5
    x_grid = np.linspace(x_min, x_max, n_grid)

    y_min = -15.5
    y_max = 15.5
    y_grid = np.linspace(y_min, y_max, n_grid)

    z_min = -15.5
    z_max = 15.5
    z_grid = np.linspace(z_min, z_max, n_grid)
    x_grid = np.linspace(x_min, x_max, n_grid)[6:7]
    y_grid = np.linspace(y_min, y_max, n_grid)[6:7]
    like_calc = Parallel(
        n_jobs=50
    )(
        delayed(
            _get_likelihood
        )(
            index,
            xb,
            yb,
            zb,
            rb,
            xs,
            ys,
            zs,
            tau_data,
            n_iter_bub,
            redshift=redshift,
            muv=muv,
            include_muv_unc=include_muv_unc,
            beta_data=beta_data,
            use_EW=use_EW,
            xH_unc=xH_unc,
        ) for index, (xb, yb, zb, rb) in enumerate(
            itertools.product(x_grid, y_grid, z_grid, r_grid)
        )
    )
    like_calc.sort(key=lambda x: x[0])
    likelihood_grid = np.array([l[1] for l in like_calc])
    likelihood_grid.reshape((len(x_grid), len(y_grid), len(z_grid), len(r_grid)))

    return likelihood_grid


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mock_direc", type=str, default=None)
    parser.add_argument("--redshift", type=float, default=7.5)
    parser.add_argument("--r_bub", type=float, default=15.0)
    parser.add_argument("--max_dist", type=float, default=15.0)
    parser.add_argument("--n_gal", type=int, default=20)
    parser.add_argument("--obs_pos", type=bool, default=False)
    parser.add_argument("--use_EW", type=bool, default=False)
    parser.add_argument("--diff_mags", type=bool, default=True)
    parser.add_argument(
        "--use_Endsley_Stark_mags",
        type=bool, default=False
    )
    parser.add_argument("--mag_unc", type=bool, default=True)
    parser.add_argument("--xH_unc", type=bool, default=False)
    parser.add_argument("--muv_cut", type=float, default=-19.0)
    parser.add_argument("--diff_poss_prob", type=bool, default=True)
    inputs = parser.parse_args()

    if inputs.diff_mags:
        if inputs.use_Endsley_Stark_mags:
            Muv = get_muv(
                10,
                7.5,
                obs='EndsleyStark21',
            )
        else:
            Muv = get_muv(
                n_gal=inputs.n_gal,
                redshift=inputs.redshift,
                muv_cut=inputs.muv_cut,
            )
    else:
        Muv = -22.0 * np.ones((inputs.n_gal))

    if inputs.use_Endsley_Stark_mags:
        beta = np.array([
            -2.11,
            -2.64,
            -1.95,
            -2.06,
            -1.38,
            -2.77,
            -2.44,
            -2.12,
            -2.59,
            -1.43,
            -2.43,
            -2.02
        ])
    else:
        beta = -2.0 * np.ones((inputs.n_gal))

    if inputs.mock_direc is None:
        td, xd, yd, zd, x_b, y_b, z_b, r_bubs = get_mock_data(
            n_gal=len(Muv),
            z_start=inputs.redshift,
            r_bubble=inputs.r_bub,
            dist=inputs.max_dist,
            ENDSTA_data=inputs.obs_pos,
            diff_pos_prob=inputs.diff_pos_prob,
        )
        tau_data_I = []
        one_J = get_js(z=inputs.redshift, muv=Muv, n_iter=len(Muv))
        for i in range(len(td)):
            eit = np.exp(-td[i])
            tau_data_I.append(
                np.trapz(
                    eit * one_J[0][i]/integrate.trapz(
                        one_J[0][i],
                        wave_em.value
                    ),
                    wave_em.value)
            )
        if inputs.use_EW:
            data = calculate_EW_factor(Muv, beta) * np.array(tau_data_I)
        else:
            data = np.array(tau_data_I)

    else:
        data = np.load(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/tau_data.npy'  # maybe I'll change this
        )
        xd = np.load(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/x_gal_mock.npy'
        )
        yd = np.load(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/y_gal_mock.npy'
        )
        zd = np.load(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/z_gal_mock.npy'
        )
        x_b = np.load(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/x_bub_mock.npy'
        )
        y_b = np.load(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/y_bub_mock.npy'
        )
        z_b = np.load(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/z_bub_mock.npy'
        )
        r_bubs = np.load(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/r_bubs_mock.npy'
        )

    likelihoods = sample_bubbles_grid(
        tau_data=np.array(data),
        xs=xd,
        ys=yd,
        zs=zd,
        n_iter_bub=30,
        n_grid=13,
        redshift=inputs.redshift,
        muv=Muv,
        include_muv_unc=inputs.mag_unc,
        use_EW=inputs.use_EW,
        beta_data=beta,
        xH_unc=inputs.xH_unc
    )
    np.save(
        '/home/inikolic/projects/Lyalpha_bubbles/code/likelihoods.npy',
        likelihoods
    )
    np.save(
        '/home/inikolic/projects/Lyalpha_bubbles/code/x_gal_mock.npy',
        np.array(xd)
    )
    np.save(
        '/home/inikolic/projects/Lyalpha_bubbles/code/y_gal_mock.npy',
        np.array(yd)
    )
    np.save(
        '/home/inikolic/projects/Lyalpha_bubbles/code/z_gal_mock.npy',
        np.array(zd)
    )
    np.save(
        '/home/inikolic/projects/Lyalpha_bubbles/code/x_bub_mock.npy',
        np.array(x_b)
    )
    np.save(
        '/home/inikolic/projects/Lyalpha_bubbles/code/y_bub_mock.npy',
        np.array(y_b)
    )
    np.save(
        '/home/inikolic/projects/Lyalpha_bubbles/code/z_bub_mock.npy',
        np.array(z_b)
    )
    np.save(
        '/home/inikolic/projects/Lyalpha_bubbles/code/r_bubs_mock.npy',
        np.array(r_bubs)
    )
    np.save(
        '/home/inikolic/projects/Lyalpha_bubbles/code/data.npy',
        np.array(data),
    )
