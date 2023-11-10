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

from venv.galaxy_prop import get_js, get_mock_data
from venv.igm_prop import get_bubbles, calculate_taus
from venv.igm_prop import calculate_taus_i

from venv.data.EndSta import get_ENDSTA_gals

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
        muv=None,
        include_muv_unc=False,
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
    for xg, yg, zg, muvi in zip(xs, ys, zs, muv):
        taus_now = []
        red_s = z_at_value(
            Cosmo.comoving_distance,
            Cosmo.comoving_distance(7.5) + zg * u.Mpc
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
            x_outs, y_outs, z_outs, r_bubs = get_bubbles(
                0.8,
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
                dist = dist,
            )
            #print("Tau_now_i", tau_now_i,"x_outs", x_outs,"y_outs", y_outs,"z_outs", z_outs,"r_bubs", r_bubs,"red_s", red_s,"z_end_bub", z_end_bub)
            eit_l = np.exp(-np.array(tau_now_i))
            res = np.trapz(
                eit_l * j_s[0] / integrate.trapz(
                    j_s[0][0],
                    wave_em.value),
                wave_em.value
            )
            if np.all(np.array(res)<10):
                taus_now.extend(res)
            else:
                print("smth wrong", res, flush=True )

        taus_tot.append(taus_now)
    #print(taus_tot) 
    taus_tot_b = []
    #for li in taus_tot:
    #    if np.all(np.array(li)<10.0):
    #        taus_tot_b.append(li)
    print(np.shape(taus_tot_b), np.shape(tau_data), flush=True)
    #assert 1==0
    try:
        taus_tot_b = []
        for li in taus_tot:
            if np.all(np.array(li)<10.0):
                taus_tot_b.append(li)
        #taus_tot = np.array(taus_tot_b)[np.array(taus_tot)<10] 
        print(np.shape(taus_tot_b), np.shape(tau_data), flush=True)
     #   assert 1==0
        for ind_data, tau_line in enumerate(np.array(taus_tot_b)):
            tau_kde = gaussian_kde((np.array(tau_line)))
            if tau_data[ind_data] < 1e-4:
                likelihood *= tau_kde.integrate_box(0, 1e-4)
            else:
                likelihood *= tau_kde.evaluate((tau_data[ind_data]))
        print(
            np.array(taus_tot),
            np.array(tau_data),
            np.shape(taus_tot),
            np.shape(tau_data),
            tau_kde.evaluate(tau_data),
            "This is what evaluate does for this params",
            xb, yb, zb, rb  , flush=True
        )
    except (LinAlgError,ValueError,TypeError):
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
        muv=None,
        include_muv_unc=False,
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

    :return likelihood_grid: np.array of shape (N_grid, N_grid, N_grid, N_grid);
        likelihoods for the data on a grid defined above.
    """

    # first specify a range for bubble size and bubble position
    r_min = 5  # small bubble
    #r_max = 37  # bubble not bigger than the actual size of the box
    r_max = 25
    r_grid = np.linspace(r_min, r_max, n_grid)

    x_min = -15.5
    x_max = 15.5
    x_grid = np.linspace(x_min, x_max, n_grid)

    y_min = -15.5
    y_max = 15.5
    y_grid = np.linspace(y_min, y_max, n_grid)

    z_min = -10.5
    z_max = 10.5
    z_grid = np.linspace(z_min, z_max, n_grid)
    x_grid = np.array([0.0])
    y_grid = np.array([0.0])
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
            muv=muv,
            include_muv_unc=include_muv_unc,
        ) for index, (xb, yb, zb, rb) in enumerate(
            itertools.product(x_grid, y_grid, z_grid, r_grid)
        )
    )
    like_calc.sort(key=lambda x: x[0])
    likelihood_grid = np.array([l[1] for l in like_calc])
    likelihood_grid.reshape((n_grid, n_grid))

    return likelihood_grid


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mock_direc", type=str, default=None)
    inputs = parser.parse_args()
    Muv = np.array([
        -21.5,
        -21.1,
        -21.3,
        -21.0,
        -21.3,
        -20.4,
        -21.5,
        -20.6,
        -20.7,
        -22.0,
        -21.2,
        -20.8,
    ])
    if inputs.mock_direc is None:
        td, xd, yd, zd, x_b, y_b, z_b, r_bubs = get_mock_data(
            n_gal=len(Muv),
            r_bubble=10,
            dist=10,
            ENDSTA_data=True,
        )
        tau_data_I = []
        one_J = get_js(z=7.5,muv=Muv, n_iter = len(Muv))
        for i in range(len(td)):
            eit = np.exp(-td[i])
            tau_data_I.append(
                np.trapz(
                    eit * one_J[0][i]/integrate.trapz(one_J[0][i], wave_em.value),
                    wave_em.value)
            )

    else:
        tau_data_I = np.load(
            '/home/inikolic/projects/Lyalpha_bubbles/code/'
            + inputs.mock_direc
            + '/tau_data.npy'
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
        tau_data=np.array(tau_data_I),
        xs=xd,
        ys=yd,
        zs=zd,
        n_iter_bub=30,
        n_grid=13,
        muv=Muv,
        include_muv_unc=True,
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
        '/home/inikolic/projects/Lyalpha_bubbles/code/tau_data.npy',
        np.array(tau_data_I),
    )
