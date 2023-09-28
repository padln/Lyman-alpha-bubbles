import numpy as np
from numpy.linalg import LinAlgError
from scipy import integrate
from scipy.stats import gaussian_kde

from astropy.cosmology import z_at_value
from astropy import units as u
from astropy.cosmology import Planck18 as Cosmo

from venv.galaxy_prop import get_js, get_mock_data
from venv.igm_prop import get_bubbles, calculate_taus


wave_em = np.linspace(1213, 1219., 100) * u.Angstrom


def sample_bubbles_grid(
        tau_data,
        xs,
        ys,
        zs,
        n_iter_bub=100,
        n_grid=10
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

    :return likelihood_grid: np.array of shape (N_grid, N_grid, N_grid, N_grid);
        likelihoods for the data on a grid defined above.
    """

    # first specify a range for bubble size and bubble position
    r_min = 0.5  # small bubble
    r_max = 49.5  # bubble not bigger than the actual size of the box
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

    likelihood_grid = np.zeros((n_grid, n_grid, n_grid, n_grid))

    for xi, xb in enumerate(x_grid):
        for yi, yb in enumerate(y_grid):
            for zi, zb in enumerate(z_grid):
                for Ri, Rb in enumerate(r_grid):
                    likelihood = 1.0
                    taus_tot = []
                    # For these parameters let's iterate over galaxies
                    for xg, yg, zg in zip(xs, ys, zs):
                        taus_now = []
                        red_s = z_at_value(
                            Cosmo.comoving_distance,
                            Cosmo.comoving_distance(7.5) + zg * u.Mpc
                        )
                        if ((xg - xb) ** 2 - (yg - yb) ** 2
                                + (zg - zb) ** 2 < Rb ** 2):
                            dist = zg - zb + np.sqrt(
                                Rb ** 2 - (xg - xb) ** 2 - (yg - yb) ** 2
                            )
                            z_end_bub = z_at_value(
                                Cosmo.comoving_distance,
                                Cosmo.comoving_distance(red_s) - dist * u.Mpc
                            )
                        else:
                            z_end_bub = red_s
                        for n in range(n_iter_bub):
                            j_s = get_js(-22, n_iter=40)
                            x_outs, y_outs, z_outs, r_bubs = get_bubbles(
                                7.5,
                                0.8,
                                300
                            )
                            tau_now_i = calculate_taus(
                                x_outs,
                                y_outs,
                                z_outs,
                                r_bubs,
                                red_s,
                                z_end_bub,
                                n_iter=40,
                            )
                            eit = np.exp(-np.array(tau_now_i))
                            taus_now.extend(np.trapz(
                                eit * j_s[0] / integrate.trapz(
                                    j_s[0][0],
                                    wave_em.value),
                                wave_em.value)
                            )

                        taus_tot.append(taus_now)
                    try:
                        tau_kde = gaussian_kde(np.array(taus_tot))
                        likelihood *= tau_kde.evaluate(tau_data)
                    except LinAlgError:
                        likelihood *= 0
                    likelihood_grid[xi, yi, zi, Ri] = likelihood[0]
    return likelihood_grid


if __name__ == '__main__':
    td, xd, yd, zd = get_mock_data(
        n_gal=10,
        r_bubble=10,
    )
    likelihoods = sample_bubbles_grid(
        tau_data=td,
        xs=xd,
        ys=yd,
        zs=zd,
        n_iter_bub=20,
        n_grid=10,
    )
    np.save(
        '/home/inikolic/projects/Lyalpha_bubbles/code/likelihoods.npy',
        likelihoods
    )
