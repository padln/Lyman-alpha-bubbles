import numpy as np
from numpy.linalg import LinAlgError
from scipy import integrate
from scipy.stats import gaussian_kde

from astropy.cosmology import z_at_value
from astropy import units as u
from astropy.cosmology import Planck18 as Cosmo

from venv.galaxy_prop import get_js, get_mock_data
from venv.igm_prop import get_bubbles
from venv.helpers import optical_depth


wave_em = np.linspace(1213, 1219., 100) * u.Angstrom


def calculate_taus(
        x_small,
        y_small,
        z_small,
        r_bubbles,
        red_s,
        z_end_bub,
        n_iter=1000
):
    """
    Calculates taus for random sightlines for a given bubble configuration.

    :param x_small: numpy.array;
        x-coordinates of centers of outside bubbles.
    :param y_small: numpy.array;
        y-coordinates of centers of outside bubbles.
    :param z_small: numpy.array;
        z-coordinates of centers of outside bubbles.
    :param r_bubbles: numpy.array;
        radii of outside bubbles.
    :param red_s: float;
        redshift of the center of the main bubble.
    :param z_end_bub: float;
        redshift of the edge from which we calculate taus.
    :param n_iter: integer;
        number of random sightlines sampled.

    :return taus: numpy.arrayM
        Optical depths for a given configuration.
    """
    x_random = np.random.uniform(-10, 10, size=n_iter)
    y_random = np.random.uniform(-10, 10, size=n_iter)
    taus = []

    z_start = z_end_bub

    for i, (xr, yr) in enumerate(zip(x_random, y_random)):
        z_edge_up = []
        z_edge_lo = []
        red_edge_up = []
        red_edge_lo = []
        for xb, yb, zb, rb in zip(x_small, y_small, z_small, r_bubbles):
            # initialize lists of intersections.
            # check if galaxy will intersect the bubble.
            # note that bubbles behind the galaxy don't influence it.
            if (xr - xb) ** 2 + (yr - yb) ** 2 < rb ** 2:
                # calculate z-s and redshifts at which it'll happen.
                z_edge_up_i = zb - np.sqrt(
                    rb ** 2 - ((xr - xb) ** 2 + (yr - yb) ** 2))
                z_edge_lo_i = zb + np.sqrt(
                    rb ** 2 - ((xr - xb) ** 2 + (yr - yb) ** 2))
                # get the redshifts
                red_edge_up_i = z_at_value(
                    Cosmo.comoving_distance,
                    Cosmo.comoving_distance(
                        7.5) - z_edge_up_i * u.Mpc - 5 * u.Mpc
                    # the radius of the big bubble
                )
                red_edge_lo_i = z_at_value(
                    Cosmo.comoving_distance,
                    Cosmo.comoving_distance(
                        7.5) - z_edge_lo_i * u.Mpc - 5 * u.Mpc
                    # the radius of the big bubble
                )
                z_edge_up.append(np.copy(z_edge_up_i)[0])
                z_edge_lo.append(np.copy(z_edge_lo_i)[0])
                red_edge_up.append(np.copy(red_edge_up_i)[0])
                red_edge_lo.append(np.copy(red_edge_lo_i)[0])
        z_edge_up = np.array(z_edge_up)
        z_edge_lo = np.array(z_edge_lo)
        red_edge_up = np.array(red_edge_up)
        red_edge_lo = np.array(red_edge_lo)
        if len(z_edge_up) == 0:
            # galaxy doesn't intersect any of the bubbles:
            taus.append(
                np.array(optical_depth(
                    wave_em,
                    t=1 * u.K,
                    z_min=5.5,
                    z_max=z_start,
                    z_s=7.5,
                    z_bubble_center=7.5,
                    inside_hii=False
                )
                )
            )
            continue
        indices_up = z_edge_up.argsort()

        z_edge_up_sorted = z_edge_up[indices_up]
        z_edge_lo_sorted = z_edge_lo[indices_up]
        red_edge_up_sorted = red_edge_up[indices_up]
        red_edge_lo_sorted = red_edge_lo[indices_up]

        indices_to_del_lo = []
        indices_to_del_up = []

        for j in range(len(z_edge_up) - 1):
            if len(z_edge_up_sorted) != 1:
                if red_edge_lo_sorted[j] < red_edge_up_sorted[j + 1]:
                    # got an overlapping bubble
                    indices_to_del_lo.append(j)
                    indices_to_del_up.append(j + 1)
        z_edge_lo_sorted = np.delete(z_edge_lo_sorted, indices_to_del_lo)
        z_edge_up_sorted = np.delete(z_edge_up_sorted, indices_to_del_up)
        red_edge_up_sorted = np.delete(red_edge_up_sorted, indices_to_del_up)
        red_edge_lo_sorted = np.delete(red_edge_lo_sorted, indices_to_del_lo)
        tau_i = np.zeros(len(wave_em))
        for index, (z_up_i, z_lo_i, red_up_i, red_lo_i) in enumerate(
                zip(z_edge_up_sorted, z_edge_lo_sorted, red_edge_up_sorted,
                    red_edge_lo_sorted)):
            if index == 0:
                if z_up_i < 0 and z_lo_i < 0:
                    print("wrong bubble, it's above the galaxy.")
                    raise ValueError
                if z_up_i < 0 < z_lo_i:
                    tau_i = np.array(
                        optical_depth(
                            wave_em,
                            t=1e4 * u.K,
                            z_min=red_lo_i,
                            z_max=z_start,
                            z_s=red_s,
                            z_bubble_center=red_s,
                            inside_hii=True
                        )
                    )
                else:

                    tau_i = np.array(
                        optical_depth(
                            wave_em,
                            t=1 * u.K,
                            z_min=red_up_i,
                            z_max=z_start,
                            z_s=red_s,
                            z_bubble_center=7.5,
                            inside_hii=False
                        )
                    )
                    tau_i += np.array(
                        optical_depth(
                            wave_em,
                            t=1e4 * u.K,
                            z_min=red_lo_i,
                            z_max=red_up_i,
                            z_s=red_s,
                            z_bubble_center=7.5,
                            inside_hii=True
                        )
                    )
            elif index != len(z_edge_up_sorted):
                tau_i += np.array(
                    optical_depth(
                        wave_em,
                        t=1 * u.K,
                        z_min=red_up_i,
                        z_max=red_edge_lo_sorted[index - 1],
                        z_s=red_s,
                        z_bubble_center=7.5,
                        inside_hii=False
                    )
                )
                tau_i += np.array(
                    optical_depth(
                        wave_em,
                        t=1e4 * u.K,
                        z_min=red_lo_i,
                        z_max=red_up_i,
                        z_s=red_s,
                        z_bubble_center=7.5,
                        inside_hii=True
                    )
                )
            else:
                tau_i += np.array(
                    optical_depth(
                        wave_em,
                        t=1 * u.K,
                        z_min=red_up_i,
                        z_max=red_edge_lo_sorted[index - 1],
                        z_s=red_s,
                        z_bubble_center=7.5,
                        inside_hii=False
                    )
                )
                tau_i += np.array(
                    optical_depth(
                        wave_em,
                        t=1e4 * u.K,
                        z_min=red_lo_i,
                        z_max=red_up_i,
                        z_s=red_s,
                        z_bubble_center=7.5,
                        inside_hii=True
                    )
                )
                tau_i += np.array(
                    optical_depth(
                        wave_em,
                        t=1 * u.K,
                        z_min=5.5,
                        z_max=red_lo_i,
                        z_s=red_s,
                        z_bubble_center=7.5,
                        inside_hii=False
                    )
                )
        taus.append(tau_i)
    return taus


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
                            xs, ys, zs, r_bubs = get_bubbles(
                                7.5,
                                0.8,
                                300
                            )
                            tau_now_i = calculate_taus(
                                xs,
                                ys,
                                zs,
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
                    likelihood_grid[xi, yi, zi, Ri] = likelihood
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
