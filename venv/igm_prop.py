import numpy as np
from scipy import integrate
import os
import pandas as pd

from venv.helpers import optical_depth


def get_tl_data(
        datadir='/home/inikolic/projects/Lyalpha_bubbles/TL+23/',
        bg=True,
        z=8,
        xhi=0.8
):
    """
    Get Ting-Yi's data.

    :param datadir: string;
        where the data is stored
    :param bg: boolean;
        whether to use the rapid reionization model. Otherwise, fg is used.
    :param z: integer;
        Redshift to use. Options are 7,8,9
    :param xhi: float;
        Neutral fraction

    :return r_hist: numpy.array;
        Histogram values of radii.
    :return p_log_r_norm: numpy.array;
        histogram values of the distribution.
    """
    dir_bg_z8_rall=datadir+'bg_rall_z'+str(z) + '/' # dir for full box bubble list [Rbub]
    dir_fg_z8_rall=datadir+'fg_rall_z'+str(z) + '/'

    if bg:
        box_dir = dir_bg_z8_rall
    else:
        box_dir = dir_fg_z8_rall

    bub_list = {}
    box_fills = np.sort(os.listdir(box_dir))
    for i, fil in enumerate(box_fills):
        ind = fil.find('nf')
        xhi = np.round(float(fil[ind + 2:ind + 6]), 1)
        data = pd.read_csv(box_dir + fil)
        bub_list[str(xhi)] = pd.DataFrame(
            data=np.vstack((data['R_cMpc'], data['R_cMpc'])).T,
            columns=['Muv', 'R_cMpc'])
    df = bub_list[str(xhi)]
    r_bubs = df.R_cMpc
    bins = np.logspace(np.log10(0.4), np.log10(1.1 * np.max(r_bubs)), 12)
    pr_hist, r_hist_edge = np.histogram(r_bubs, density=True, bins=bins)
    r_hist = 0.5 * (r_hist_edge[:-1] + r_hist_edge[1:])  # calculate the centers
    yy = pr_hist * r_hist  # calculate the y values for the dP/dlogR plot

    p_log_r_norm = yy / np.trapz(yy, x=r_hist)  # normalize the distribution
    return r_hist, p_log_r_norm


def bubble_size_distro(
        r,
        r_hist=None,
        p_log_r_norm=None
):
    """
    Calculate the PDF of the bubble size distribution.
    Inputs
    ----------
    r: float
        Bubble size for which we evaluate the distribution.
    r_hist: None or list-like
        If given, it contains radii-bins for the arbitrary distribution.
    p_log_r_norm: None or list-like
        If given, it contains pdf-values for radii-bins of the distribution.

    Output
    ----------
    PDF_R: float,
        Value of the PDF at a given R.

    Note:
    The function evaluates the given PDF if its values are given. Else, it uses
    a mock distribution used in some plots.
    """

    if type(r_hist) != type(p_log_r_norm) and len(r_hist) != len(p_log_r_norm):
        raise ValueError("Your manual PDF arrays are inconsistent")

    if r_hist is None:
        alpha = 0.1
        beta = 1.0
        gamma = 1 / 10

        return np.exp(-r * beta) * (r * gamma) ** alpha

    else:
        return np.interp(r, r_hist, p_log_r_norm)


def get_bubbles(
        xh,
        z_v,
        use_tl_result=True,
):
    """
    Function takes in some bubble distribution and returns an instance of
    bubbles.

    Input:_
    ----------
    xh: float,
        neutral fraction at which we're calculating the distribution.
    z_v: float,
        the extent in Mpc of bubbles away from the main one.
    use_tl_result: boolean,
        whether to use TL+23 bubble size distribution.

    Returns:
    bubble_xs: list of floats,
        x-positions of bubbles found.
    bubble_ys: list of floats,
        y-positions of bubbles found.
    bubble_zs: list of floats,
        z-positions of bubbles found. This is a LoS axis
    bubble_rs: list of floats,
        radii of bubbles found.
    ---------

    Note:
    The code takes a fixed bubble_size_distro function which is not optimal if
    I want to change it. Something to think about.

    """
    bubble_xs = []
    bubble_ys = []
    bubble_zs = []
    bubble_rs = []
    v_tot = 0.0

    if use_tl_result:
        r_hist, p_log_r_norm = get_tl_data(
            xhi=xh
        )
        rs = np.logspace(np.log10(min(Rhist)), np.log10(max(Rhist)), 1000)
        cdf = integrate.cumtrapz(
            bubble_size_distro(
                rs,
                r_hist=r_hist,
                p_log_r_norm=p_log_r_norm  # better way to do this tomorrow
            ) / rs,
            rs
        )
    else:
        rs = np.logspace(-1,3, 1000)
        cdf = integrate.cumtrapz(
            bubble_size_distr(
                rs
            ),
            rs
        )

    try_i = 0
    tolerance = 0.01
    that_it = False
    while abs(v_tot - (1 - xh) * z_v * max(Rhist) * max(Rhist)) / (
            (1 - xh) * z_v * max(Rhist) * max(Rhist)) > tolerance:

        random_numb = np.random.uniform(size=1)
        bubble_now = np.interp(random_numb, cdf / cdf[-1], rs[:-1])

        random_x = np.random.uniform(-max(Rhist), max(Rhist))
        random_y = np.random.uniform(-max(Rhist), max(Rhist))
        random_z = np.random.uniform(0, z_v)

        v = 4. * np.pi / 3. * bubble_now ** 3
        v_ded = 0.0

        if random_x < -max(Rhist) + bubble_now:
            h = -max(Rhist) - random_x + bubble_now
            v_ded = np.pi * 2. / 3. * h ** 2 * (3 * bubble_now - h)
        elif random_x > max(Rhist) - bubble_now:
            h = bubble_now - (max(Rhist) - random_x)
            v_ded = np.pi * 2. / 3. * h ** 2 * (3 * bubble_now - h)

        v = v - v_ded
        v_ded = 0.
        if random_y < -max(Rhist) + bubble_now:
            h = -max(Rhist) - random_y + bubble_now
            v_ded = np.pi * 2. / 3. * h ** 2 * (3 * bubble_now - h)
        elif random_y > max(Rhist) - bubble_now:
            h = bubble_now - (max(Rhist) - random_y)
            v_ded = np.pi * 2. / 3. * h ** 2 * (3 * bubble_now - h)

        v = v - v_ded
        v_ded = 0.
        if random_z < 0 + bubble_now:
            h = bubble_now - random_z
            v_ded = np.pi * 2. / 3. * h ** 2 * (3 * bubble_now - h)
        elif random_z > z_v - bubble_now:
            h = bubble_now - (z_v - random_z)
            v_ded = np.pi * 2. / 3. * h ** 2 * (3 * bubble_now - h)

        v = v - v_ded
        break_it = False
        v_ded = 0
        for bx, by, bz, br in zip(bubble_xs, bubble_ys, bubble_zs, bubble_rs):
            if (bx - random_x) ** 2 + (by - random_y) ** 2 + (
                    bz - random_z) ** 2 < (br + bubble_now) ** 2:
                dist = np.sqrt((bx - random_x) ** 2 + (by - random_y) ** 2 + (
                            bz - random_z) ** 2)
                if br > bubble_now:
                    big = br
                    small = bubble_now
                else:
                    small = br
                    big = bubble_now
                if dist + small < big:
                    break_it = True
                v_ded += (np.pi * (big + small - dist) ** 2 * (
                            dist ** 2 + 2 * dist * small - 3 * small ** 2 + 2 *
                            dist * big + 6 * big * small - 3 * big ** 2)
                          / 12 / dist
                          )
        v = v - v_ded
        if break_it or v < 0:

            try_i += 1
            if try_i >= 5:
                break
            else:
                continue

        if (v_tot + v) / ((1 - xh) * z_v * max(Rhist) * max(Rhist)) > 1.0:
            if abs(v_tot + v - (1 - xh) * z_v * max(Rhist) * max(Rhist)) / (
                    v_tot + v) < tolerance:
                that_it = True
            else:
                continue

        v_tot += v
        try_i = 0
        bubble_xs.append(random_x)
        bubble_ys.append(random_y)
        bubble_zs.append(random_z)
        bubble_rs.append(bubble_now)
        if that_it:
            break
    return bubble_xs, bubble_ys, bubble_zs, bubble_rs


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
