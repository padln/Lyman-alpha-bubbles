import numpy as np
from venv.galaxy_prop import get_js, p_EW
from venv.igm_prop import get_xH, get_bubbles


class OutsideContainer:
    def __init__(self):
        self.j_s_full = []
        self.x_h_full = []
        self.x_bub_out_full = []
        self.y_bub_out_full = []
        self.z_bub_out_full = []
        self.r_bub_out_full = []
        self.la_flux_out_full = []

    def add_j_s(self, j_s_now):
        self.j_s_full.append(j_s_now)

    def add_xhs(self, x_h_now):
        self.x_h_full.append(x_h_now)

    def add_out_bubble(self, x_out_now, y_out_now, z_out_now, r_out_now):
        self.x_bub_out_full.append(x_out_now)
        self.y_bub_out_full.append(y_out_now)
        self.z_bub_out_full.append(z_out_now)
        self.r_bub_out_full.append(r_out_now)

    def add_la_flux(self, l_flux_now):
        self.la_flux_out_full.append(l_flux_now)


def get_content(
        Muvs,
        beta=None,
        n_iter_bub=20,
        n_inside_tau=1000,
        include_muv_unc=False,
        fwhm_true=True,
        redshift=7.5,
        xh_unc=False,
        high_prob_emit = False,
        EW_fixed = False,
):
    """
        Function fills up the container which has all of the forward model parts
        that can be computed beforehand. This consists of the galaxy properties
        as well as the outside bubble configuration. The only thing main part of
        the code then needs to do is to implement the main bubble configuration
        and calculate the likelihood.
        Note that n_iter_bub and n_inside_tau have fiducial values that show
        convergence in the integrated flux.
    """

    cont_now = OutsideContainer()

    if beta is None:
        beta = np.array([-2.0] * len(Muvs))

    for index_gal, muv_i in enumerate(Muvs):
        j_s_gal_i = np.zeros((n_iter_bub * n_inside_tau, 100))
        x_h_gal_i = np.zeros(n_iter_bub)
        x_out_gal_i = []
        y_out_gal_i = []
        z_out_gal_i = []
        r_out_gal_i = []
        la_flux_gal_i = np.zeros((n_iter_bub * n_inside_tau))
        for bubble_iter in range(n_iter_bub):
            j_s = get_js(
                muv=muv_i,
                n_iter=n_inside_tau,
                include_muv_unc=include_muv_unc,
                fwhm_true=fwhm_true,
            )
            j_s_gal_i[
            bubble_iter * n_inside_tau: (bubble_iter + 1) * n_inside_tau, :
            ] = j_s[0][:n_inside_tau]
            if xh_unc:
                x_H = get_xH(redshift)  # using the central redshift.
            else:
                x_H = 0.65
            x_h_gal_i[bubble_iter] = x_H
            x_outs, y_outs, z_outs, r_bubs = get_bubbles(
                x_H,
                200
            )

            x_out_gal_i.append(x_outs)
            y_out_gal_i.append(y_outs)
            z_out_gal_i.append(z_outs)
            r_out_gal_i.append(r_bubs)

            lae_now_i = np.array(
                [p_EW(
                    muv_i,
                    beta[index_gal],
                    high_prob_emit=high_prob_emit,
                    EW_fixed=EW_fixed,
                )[1] for blah in range(n_inside_tau)]
            )

            la_flux_gal_i[
            bubble_iter * n_inside_tau: (bubble_iter + 1) * n_inside_tau
            ] = lae_now_i

        cont_now.add_j_s(j_s_gal_i)
        cont_now.add_xhs(x_h_gal_i)
        cont_now.add_out_bubble(
            x_out_gal_i,
            y_out_gal_i,
            z_out_gal_i,
            r_out_gal_i,
        )
        cont_now.add_la_flux(
            la_flux_gal_i
        )
    return cont_now

