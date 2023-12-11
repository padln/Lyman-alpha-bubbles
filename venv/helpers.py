from astropy import constants as const
from astropy import units as u
import numpy as np
from astropy.cosmology import Planck15

wave_Lya = 1215.67 * u.Angstrom
sigma_ion0 = 6.304e-18*u.cm**2.
freq_Lya = (const.c / wave_Lya).to(u.Hz)


def I(x):
    return x**4.5 / (1-x) + 9/7 * x**3.5  + 9/5 * x**2.5 + 3 * x**1.5 + 9 * x**0.5 - np.log(abs((1+x**0.5)/(1-x**0.5)))


def comoving_distance_from_source_Mpc(z_2, z_1):
    """
    COMOVING distance between z_1 and z_2
    """
    R_com = (z_1 - z_2)*(const.c / Planck15.H(z=z_1)).to(u.Mpc)
    return R_com


def wave_to_dv(
        wave
):
    """
    Wavelength to velocity offset.
    :param wave: float;
        wavelength of interest.
    :return dv: float;
        velocity offset.
    """
    return ((wave - wave_Lya)*const.c/wave_Lya).to(u.km/u.s)


def gaussian(
        x,
        mu,
        sig
):
    """
    Gaussian shape at a given position.

    :param x: float;
        x-coordinate for which we evaluate the Gaussian.
    :param mu: float;
        mean of the gaussian.
    :param sig: float;
        STD of the gaussian.

    :return PDF: float,
         Value of the gaussian.
    """
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


def dt_dz_func(z):
    return 1. / ((1. + z) * Planck15.H(z))


class LyaCrossSection(object):
    """
    Make Lya cross-section given temperature
    """

    def __init__(self, t=1.e4 * u.K):
        """

        """
        self.t = t

        # Calculate thermal velocity
        self.v_therm = self.v_thermal()
        self.d_freq_Lya = (freq_Lya * self.v_therm / const.c).to(u.Hz)

        # Voigt parameter
        self.av_T = self.av()

        # Cross-section peak
        self.sig_Lya0 = 5.9e-14 * (self.t / 1.e4 / u.K) ** -0.5

        return

    def av(self):
        return 4.7e-4 * (self.t.value / 1.e4) ** (-0.5)

    def voigt_badapprox(self, x):
        """Clumsy approximation for Voigt
        """
        phix = np.exp(-x ** 2.) + self.av_T / np.sqrt(np.pi) / x ** 2.
        phix[phix > 1.] = 1.
        return phix

    def voigt(self, x):
        """Voigt function approximation from Tasitsiomi 2006

        https://ui.adsabs.harvard.edu/abs/2006ApJ...645..792T/abstract

        Good to >1% for T>2K. Correctly normalized to return H(av, x).
        int(phix)  = 1
        int(Voigt) = sqrt(pi)

        Args:
            x (ndarray): dimensionless frequency

        Returns:
            Voigt function
        """

        z = (x ** 2. - 0.855) / (x ** 2. + 3.42)

        q = z * (1 + 21. / x ** 2.) * self.av_T / np.pi / (x ** 2. + 1) * \
            (0.1117 + z * (4.421 + z * (-9.207 + 5.674 * z)))

        phix = np.exp(-x ** 2.) / 1.77245385

        phix[z > 0] += q[z > 0]

        return phix * np.sqrt(np.pi)

    def v_thermal(self):
        """
        Thermal velocity (Maxwell-Boltzmann)
        """
        return (1. / np.sqrt(const.m_p / 2. / const.k_B / self.t)).to(
            u.km / u.s)

    def lya_wave_to_x(self, wave):
        """
        Convert wavelength to x
        """
        freq = (const.c / wave).to(u.Hz)

        # Dimensionless frequency
        x = (freq - freq_Lya) / self.d_freq_Lya

        return x

    def lya_x_to_wave(self, x):
        """
        Convert x to wavelength
        """
        # Wavelength
        wave = const.c / (x * self.d_freq_Lya + freq_Lya)

        return wave.to(u.Angstrom)

    def lya_cross_sec(self, wave, return_x=False):

        x = self.lya_wave_to_x(wave)

        sig_lya = self.sig_Lya0 * self.voigt(x) * u.cm ** 2.

        if return_x:
            return x, sig_lya
        else:
            return sig_lya

    def lya_cross_sec_x(self, x):

        sig_lya = self.sig_Lya0 * self.voigt(x) * u.cm ** 2.

        return sig_lya

    @staticmethod
    def optical_depth_weinberg97(self, z, n_hi):
        """
        Lya optical depth from https://ui.adsabs.harvard.edu/abs/1997ApJ...490..564W/abstract Eqn 3
        for uniform IGM
        tau = πe^2/(m_e c) * f_alpha lambda_alpha * n_HI/H(z)
        """
        tau = 1.34e-17 * u.cm ** 3 / u.s * Planck15.H(z).to(1. / u.s) * n_hi

        return tau.value


def n_h(z, x_p=0.75):
    """IGM hydrogen proper number density
    Args:
        z (ndarray): redshift
        x_p : Helium fraction
    Returns:
        number density in cm^-3
    Cen & Haiman 2000::
    >>> 8.5e-5 * ((1. + z)/8.)**3. / (u.cm**3.)
    """
    return (x_p * Planck15.Ob0 * Planck15.critical_density0 * (1+z)**3.
            / const.m_p).to(u.cm**(-3.))


def alpha_rec_b(t):
    """Recombination rate for `case B` recombination of hydrogen.
       Fitting formulae from Hui & Gnedin (1997) (in the Appendix)
       accurate to ~0.7% from 1 - 10^9K
        input T unit-less (but in K)
       Returns recombination rate in cm**3/s
    """
    lhi = 2 * 157807.0 / t
    alpha_b = 2.753e-14 * lhi**1.5 / (1.0 + (lhi / 2.740)**0.407)**2.242

    return alpha_b * u.cm**3. / u.s


def xhi_r(r, z_s, n_dot_ion, f_esc=1., c=3., delta=1, t=1e4, alpha=-1.8):
    """
    r is proper distance
    Neutral fraction from source
    (Mesinger+04)
    for f_esc = 0.5, N_ion = 1e-57/s
    Args:
        r (float): radius of the bubble.
        z_s: redshift of the source.
        n_dot_ion: ionizing photon production
        f_esc: escape fraction
        c: clumping factor
        delta: overDensity
        t: temperature of IGM
        alpha: gamma scaling.
    """
    j_source = f_esc * n_dot_ion * (alpha/(alpha - 3)) * sigma_ion0

    gamma12_source = 1./(4. * np.pi * r**2.) * j_source
    gamma12_background = 0.  # J_bg * bubbles.Gamma12(z_s) / u.s

    xhi = c * delta * n_h(z_s) * alpha_rec_b(t)/(
            gamma12_background + gamma12_source
    )

    return xhi.to(u.s/u.s)


def optical_depth(
        wave_em,
        t,
        z_min,
        z_max,
        z_s=7.,
        z_bubble_center=None,
        inside_hii=True,
        c_hii=3.,
        xtab_len=100,
        over_density=1.,
        n_dot_ion=1.e57/u.s
):
    """
    Lya optical depth as a function of wavelength
    using definition of optical depth and Lya cross-section
    """
    cross_sec = LyaCrossSection(t)

    # Redshift array
    z_tab_ends = np.array([z_min, z_max])

    # Observed wavelength
    wave_obs = wave_em * (1. + z_s)
    if z_bubble_center is None:
        z_bubble_center = z_s

    # Range of redshifted wavelength and x
    wave_z_ends = wave_obs[:, None]/(1+z_tab_ends)
    x_z_ends = cross_sec.lya_wave_to_x(wave_z_ends).value

    tau = np.zeros(len(wave_obs))
    for ww, w_obs in enumerate(wave_obs):

        # Make xtab
        if (x_z_ends[ww] < 0).all():
            xtab = -np.logspace(
                np.log10(-x_z_ends[ww].min()),
                np.log10(-x_z_ends[ww].max()),
                xtab_len
            )
            xtab = np.sort(xtab)
        elif (x_z_ends[ww] > 0).all():
            xtab = np.logspace(
                np.log10(x_z_ends[ww].min()),
                np.log10(x_z_ends[ww].max()),
                xtab_len)
            xtab = np.sort(xtab)
        else:
            xtab_neg = -np.logspace(
                -1,
                np.log10(-x_z_ends[ww].min()),
                int(xtab_len/2))
            xtab_pos = np.logspace(
                -1,
                np.log10(x_z_ends[ww].max()),
                int(xtab_len/2)
            )
            xtab = np.sort(np.concatenate((xtab_neg, xtab_pos)))

        # Get wave_redshift
        wave_redshift = cross_sec.lya_x_to_wave(xtab)

        # Get z tab
        z_tab = w_obs/wave_redshift - 1.

        # Residual neutral fraction
        if inside_hii:
            r_com = comoving_distance_from_source_Mpc(z_tab, z_bubble_center)
            r_p = r_com / (1+z_bubble_center)
            xhi = xhi_r(
                r=r_p,
                z_s=z_bubble_center,
                n_dot_ion=n_dot_ion,
                f_esc=1.,
                c=c_hii,
                t=t.value
            )
        else:
            xhi = 1.

        # Cross-section
        lya_cross = cross_sec.lya_cross_sec_x(xtab)
        # Calculate optical depth
        pre_fact = (const.c * dt_dz_func(z_tab) * xhi * over_density
                    * n_h(z_tab)).to(1./u.cm**2.)
        d_tau = pre_fact * lya_cross

        tau[ww] = np.trapz(d_tau, z_tab)

    return tau
