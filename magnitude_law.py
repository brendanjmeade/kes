"""
Deficit-weighted magnitude law: a moment-constrained MaxEnt tilt of the G-R law

Loading-origin events draw their magnitude from

    ln p(M | l) = ln p_GR(M) + theta(l) * phi(M),   M in [M_min, M_max]

where l = ln(D_eff / D_w) is the state coordinate (the effective slip-deficit
reservoir relative to its reference), phi(M) the tilt shape and theta(l) the
Lagrange multiplier of a moment-budget constraint:

    shape "moment":  phi = (m(M) / m(M_max))^p   (p = 1: the MaxEnt density under
                     a constraint on the mean geometric moment, i.e. Kagan's
                     tapered G-R with a state-dependent, sign-free corner;
                     acts on 10^(1.5 M), so only the top ~half magnitude unit)
    shape "gamma":   phi = gamma(M) / gamma(M_max)  (the joint MaxEnt over (i, M)
                     with the spatial partition function retained; collapses to
                     (D/D_ref)^(s gamma(M)); concave, lifts M6-6.5 nearly as much
                     as M7+)
    shape "linear":  phi = (M - M_min) / (M_max - M_min)  (a b-value tilt,
                     b(D) = b - theta / (ln 10 (M_max - M_min)))

    target "loglinear":  theta = mu * l / c_phi with c_phi = Cov_GR(m, phi)/E_GR[m],
                         so that d ln E[m] / d ln D = mu at D = D_w
    target "power":      theta solved so that E[m | D] = E_GR[m] (D / D_w)^mu exactly

theta(l = 0) = 0, so at the reference reservoir the law is today's truncated G-R
and the moment-balance calibration constants are unchanged. mu = 0 disables the
tilt (the simulator then uses its legacy closed-form draw and consumes the same
random stream).

Sampling consumes exactly ONE uniform per draw: the analytic truncated-G-R
inverse when theta == 0 (optionally capped at M_cap for Bath-limited
aftershocks), otherwise an interpolated inverse of the tabulated CDF on a
magnitude grid (max error ~5e-7 in M at 2001 nodes).
"""

import numpy as np

from moment import magnitude_to_seismic_moment
from spatial_prob import gamma_magnitude_dependent

LN10 = np.log(10.0)
_LN_TINY = -700.0


def truncated_gr_inverse(u, b, M_lo, M_hi):
    """Analytic inverse CDF of the doubly truncated G-R law (one uniform)."""
    denom = 10 ** (-b * M_lo) - 10 ** (-b * M_hi)
    return -np.log10(10 ** (-b * M_lo) - u * denom) / b


class TiltedGR:
    """
    Tabulated deficit-tilted G-R law on a magnitude grid

    Parameters
    ----------
    config : Config (needs the magnitude_tilt_* keys, b_value, M_min, M_max,
             shear_modulus_Pa, element_area_m2, geom_loading_rate_total)
    D_ref : float
        Reference reservoir (m^3); the neutral point D_w defaults to it.
    slip_rate : (n_elements,) array, optional
        Needed for magnitude_tilt_deficit == "shadow" (reference field v_i T_w).
    """

    def __init__(self, config, D_ref, slip_rate=None):
        # Catalog (child) law: G-R with the catalog b on [M_min, M_max]. Omori
        # children under the Bath split always draw from it (capped below the
        # parent). Loading-origin events draw from the LOADING law: the same
        # G-R unless magnitude_load_M_min raises the floor (small events then
        # exist only as aftershock children) or magnitude_load_b changes b.
        self.b_child = float(config.b_value)
        self.M_min_child = float(config.M_min)
        self.M_max = float(config.M_max)
        M_lo = float(getattr(config, "magnitude_load_M_min", 0.0))
        self.M_min = M_lo if M_lo > self.M_min_child else self.M_min_child
        if self.M_min >= self.M_max:
            raise ValueError("magnitude_load_M_min must be below M_max")
        b_load = float(getattr(config, "magnitude_load_b", 0.0))
        self.b = b_load if b_load > 0 else self.b_child
        n = int(getattr(config, "magnitude_grid_n", 2001))
        self.M = np.linspace(self.M_min, self.M_max, n)
        self.dM = self.M[1] - self.M[0]
        self.m = magnitude_to_seismic_moment(self.M) / config.shear_modulus_Pa  # m^3
        self.m_max = float(self.m[-1])
        self.element_area_m2 = float(config.element_area_m2)

        # Loading-law G-R density on the grid (trapezoid normalization)
        self.ln_pgr = -self.b * LN10 * self.M
        pgr = np.exp(self.ln_pgr - self.ln_pgr.max())
        self.pgr = pgr / self._integral(pgr)
        self.E_gr = float(self._integral(self.m * self.pgr))

        # Child (catalog) law on its own grid, for the cascade budget
        self.M_child = np.linspace(self.M_min_child, self.M_max, n)
        self.dM_child = self.M_child[1] - self.M_child[0]
        self.m_child = magnitude_to_seismic_moment(self.M_child) / config.shear_modulus_Pa
        p_child = 10.0 ** (-self.b_child * (self.M_child - self.M_min_child))
        self.p_child = p_child / np.trapezoid(p_child, dx=self.dM_child)

        # Tilt shape
        self.shape = getattr(config, "magnitude_tilt_shape", "moment")
        self.power = float(getattr(config, "magnitude_tilt_power", 1.0))
        if self.shape == "moment":
            phi = (self.m / self.m_max) ** self.power
        elif self.shape == "gamma":
            g = gamma_magnitude_dependent(
                self.M, config.gamma_min, config.gamma_max, config.alpha_spatial, self.M_min
            )
            phi = g / g[-1] if g[-1] > 0 else np.zeros_like(g)
        elif self.shape == "linear":
            phi = (self.M - self.M_min) / (self.M_max - self.M_min)
        else:
            raise ValueError(f"Unknown magnitude_tilt_shape: {self.shape!r}")
        self.pivot = float(getattr(config, "magnitude_tilt_pivot", self.M_min))
        if self.pivot > self.M_min:
            phi_p = float(np.interp(self.pivot, self.M, phi))
            phi = np.maximum(phi - phi_p, 0.0)
        self.phi = phi
        # c_phi = Cov_GR(m, phi) / E_GR[m]  ->  d ln E[m] / d theta at theta = 0
        self.c_phi = float(
            (self._integral(self.m * self.phi * self.pgr)
             - self.E_gr * self._integral(self.phi * self.pgr)) / self.E_gr
        )

        self.mu = float(getattr(config, "magnitude_tilt_mu", 0.0))
        self.target = getattr(config, "magnitude_tilt_target", "loglinear")
        if self.target not in ("loglinear", "power"):
            raise ValueError(f"Unknown magnitude_tilt_target: {self.target!r}")
        self.theta_max = float(getattr(config, "magnitude_tilt_theta_max", 200.0))
        self.cap_fill = float(getattr(config, "magnitude_cap_fill_fraction", 0.0))

        # Deficit reference (neutral point of the tilt)
        self.D_ref = float(D_ref)
        L_tot = float(getattr(config, "geom_loading_rate_total", 0.0))
        ref_years = float(getattr(config, "magnitude_tilt_reference_years", 0.0))
        self.D_w = ref_years * L_tot if (ref_years > 0 and L_tot > 0) else self.D_ref

        # Deficit coordinate
        self.deficit_mode = getattr(config, "magnitude_tilt_deficit", "reservoir")
        if self.deficit_mode not in ("reservoir", "shadow"):
            raise ValueError(f"Unknown magnitude_tilt_deficit: {self.deficit_mode!r}")
        self._gamma_top = float(gamma_magnitude_dependent(
            self.M_max, config.gamma_min, config.gamma_max, config.alpha_spatial, self.M_min
        ))
        self._lnZ_ref = None
        if self.deficit_mode == "shadow":
            if slip_rate is None or L_tot <= 0 or self._gamma_top <= 0:
                raise ValueError("magnitude_tilt_deficit='shadow' needs slip_rate, L_tot and gamma(M_max) > 0")
            T_w = self.D_w / L_tot
            ref_field = np.maximum(np.asarray(slip_rate, dtype=float) * T_w, 1e-10)
            self._lnZ_ref = float(np.log(np.sum(ref_field ** self._gamma_top)))
            self._n_elements = ref_field.size

        # theta(l) table for the exact power-law target
        self._ell_grid = None
        self._theta_grid = None
        if self.target == "power" and self.mu != 0.0:
            self._build_power_table()

    # ------------------------------------------------------------------ helpers
    def _integral(self, f):
        return np.trapezoid(f, dx=self.dM)

    def _cumulative(self, p):
        """Trapezoid cumulative integral on the grid (F[0] = 0)."""
        F = np.empty_like(p)
        F[0] = 0.0
        np.cumsum(0.5 * (p[1:] + p[:-1]) * self.dM, out=F[1:])
        return F

    def _build_power_table(self):
        ell = np.linspace(np.log(0.02), np.log(6.0), 500)
        E_lo = self._expected_moment_theta(-self.theta_max)
        E_hi = self._expected_moment_theta(self.theta_max)
        theta = np.empty_like(ell)
        for i, l in enumerate(ell):
            target = self.E_gr * np.exp(self.mu * l)
            if target <= E_lo:
                theta[i] = -self.theta_max
                continue
            if target >= E_hi:
                theta[i] = self.theta_max
                continue
            lo, hi = -self.theta_max, self.theta_max
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if self._expected_moment_theta(mid) < target:
                    lo = mid
                else:
                    hi = mid
            theta[i] = 0.5 * (lo + hi)
        self._ell_grid, self._theta_grid = ell, theta

    def _log_density(self, theta, M_cap=None, D=None):
        """Unnormalized-then-normalized ln p on the grid (masked nodes -> -inf)."""
        lnp = self.ln_pgr + theta * self.phi
        if M_cap is not None and M_cap < self.M_max:
            lnp = np.where(self.M <= M_cap, lnp, -np.inf)
        if self.cap_fill > 0 and D is not None:
            lnp = np.where(self.m <= self.cap_fill * D, lnp, -np.inf)
        if not np.isfinite(lnp[0]):
            # Cap below M_min: nothing hostable; fall back to the smallest node
            lnp = np.where(self.M <= self.M[0], self.ln_pgr, -np.inf)
        return lnp

    def _pdf_theta(self, theta, M_cap=None, D=None):
        lnp = self._log_density(theta, M_cap, D)
        p = np.exp(np.maximum(lnp - lnp.max(), _LN_TINY))
        p[~np.isfinite(lnp)] = 0.0
        return p / self._integral(p)

    def _expected_moment_theta(self, theta, M_cap=None, D=None):
        return float(self._integral(self.m * self._pdf_theta(theta, M_cap, D)))

    # ---------------------------------------------------------------- state API
    def log_deficit(self, m_working, h=None):
        """
        State coordinate l = ln(D_eff / D_w)

        "reservoir": D_eff = A * sum(m_i).  "shadow": the h-weighted log
        effective deficit l = [ln sum_i h_i m_i^g - ln Z_ref - ln(sum h_i / N)] / g
        with g = gamma(M_max) (equals ln(D / D_w) to ~0.01 when h == 1).
        """
        if self.deficit_mode == "reservoir":
            D = max(float(np.sum(m_working)) * self.element_area_m2, 1e-300)
            return float(np.log(D / self.D_w))
        m_safe = np.maximum(m_working, 1e-10)
        if h is None:
            Z = np.sum(m_safe ** self._gamma_top)
            mean_h = 1.0
        else:
            Z = np.sum(h * m_safe ** self._gamma_top)
            mean_h = max(float(np.sum(h)) / self._n_elements, 1e-300)
        return float((np.log(max(Z, 1e-300)) - self._lnZ_ref - np.log(mean_h)) / self._gamma_top)

    def theta(self, ell):
        """Tilt multiplier at state coordinate l (scalar or array)."""
        if self.mu == 0.0:
            return np.zeros_like(np.asarray(ell, dtype=float)) if np.ndim(ell) else 0.0
        if self.target == "loglinear":
            th = self.mu * np.asarray(ell, dtype=float) / self.c_phi
            return np.clip(th, -self.theta_max, self.theta_max)
        assert self._ell_grid is not None and self._theta_grid is not None
        return np.interp(ell, self._ell_grid, self._theta_grid)

    def pdf(self, ell=0.0, M_cap=None, D=None):
        """(M grid, normalized density) at state l."""
        return self.M, self._pdf_theta(float(self.theta(ell)), M_cap, D)

    def expected_moment(self, ell=0.0, M_cap=None, D=None):
        """E[m | l] in m^3 (E_gr at l = 0)."""
        return self._expected_moment_theta(float(self.theta(ell)), M_cap, D)

    def tail(self, ell, M_thr):
        """P(M >= M_thr | l)."""
        _, p = self.pdf(ell)
        sel = self.M >= M_thr
        return float(self._integral(np.where(sel, p, 0.0)))

    # --------------------------------------------------------------------- draw
    def draw(self, u, ell=None, M_cap=None, D=None, child=False):
        """
        Magnitude from one uniform u

        child=True (Omori child): analytic inverse of the catalog G-R on
        [M_min, min(M_max, M_cap)], never tilted.  Loading events: ell None or
        theta(ell) == 0 (and no fill cap) -> analytic inverse of the loading
        G-R on [M_min_load, M_max]; otherwise the interpolated inverse of the
        tabulated CDF of the tilted law.
        """
        if child or M_cap is not None:
            M_hi = self.M_max if M_cap is None else min(self.M_max, float(M_cap))
            if M_hi <= self.M_min_child:
                return self.M_min_child
            return float(truncated_gr_inverse(u, self.b_child, self.M_min_child, M_hi))
        theta = 0.0 if ell is None else float(self.theta(ell))
        if theta == 0.0 and not (self.cap_fill > 0 and D is not None):
            return float(truncated_gr_inverse(u, self.b, self.M_min, self.M_max))
        p = self._pdf_theta(theta, M_cap, D)
        F = self._cumulative(p)
        target = u * F[-1]
        k = int(np.searchsorted(F, target, side="right")) - 1
        k = min(max(k, 0), F.size - 2)
        width = F[k + 1] - F[k]
        frac = (target - F[k]) / width if width > 0 else 0.0
        return float(self.M[k] + min(max(frac, 0.0), 1.0) * self.dM)

    # ---------------------------------------------------------------- analysis
    def log_norm(self, theta):
        """ln N(theta) = ln int p_GR exp(theta phi) dM, vectorized over theta."""
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        out = np.empty(theta.size)
        for i0 in range(0, theta.size, 256):
            th = theta[i0:i0 + 256]
            lnw = th[:, None] * self.phi[None, :]
            shift = lnw.max(axis=1, keepdims=True)
            w = np.exp(lnw - shift) * self.pgr[None, :]
            out[i0:i0 + 256] = np.log(np.trapezoid(w, dx=self.dM, axis=1)) + shift[:, 0]
        return out

    def log_weight(self, magnitude, ell):
        """ln p(M | l) - ln p_GR(M): importance weight of the tilted law over G-R."""
        magnitude = np.asarray(magnitude, dtype=float)
        theta = np.asarray(self.theta(np.asarray(ell, dtype=float)), dtype=float)
        phi_M = np.interp(magnitude, self.M, self.phi)
        return theta * phi_M - self.log_norm(theta).reshape(theta.shape)

    def marginal(self, thetas, weights=None):
        """Weighted time-average of p(M | theta) over a sample of thetas."""
        thetas = np.asarray(thetas, dtype=float)
        weights = np.ones_like(thetas) if weights is None else np.asarray(weights, dtype=float)
        acc = np.zeros_like(self.M)
        for i0 in range(0, thetas.size, 256):
            th, wt = thetas[i0:i0 + 256], weights[i0:i0 + 256]
            lnw = th[:, None] * self.phi[None, :]
            p = np.exp(lnw - lnw.max(axis=1, keepdims=True)) * self.pgr[None, :]
            p /= np.trapezoid(p, dx=self.dM, axis=1)[:, None]
            acc += (wt[:, None] * p).sum(axis=0)
        return self.M, acc / max(weights.sum(), 1e-300)

    def child_curves(self, dM_bath, n_child=None):
        """
        For every loading-grid node M (as a parent): eligibility
        (M - dM_bath >= M_min of the catalog), the mean geometric moment of a
        child drawn from the catalog G-R on [M_min, M - dM_bath], and (if
        n_child, the expected offspring of a child of magnitude M' on the child
        grid, is given) the mean offspring per child.
        """
        def cum(f):
            F = np.empty_like(f)
            F[0] = 0.0
            np.cumsum(0.5 * (f[1:] + f[:-1]) * self.dM_child, out=F[1:])
            return F
        Fp = cum(self.p_child)
        Fm = cum(self.m_child * self.p_child)
        M_cap = self.M - dM_bath
        mass = np.interp(M_cap, self.M_child, Fp, left=0.0)
        mom = np.interp(M_cap, self.M_child, Fm, left=0.0)
        eligible = M_cap >= self.M_min_child
        mean_child = np.where(mass > 0, mom / np.maximum(mass, 1e-300), 0.0)
        mean_offspring = np.zeros_like(mean_child)
        if n_child is not None:
            Fn = cum(self.p_child * n_child)
            nsum = np.interp(M_cap, self.M_child, Fn, left=0.0)
            mean_offspring = np.where(mass > 0, nsum / np.maximum(mass, 1e-300), 0.0)
        return eligible, mean_child, mean_offspring

    def describe(self):
        lines = [
            f"  Magnitude law: tilted G-R (shape={self.shape}, power={self.power}, target={self.target}, "
            f"mu={self.mu}, deficit={self.deficit_mode})",
            f"    loading law: G-R b = {self.b} on [{self.M_min}, {self.M_max}]; "
            f"catalog/child law: b = {self.b_child} on [{self.M_min_child}, {self.M_max}]",
            f"    grid {self.M.size} nodes, E_GR[m] = {self.E_gr:.4e} m^3, c_phi = {self.c_phi:.4f} "
            f"(theta = {self.mu / self.c_phi if self.c_phi else np.nan:.3f} * ln(D/D_w) for loglinear)",
            f"    D_w = {self.D_w:.3e} m^3 ({self.D_w / self.D_ref:.3f} D_ref), |theta| <= {self.theta_max}",
        ]
        if self.pivot > self.M_min:
            lines.append(f"    pivot M_p = {self.pivot}")
        if self.cap_fill > 0:
            lines.append(f"    fill-fraction cap m(M) <= {self.cap_fill} D")
        if self.mu != 0.0:
            for r in (0.6, 0.8, 1.0, 1.2, 1.5):
                l = np.log(r)
                lines.append(
                    f"    D/D_w = {r:3.1f}: theta = {float(self.theta(l)):8.3f}, "
                    f"E[m]/E_GR = {self.expected_moment(l) / self.E_gr:6.3f}, "
                    f"P(M>=7)/GR = {self.tail(l, 7.0) / max(self.tail(0.0, 7.0), 1e-300):6.3f}, "
                    f"P(M>=6.5)/GR = {self.tail(l, 6.5) / max(self.tail(0.0, 6.5), 1e-300):6.3f}"
                )
        return "\n".join(lines)
