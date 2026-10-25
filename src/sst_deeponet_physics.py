"""SST residual equations preserved from train_sst_pod.py."""
import torch

def make_residuals(dx, dz, wall_distance):
    def derivative(v, matrix):
        return torch.sparse.mm(matrix, v.T).T

    def residuals(fields, re):
        # All terms are nondimensionalized with chord=1 and case Uinf.
        u, w, p, k, omega, nut = fields.unbind(1)
        k = k.clamp_min(1e-9)
        omega = omega.clamp_min(1e-8)
        nut = nut.clamp_min(0)
        nu = 1 / re[:, None]
        ux, uz, wx, wz = derivative(u, dx), derivative(u, dz), derivative(w, dx), derivative(w, dz)
        px, pz = derivative(p, dx), derivative(p, dz)
        kx, kz = derivative(k, dx), derivative(k, dz)
        ox, oz = derivative(omega, dx), derivative(omega, dz)
        div = ux + wz
        s2 = 2 * ux.square() + 2 * wz.square() + (uz + wx).square()
        g0 = (s2 - 2 / 3 * div.square()).clamp_min(0)
        y = torch.tensor(wall_distance, dtype=torch.float32)[None, :].clamp_min(1e-6)
        beta_star, a1, b1, c1 = 0.09, 0.31, 1.0, 10.0
        alpha_k1, alpha_k2 = 0.85, 1.0
        alpha_o1, alpha_o2 = 0.5, 0.856
        gamma1, gamma2 = 5 / 9, 0.44
        beta1, beta2 = 0.075, 0.0828
        cd = 2 * alpha_o2 * (kx * ox + kz * oz) / omega
        arg1 = torch.minimum(torch.maximum(torch.sqrt(k) / (beta_star * omega * y),
                          500 * nu / (y.square() * omega)),
                          4 * alpha_o2 * k / (cd.clamp_min(1e-10) * y.square()))
        f1 = torch.tanh(arg1.clamp_max(10).pow(4))
        arg2 = torch.maximum(2 * torch.sqrt(k) / (beta_star * omega * y),
                             500 * nu / (y.square() * omega))
        f2 = torch.tanh(arg2.clamp_max(100).square())
        blend = lambda first, second: f1 * first + (1 - f1) * second
        alpha_k = blend(alpha_k1, alpha_k2)
        alpha_o = blend(alpha_o1, alpha_o2)
        gamma, beta = blend(gamma1, gamma2), blend(beta1, beta2)
        closure = a1 * k / torch.maximum(a1 * omega, b1 * f2 * torch.sqrt(s2.clamp_min(0)))
        g = nut * g0
        pk = torch.minimum(g, c1 * beta_star * k * omega)
        g_by_nu = torch.minimum(g0, c1 / a1 * beta_star * omega *
                                torch.maximum(a1 * omega, b1 * f2 * torch.sqrt(s2.clamp_min(0))))
        effective = nu + nut
        tau_xx = effective * (2 * ux - 2 / 3 * div)
        tau_zz = effective * (2 * wz - 2 / 3 * div)
        tau_xz = effective * (uz + wx)
        rx = u * ux + w * uz + px - derivative(tau_xx, dx) - derivative(tau_xz, dz)
        rz = u * wx + w * wz + pz - derivative(tau_xz, dx) - derivative(tau_zz, dz)
        dk = nu + alpha_k * nut
        do = nu + alpha_o * nut
        rk = u * kx + w * kz - derivative(dk * kx, dx) - derivative(dk * kz, dz) \
             - pk + 2 / 3 * div * k + beta_star * omega * k
        ro = u * ox + w * oz - derivative(do * ox, dx) - derivative(do * oz, dz) \
             - gamma * g_by_nu + 2 / 3 * gamma * div * omega + beta * omega.square() \
             - 2 * (1 - f1) * alpha_o2 * (kx * ox + kz * oz) / omega
        return {
            "continuity": div, "momentum_x": rx, "momentum_z": rz,
            "k_transport": rk, "omega_transport": ro,
            "nut_closure": nut - closure,
        }

    return residuals
