# Independent finite-difference check of the thermal PDE

`battery_pinn.fdm` is a plain NumPy solver for the same one-dimensional heat equation that the
network's PDE residual encodes:

    rho*cp * dT/dt = k * d2T/dx2 + q(t) - h * (T - T_ambient)

It is used to test the physics without any neural network. Space is normalised to `[0, 1]`
and time is in seconds. `T` is in degrees Celsius, `q` in W/m3, `k` in W/(m K), `rho*cp` in
J/(m3 K) and the Newton coefficient `h` in W/(m3 K).

The scheme is backward Euler in time and second-order central differences in space. Ends are
insulated (`boundary="neumann"`, the default) or held at the ambient temperature
(`boundary="dirichlet"`). The Newton cooling term `h` is zero unless asked for.
`solve_heat_sei` additionally integrates `dc/dt = -rate` and adds the SEI reaction heat
`H_SEI * rate`, as `sei_thermal_pde` does.

## Stated tolerances

Each analytical case in `test/test_fdm.py` has its own tolerance:

| Case | Reference | Tolerance |
| --- | --- | --- |
| No source, ambient start | `T = T_ambient` everywhere | `np.allclose` defaults |
| Uniform heating, insulated | `T = T0 + q t / (rho cp)` | relative 1e-9 (exact for backward Euler) |
| Cooling to ambient (2000 steps) | `T_a + (T0 - T_a) exp(-h t / rho cp)` | 0.05 K absolute |
| Balanced steady state | `T_a + q / h` | 1e-6 K absolute |
| Dirichlet steady state (41 points) | `T_a + q x (1 - x) / (2 k)` | 1e-3 K absolute |
| SEI decay at pinned temperature | `c = exp(-k t)` | 2e-3 absolute |
| SEI reaction heat, insulated | `dT = H_SEI (1 - c) / (rho cp)` | 2 % relative |

The cooling and SEI cases are first-order in time, so their error shrinks in proportion to
`1 / n_t`; raise `n_t` if you need a tighter match.

`test/test_physics_residual.py` evaluates the network PDE residual functions on fixed analytic
temperature fields (a heating ramp, conduction sign, Joule heating, the Arrhenius rate,
temperature dependence and Newton cooling), so a wrong sign or unit is caught without a model.

## The `simulate` command

```bash
python -m battery_pinn simulate --current 10 --resistance 2 --ambient 25 --duration 100
python -m battery_pinn simulate --current 5 --resistance 2 --newton-cooling --boundary dirichlet
```

It prints the heat source, then one `x  temperature` line per space point at the final time.
The Joule source is `I^2 * R`. With the defaults (insulated ends, no cooling) the profile is
uniform and equals `T_ambient + I^2 R t / (rho cp)`. A run takes well under a second (it is
tested to finish in under 1 s). It does not load or train any model.
