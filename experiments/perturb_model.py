"""Copy of model_functions.multiregion_nmrnn with perturbation hooks; model_functions.py itself is untouched.

Hooks (all off by default; with all off the output is bit-identical to mf.multiregion_nmrnn, see verify()):
  nm_clamp  : (T, n_nm) array, NaN = free. Non-NaN entries overwrite the SNc state x_nm for that step,
              after the noise is added and after the update. -10 shuts SNc off (rate 0); +10 saturates it (rate ~1).
  lesion_bg : (n_bg,) bool mask, True = lesioned SPN. The state is held at -10 (rate 0) at every step,
              including the initial state, so the unit sends nothing and records as silent.
  lesion_nm : (n_nm,) bool mask, the same for SNc units.
  ctx_off_from : int step index. From this step to the end of the trial every cortical unit's state is held
              at -10 (rate 0), after the noise and after the update, so the cue input arriving at that step is erased too.

install(lesion_bg, lesion_nm, ctx_off) swaps mf.batched_nm_rnn for a lesioned version. fit_nm_rnn, test_model,
simulate_opto and test_pnr_nm look that name up at call time, so they run unchanged on the lesioned network.
"""
import math
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
import model_functions as mf
from model_functions import exc, inh, nln, _asymmetric_trace_update

OFF, ON = -10.0, 10.0
_orig_batched = mf.batched_nm_rnn


def multiregion_nmrnn(params, x_0, z_0, inputs, tau_x, tau_z, modulation=True, opto_stimulation=None,
                      noise_std=0.05, rng_key=None, nm_clamp=None, lesion_bg=None, lesion_nm=None, ctx_off_from=None):
    def les(x, mask):
        return x if mask is None else jnp.where(mask, OFF, x)

    def clampnm(x, cl):
        return x if cl is None else jnp.where(jnp.isnan(cl), x, cl)

    x_bg0, x_c0, x_t0 = x_0
    x_nm0 = z_0
    rng_key = jr.PRNGKey(0) if rng_key is None else rng_key
    rng_key, init_key, step_key = jr.split(rng_key, 3)
    if noise_std > 0:
        x_bg0 += noise_std * jr.normal(init_key, x_bg0.shape)
        x_c0 += noise_std * jr.normal(init_key, x_c0.shape)
        x_t0 += noise_std * jr.normal(init_key, x_t0.shape)
        x_nm0 += noise_std * jr.normal(init_key, x_nm0.shape)
    x_bg0, x_nm0 = les(x_bg0, lesion_bg), les(x_nm0, lesion_nm)

    J_bg, B_bgc, J_c, B_cu, B_ct = params['J_bg'], params['B_bgc'], params['J_c'], params['B_cu'], params['B_ct']
    J_t, B_tbg, J_nm, B_nmc, B_nmbg = params['J_t'], params['B_tbg'], params['J_nm'], params['B_nmc'], params['B_nmbg']
    m, c, C, rb = params['m'], params['c'], params['C'], params['rb']
    tau_c = tau_bg = tau_t = tau_x
    tau_nm = tau_z
    tau_d1_rise = params.get('tau_d1_rise', 10.0)
    tau_d1_fall = params.get('tau_d1_fall', 15000.0)
    tau_d2_rise = params.get('tau_d2_rise', 20.0)
    tau_d2_fall = params.get('tau_d2_fall', 15000.0)
    gain_e_d1 = params.get('gain_e_d1', 1.0)
    gain_e_d2 = params.get('gain_e_d2', 1.0)

    num_bg_cells = J_bg.shape[0]
    num_c_cells = J_c.shape[0]
    n_d1_cells = num_bg_cells // 2
    n_d2_cells = num_bg_cells - n_d1_cells
    T = inputs.shape[0]
    if opto_stimulation is None:
        opto_stimulation = jnp.zeros((T, num_bg_cells))

    da0 = jnp.ravel(jax.nn.sigmoid(exc(m) @ nln(x_nm0) + c))[0]
    e_d10 = da0
    e_d20 = 1.0 - da0

    def ctx(x, t):
        return x if ctx_off_from is None else jnp.where(t >= ctx_off_from, OFF, x)

    def _step(x_and_z, u, stim, cl, step_rng_key, t):
        x_bg, x_c, x_t, x_nm, e_d1, e_d2 = x_and_z
        if noise_std > 0:
            coef = noise_std / math.sqrt(2 * tau_x)
            x_bg += coef * jr.normal(step_rng_key, x_bg.shape)
            x_c += coef * jr.normal(step_rng_key, x_c.shape)
            x_t += coef * jr.normal(step_rng_key, x_t.shape)
            coefz = noise_std / math.sqrt(2 * tau_z)
            x_nm += coefz * jr.normal(step_rng_key, x_nm.shape)
        x_bg = les(x_bg, lesion_bg)
        x_nm = les(clampnm(x_nm, cl), lesion_nm)  # lesioned units stay dead under SNc stimulation
        x_c = ctx(x_c, t)

        x_c = (1.0 - (1. / tau_c)) * x_c + (1. / tau_c) * J_c @ nln(x_c)
        x_c += (1. / tau_c) * B_cu @ u
        x_c += (1. / tau_c) * exc(B_ct) @ nln(x_t)
        x_c = ctx(x_c, t)  # BG and SNc below read the updated cortex

        if modulation:
            U = jnp.concatenate((jnp.ones((n_d1_cells, 1)), jnp.ones((n_d2_cells, 1)) * -1))
            V_bg = jnp.ones((num_bg_cells, 1))
            V_c = jnp.ones((num_c_cells, 1))
            s = jnp.ravel(jax.nn.sigmoid(exc(m) @ nln(x_nm) + c))[0]
            e_d1 = _asymmetric_trace_update(e_d1, s, tau_d1_rise, tau_d1_fall)
            e_d2 = _asymmetric_trace_update(e_d2, 1.0 - s, tau_d2_rise, tau_d2_fall)
            e_d1 = jnp.clip(e_d1, 0.0, 1.0)
            e_d2 = jnp.clip(e_d2, 0.0, 1.0)
            G_bg = jnp.exp(s * U @ V_bg.T)
            G_c = jnp.exp(s * U @ V_c.T)
            pop_gain = jnp.concatenate((
                jnp.ones((n_d1_cells,)) * (1.0 + gain_e_d1 * e_d1),
                jnp.ones((n_d2_cells,)) * (1.0 + gain_e_d2 * e_d2),
            ))
        else:
            G_bg = jnp.ones((num_bg_cells, num_bg_cells))
            G_c = jnp.ones((num_bg_cells, num_c_cells))
            pop_gain = jnp.ones((num_bg_cells,))

        bg_rec = (G_bg * inh(J_bg)) @ nln(x_bg)
        bg_ctx = (G_c * exc(B_bgc)) @ nln(x_c)
        x_bg = (1.0 - (1. / tau_bg)) * x_bg + (1. / tau_bg) * pop_gain * (bg_rec + bg_ctx)
        x_bg += (1. / tau_bg) * stim
        x_bg = les(x_bg, lesion_bg)

        x_nm = (1.0 - (1. / tau_nm)) * x_nm + (1. / tau_nm) * J_nm @ nln(x_nm)
        x_nm += (1. / tau_nm) * exc(B_nmc) @ nln(x_c)
        x_nm += (1. / tau_nm) * inh(B_nmbg) @ nln(x_bg)
        x_nm = les(clampnm(x_nm, cl), lesion_nm)  # lesioned units stay dead under SNc stimulation

        x_t = (1.0 - (1. / tau_t)) * x_t + (1. / tau_t) * J_t @ nln(x_t)
        tbg = jnp.concatenate((exc(B_tbg[:, : n_d1_cells]), inh(B_tbg[:, n_d1_cells:])), axis=1)
        x_t += (1. / tau_t) * tbg @ nln(x_bg)

        y = exc(C) @ nln(x_t) + rb
        return (x_bg, x_c, x_t, x_nm, e_d1, e_d2), (y, x_bg, x_c, x_t, x_nm)

    step_keys = jr.split(step_key, T)
    ts = jnp.arange(T)
    if nm_clamp is None:
        f, xs = (lambda s, a: _step(s, a[0], a[1], None, a[2], a[3])), (inputs, opto_stimulation, step_keys, ts)
    else:
        f, xs = (lambda s, a: _step(s, a[0], a[1], a[2], a[3], a[4])), (inputs, opto_stimulation, nm_clamp, step_keys, ts)
    _, (y, xbg, xc, xt, xnm) = lax.scan(f, (x_bg0, x_c0, x_t0, x_nm0, e_d10, e_d20), xs)
    return y, (xbg, xc, xt), xnm


def cue_onset(inputs, default=300):
    """First step with any nonzero input (the cue onset); default for a trial with no cue."""
    on = jnp.any(inputs != 0, axis=-1)
    return jnp.where(on.any(), jnp.argmax(on), default)


CTX_DELAY = 10  # cortex silencing starts this many steps (100 ms) after cue onset, i.e. at cue offset, so the cue gets in


def make_batched(lesion_bg=None, lesion_nm=None, clamp=False, ctx_off=False):
    """Same signature as mf.batched_nm_rnn; with clamp=True an 11th argument nm_clamp (T, n_nm) is shared by all trials.
    ctx_off=True silences cortex from CTX_DELAY steps after each trial's own cue onset (step 300 + CTX_DELAY on
    no-cue trials) to the end of the trial."""
    def f(params, x0, z0, inputs, tau_x, tau_z, modulation, stim, noise_std, keys, nm_clamp=None):
        return multiregion_nmrnn(params, x0, z0, inputs, tau_x, tau_z, modulation, stim, noise_std, keys,
                                 nm_clamp=nm_clamp, lesion_bg=lesion_bg, lesion_nm=lesion_nm,
                                 ctx_off_from=cue_onset(inputs) + CTX_DELAY if ctx_off else None)
    axes = (None, None, None, 0, None, None, None, 0, None, 0) + ((None,) if clamp else ())
    return vmap(f, in_axes=axes)


def install(lesion_bg=None, lesion_nm=None, ctx_off=False):
    """Route every mf.batched_nm_rnn call (training and all repo test scripts) through the lesioned / cortex-silenced model."""
    mf.batched_nm_rnn = (_orig_batched if lesion_bg is None and lesion_nm is None and not ctx_off
                         else make_batched(lesion_bg, lesion_nm, ctx_off=ctx_off))


def verify(params, n=8):
    """Max |diff| between this copy (hooks off) and mf.multiregion_nmrnn, with noise on."""
    import config_script as cs
    inp, _, _ = mf.self_timed_movement_task(jnp.arange(250, 250 + n), cs.config['T_cue'], cs.config['T_wait'],
                                            cs.config['T_movement'], cs.config['T'])
    keys = jr.split(jr.PRNGKey(5), n)
    a = _orig_batched(params, cs.x0, cs.z0, inp, cs.config['tau_x'], cs.config['tau_z'], True, None, 0.05, keys)
    b = make_batched()(params, cs.x0, cs.z0, inp, cs.config['tau_x'], cs.config['tau_z'], True, None, 0.05, keys)
    return max(float(jnp.max(jnp.abs(u - v))) for u, v in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))
