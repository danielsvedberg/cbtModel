"""Instrumented copy of cbtModel's mf.multiregion_nmrnn (rank-2 DA gain), ported from cbtModel1D seeds/2k_tau tooling.
Model: s = (s_d1, s_d2) = da_signals(m, c, x_nm) = tanh(exc(m) @ nln(x_nm) + nm_bias(c)); dSPN rows get log-gain +s_d1,
iSPN rows -s_d2, applied to the cortex->BG weights and to the BG decay tau; no gain on BG recurrence; no excitability traces.
model_functions.py itself is untouched; verify() checks that with every hook off the output is bit-identical to
jit(mf.batched_nm_rnn).

Hooks, each a per-step array over the T steps (NaN = free / model value):
  nm_in  (T, n_nm) : added to x_nm as (1/tau_z) * nm_in, like the striatal opto input
  s_clamp (T, 2)   : overrides the DA signals (s_d1, s_d2)
  g_ctx  (T, 2)    : overrides the (dSPN, iSPN) gains on cortex->BG input and BG tau (gain = exp(log-gain))
  g_rec  (T,)      : kept for interface compatibility; ignored (the model has no gain on BG recurrence)
run_batch() runs the model's own (patched) source when only nm_in is used, so it is exact; the hand-written
run_one is used only for the s_clamp / g_ctx hooks (verify_handwritten(): ~3e-6).
step_map() gives the noise-free one-step map of the 60 non-SNc states with s held fixed, for Jacobians.
cbtModel1D's scaling_analysis.main() deep-dive assumes a scalar s and is not ported; scaling_screen uses only nm_in.
"""
import math, inspect
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
import model_functions as mf
from model_functions import nln, exc, inh
import config_script as cs

N_BG, N_NM = cs.config['n_bg'], cs.config['n_nm']
N1 = N_BG // 2
T = cs.config['T']
TAU_X, TAU_Z = cs.config['tau_x'], cs.config['tau_z']
NAN_T = jnp.full((T,), jnp.nan)
_src = inspect.getsource(mf.multiregion_nmrnn)
for _anchor in ['U = jnp.zeros((num_bg_cells, 2)).at[:n_d1_cells, 0].set(1.0).at[n_d1_cells:, 1].set(-1.0)',
                's = da_signals(m, c, x_nm)', 'G_bg_e = jnp.exp(l_bg[:, None] @ V_c.T)', 'g_bg_tau = jnp.exp(l_bg)',
                'bg_rec = (inh(J_bg)) @ nln(x_bg)']:
    assert _anchor in _src, f'model source changed, hooks.py out of date: {_anchor}'
REC_SIGN, TAU_GAIN = 0, True  # reported in metrics_scaling.json
U2 = jnp.zeros((N_BG, 2)).at[:N1, 0].set(1.0).at[N1:, 1].set(-1.0)


def da_signal(p, x_nm):
    return mf.da_signals(p['m'], p['c'], x_nm)  # (2,)


def _bg_ctx_t_update(p, x_bg, x_c, x_t, u, stim, l_bg):
    """cortex and BG updates of one step given the per-SPN log-gains l_bg (num_bg,); same ops as the model."""
    tau = TAU_X
    x_c = (1.0 - (1. / tau)) * x_c + (1. / tau) * p['J_c'] @ nln(x_c)
    x_c += (1. / tau) * p['B_cu'] @ u
    x_c += (1. / tau) * exc(p['B_ct']) @ nln(x_t)
    G_c = jnp.exp(l_bg[:, None] @ jnp.ones((N_BG, 1)).T)
    bg_rec = (inh(p['J_bg'])) @ nln(x_bg)
    bg_ctx = (G_c * exc(p['B_bgc'])) @ nln(x_c)
    x_bg = (1.0 - (1. / (tau * jnp.exp(l_bg)))) * x_bg + (1. / tau) * (bg_rec + bg_ctx)
    x_bg += (1. / tau) * stim
    return x_bg, x_c


def _thal_update(p, x_t, x_bg):
    tau = TAU_X
    x_t = (1.0 - (1. / tau)) * x_t + (1. / tau) * p['J_t'] @ nln(x_t)
    tbg = jnp.concatenate((exc(p['B_tbg'][:, :N1]), inh(p['B_tbg'][:, N1:])), axis=1)
    x_t += (1. / tau) * tbg @ nln(x_bg)
    return x_t


def run_one(p, inputs, stim, nm_in, s_clamp, g_rec, g_ctx, noise_std, rng_key):
    x_bg0, x_c0, x_t0 = cs.x0
    x_nm0 = cs.z0
    rng_key, init_key, step_key = jr.split(rng_key, 3)
    if noise_std > 0:
        x_bg0 += noise_std * jr.normal(init_key, x_bg0.shape)
        x_c0 += noise_std * jr.normal(init_key, x_c0.shape)
        x_t0 += noise_std * jr.normal(init_key, x_t0.shape)
        x_nm0 += noise_std * jr.normal(init_key, x_nm0.shape)

    def _step(carry, xs):
        x_bg, x_c, x_t, x_nm = carry
        u, st, nmi, sc, gr, gc, k = xs
        if noise_std > 0:
            coef = noise_std / math.sqrt(2 * TAU_X)
            x_bg += coef * jr.normal(k, x_bg.shape)
            x_c += coef * jr.normal(k, x_c.shape)
            x_t += coef * jr.normal(k, x_t.shape)
            coefz = noise_std / math.sqrt(2 * TAU_Z)
            x_nm += coefz * jr.normal(k, x_nm.shape)
        s = da_signal(p, x_nm)
        s = jnp.where(jnp.isnan(sc), s, sc)
        l_bg = U2 @ s
        l_hook = jnp.concatenate((jnp.full(N1, jnp.log(gc[0])), jnp.full(N_BG - N1, jnp.log(gc[1]))))
        l_bg = jnp.where(jnp.isnan(l_hook), l_bg, l_hook)
        x_bg, x_c = _bg_ctx_t_update(p, x_bg, x_c, x_t, u, st, l_bg)
        x_nm = (1.0 - (1. / TAU_Z)) * x_nm + (1. / TAU_Z) * p['J_nm'] @ nln(x_nm)
        x_nm += (1. / TAU_Z) * exc(p['B_nmc']) @ nln(x_c)
        x_nm += (1. / TAU_Z) * inh(p['B_nmbg']) @ nln(x_bg)
        x_nm += (1. / TAU_Z) * nmi
        x_t = _thal_update(p, x_t, x_bg)
        y = exc(p['C']) @ nln(x_t) + p['rb']
        return (x_bg, x_c, x_t, x_nm), (y, x_bg, x_c, x_t, x_nm, s, jnp.ones(()), jnp.exp(jnp.stack((l_bg[0], l_bg[-1]))))

    keys = jr.split(step_key, inputs.shape[0])
    _, out = lax.scan(_step, (x_bg0, x_c0, x_t0, x_nm0), (inputs, stim, nm_in, s_clamp, g_rec, g_ctx, keys))
    return out  # y, x_bg, x_c, x_t, x_nm, s, g_rec, g_ctx


_run_batch_handwritten = jax.jit(vmap(run_one, in_axes=(None, 0, 0, 0, 0, 0, 0, None, 0)), static_argnums=(7,))


def _build_exact():
    """mf.multiregion_nmrnn's own source with SNc input appended to the stim channel (as seeds/nm_opto.py does).
    Bit-identical to the model with nm_in = 0; the hand-written run_one above differs by float32 op ordering
    (~1 ulp per step, ~3e-6 in y over a trial), so the free / nm_in-only path uses this instead."""
    src = inspect.getsource(mf.multiregion_nmrnn)
    subs = [('def multiregion_nmrnn(', 'def multiregion_nmrnn_nmin('),
            ('x_bg += (1. / tau_bg) * stim  # simulate stimulation', 'x_bg += (1. / tau_bg) * stim[:num_bg_cells]  # simulate stimulation'),
            ('x_nm += (1. / tau_nm) * inh(B_nmbg) @ nln(x_bg)  # input from BG, inhibitory',
             'x_nm += (1. / tau_nm) * inh(B_nmbg) @ nln(x_bg)  # input from BG, inhibitory\n'
             '        x_nm += (1. / tau_nm) * stim[num_bg_cells:]  # SNc input hook')]
    for old, new in subs:
        assert src.count(old) == 1, f'model source changed, patch anchor not found: {old}'
        src = src.replace(old, new)
    ns = dict(vars(mf))
    exec(src, ns)
    f = vmap(ns['multiregion_nmrnn_nmin'], in_axes=(None, None, None, 0, None, None, None, 0, None, 0))
    return jax.jit(lambda p, inputs, stim_all, noise_std, keys: f(p, cs.x0, cs.z0, inputs, TAU_X, TAU_Z, True, stim_all,
                                                                  noise_std, keys), static_argnums=(3,))


_run_exact = _build_exact()


def run_batch(p, inputs, stim, nm_in, s_clamp, g_rec, g_ctx, noise_std, keys):
    """Same interface and outputs as run_one batched. With s_clamp and g_ctx free (all NaN) it runs the model's own
    code (exact); s is then recomputed from the post-step x_nm (one-step lag, no noise replay), for plotting only."""
    import numpy as np
    if np.isnan(np.asarray(s_clamp)).all() and np.isnan(np.asarray(g_ctx)).all():
        y, (xbg, xc, xt), xnm = _run_exact(p, inputs, jnp.concatenate([stim, nm_in], -1), noise_std, keys)
        s = vmap(vmap(lambda x: da_signal(p, x)))(xnm)
        return y, xbg, xc, xt, xnm, s, jnp.ones(s.shape[:2]), jnp.exp(jnp.stack((s[..., 0], -s[..., 1]), -1))
    return _run_batch_handwritten(p, inputs, stim, nm_in, s_clamp, g_rec, g_ctx, noise_std, keys)


def free_hooks(n):
    nan2 = jnp.full((n, T, 2), jnp.nan)
    return jnp.zeros((n, T, N_BG)), jnp.zeros((n, T, N_NM)), nan2, jnp.tile(NAN_T, (n, 1)), nan2


def cue_inputs(n, start):
    inp, _, _ = mf.self_timed_movement_task(jnp.array([start]), cs.config['T_cue'], cs.config['T_wait'],
                                            cs.config['T_movement'], T)
    return jnp.repeat(inp, n, axis=0)


def verify_handwritten(p, n=20):
    """max |diff| of the hand-written step (used when s_clamp / g_ctx hooks are set) vs the model; ~3e-6 expected."""
    inp = cue_inputs(n, 300)
    keys = jr.split(jr.PRNGKey(0), n)
    worst = 0.0
    for ns in (0.0, cs.test_noise_std):
        ref = jax.jit(lambda q, i, k: mf.batched_nm_rnn(q, cs.x0, cs.z0, i, TAU_X, TAU_Z, True, None, ns, k))(p, inp, keys)
        y, xbg, xc, xt, xnm, *_ = _run_batch_handwritten(p, inp, *free_hooks(n), ns, keys)
        for a, b in zip([ref[0], *ref[1], ref[2]], [y, xbg, xc, xt, xnm]):
            worst = max(worst, float(jnp.max(jnp.abs(a - b))))
    return worst


def verify(p, n=20):
    inp = cue_inputs(n, 300)
    keys = jr.split(jr.PRNGKey(0), n)
    worst = 0.0
    for ns in (0.0, cs.test_noise_std):
        ref = jax.jit(lambda q, i, k: mf.batched_nm_rnn(q, cs.x0, cs.z0, i, TAU_X, TAU_Z, True, None, ns, k))(p, inp, keys)
        y, xbg, xc, xt, xnm, *_ = run_batch(p, inp, *free_hooks(n), ns, keys)
        for a, b in zip([ref[0], *ref[1], ref[2]], [y, xbg, xc, xt, xnm]):
            worst = max(worst, float(jnp.max(jnp.abs(a - b))))
    return worst


def step_map(p, s):
    """Noise-free one-step map of the 60 non-SNc states (x_bg, x_c, x_t) with s = (s_d1, s_d2) fixed and no cue input."""
    def f(v):
        x_bg, x_c, x_t = v[:N_BG], v[N_BG:2 * N_BG], v[2 * N_BG:]
        x_bg, x_c = _bg_ctx_t_update(p, x_bg, x_c, x_t, jnp.zeros(1), jnp.zeros(N_BG), U2 @ jnp.asarray(s))
        x_t = _thal_update(p, x_t, x_bg)
        return jnp.concatenate([x_bg, x_c, x_t])
    return f
