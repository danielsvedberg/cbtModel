import model_functions_with_ffin as mf
import plotting_functions as pf
import pickle as pkl
import jax.numpy as jnp

#load params_nm
with open('params_nm_ffin.pkl', 'rb') as f:
    params_nm = pkl.load(f)

####TESTING###############
all_ys, all_xs, all_zs = mf.test_model(params_nm, noise=True)

pf.plot_output(all_ys)
pf.plot_activity_by_area(all_xs, all_zs)
pf.plot_cue_algn_activity(all_xs, all_zs)

v_response_times = mf.get_response_times(all_ys, exclude_nan=True)
assert len(v_response_times) > 0, "No valid response times found. Check the output."
pf.plot_response_times(v_response_times)

response_times = mf.get_response_times(all_ys, exclude_nan=False)
d1d2_ratio = mf.get_d1_d2_ratio(all_xs, avg_time=True, remove_outliers=False)

# Verify D1/D2 ratio uses firing rates correctly
print("\n=== D1/D2 Ratio Verification ===")
print(f"Shape: {d1d2_ratio.shape}")
print(f"Range: [{d1d2_ratio.min():.4f}, {d1d2_ratio.max():.4f}]")
print(f"Mean: {d1d2_ratio.mean():.4f}")
print(f"All values in [0,1]? {(d1d2_ratio >= 0).all() and (d1d2_ratio <= 1).all()}")
print(f"Any NaN values? {jnp.isnan(d1d2_ratio).any()}")
print("=" * 35 + "\n")

pf.plot_ratio_rt_correlogram(d1d2_ratio, response_times)
pf.plot_d1d2ratio_SNc_correlogram(d1d2_ratio, all_zs, response_times)
pf.plot_d1d2ratio_slope_correlogram(all_xs, response_times)

pf.plot_binned_responses(all_ys, all_xs, all_zs)


