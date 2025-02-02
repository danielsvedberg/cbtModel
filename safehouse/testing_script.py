import model_functions as mf
import plotting_functions as pf
import pickle as pkl

#load params_nm
with open('params_nm.pkl', 'rb') as f:
    params_nm = pkl.load(f)

####TESTING###############
all_ys, all_xs, all_zs = mf.test_model(params_nm, noise=False)

pf.plot_output(all_ys)
pf.plot_activity_by_area(all_xs, all_zs)
pf.plot_cue_algn_activity(all_xs, all_zs)

v_response_times = mf.get_response_times(all_ys, exclude_nan=True)
pf.plot_response_times(v_response_times)

response_times = mf.get_response_times(all_ys, exclude_nan=False)
d1d2_ratio = mf.get_d1_d2_ratio(all_xs, 0,200)
pf.plot_ratio_rt_correlogram(d1d2_ratio, response_times)

pf.plot_binned_responses(all_ys, all_xs, all_zs)
