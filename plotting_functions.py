import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams.update({'font.size': 8})  #use 8pt font everywhere
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from model_functions import *
from config_script import *


def plot_loss(losses_nm):
    loss_curve_nm = [loss[-1] for loss in losses_nm]
    x_axis = np.arange(len(losses_nm)) * 200

    plt.cla()
    plt.plot(x_axis, np.log10(loss_curve_nm), label='NM RNN')
    plt.ylabel('log10(error)')
    plt.xlabel('iteration')
    plt.legend()
    plt.show()


def plot_output(all_ys):
    # Plot output activity (mean ± SEM)
    #fig = plt.figure(figsize=(4, 3))
    fig, axs = plt.subplots(1, 2, figsize=(4, 2), sharey=True)
    colors = plt.cm.coolwarm(jnp.linspace(0, 1, all_ys.shape[1]))
    ax = axs[0]
    #plot each individual trace first
    for i in range(all_ys.shape[0]):
        for j in range(all_ys.shape[1]):
            ax.plot(all_ys[i, j, :, 0], c=colors[j], linewidth=0.1)
    ax = axs[1]
    mean_ys, sem_ys = compute_mean_sem(all_ys)
    for i in range(mean_ys.shape[0]):
        ax.plot(mean_ys[i, :, 0], c=colors[i])
        ax.fill_between(
            jnp.arange(mean_ys.shape[1]),
            mean_ys[i, :, 0] - sem_ys[i, :, 0],
            mean_ys[i, :, 0] + sem_ys[i, :, 0],
            color=colors[i],
            alpha=0.3,
        )
    plt.title(f'Output (mean ± SEM)')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_activity_by_area(all_xs, all_zs):
    fig = plt.figure(figsize=(8, 6))
    for idx, name in enumerate(['D1', 'D2', 'Cortex', 'Thalamus', 'SNc', 'nm']):
        ax = plt.subplot(2, 3, idx + 1)
        area_activity = get_brain_area(name, all_xs, all_zs)
        mean_act, sem_act = compute_mean_sem(jnp.mean(area_activity, axis=3))  # Avg across neurons
        colors = plt.cm.coolwarm(jnp.linspace(0, 1, mean_act.shape[0]))
        for i in range(mean_act.shape[0]):
            ax.plot(mean_act[i], c=colors[i], label=f'Condition {i}')
            ax.fill_between(
                jnp.arange(mean_act.shape[1]),
                mean_act[i] - sem_act[i],
                mean_act[i] + sem_act[i],
                color=colors[i],
                alpha=0.3,
            )
        ax.set_title(f'{name}')
    plt.suptitle('Aligned to trial start')
    plt.tight_layout()
    plt.show()


def plot_cue_algn_activity(all_xs, all_zs):
    max_T_start = jnp.max(test_start_t)
    new_T = config['T'] - max_T_start
    # Plot activity aligned to cue (mean ± SEM)
    plt.close('all')

    aligned_xs = [[] for n in range(3)]
    aligned_zs = []
    for seed_idx in range(n_seeds):
        for i in range(3):
            aligned_xs[i].append(align_to_cue(all_xs[i][seed_idx], test_start_t, new_T=new_T))
        aligned_zs.append(align_to_cue(all_zs[seed_idx], test_start_t, new_T=new_T))
    aligned_xs = [jnp.array(aligned_x) for aligned_x in aligned_xs]
    aligned_zs = jnp.array(aligned_zs)
    x_axis = (jnp.arange(new_T+100)-100) / 100


    fig, axs = plt.subplots(5, 3, figsize=(6, 8), sharex=True)
    for idx, name in enumerate(['D1', 'D2', 'Cortex', 'Thalamus', 'SNc']):
        area_activity = get_brain_area(name, aligned_xs, aligned_zs)
        colors = plt.cm.coolwarm(jnp.linspace(0, 1, area_activity.shape[1]))
        area_activity = area_activity.mean(axis=-1)
        ax = axs[idx,0]
        for i in range(area_activity.shape[1]):
            condition_activity = area_activity[:,i]
            for j in range(condition_activity.shape[0]):
                ax.plot(x_axis, condition_activity[j], c=colors[i], label=f'Condition {i}', linewidth=0.1)
        ax = axs[idx,1]
        for i in range(area_activity.shape[1]):
            condition_activity = area_activity[:, i]
            mean_act, sem_act = compute_mean_sem(condition_activity)  # (n_conditions, T)
            ax.plot(x_axis, mean_act, c=colors[i], label=f'Condition {i}')
            ax.fill_between(
                x_axis,
                mean_act - sem_act,
                mean_act + sem_act,
                color=colors[i],
                alpha=0.3,
            )
        ax = axs[idx,2]
        mean_1 = jnp.mean(area_activity, axis=1)
        mean_act, sem_act = compute_mean_sem(mean_1)
        ax.plot(x_axis, mean_act, c='black')
        ax.fill_between(
            x_axis,
            mean_act - sem_act,
            mean_act + sem_act,
            color='black',
            alpha=0.3,
        )

        #ymin = jnp.min(mean_act - sem_act)
        #ymax = jnp.max(mean_act + sem_act)
        #ax.vlines(config['T_cue'], ymin, ymax, linestyles='dashed', label='Cue')
        #ax.vlines(config['T_cue'] + config['T_wait'], ymin, ymax, linestyles='dashed', label='Wait')
        #ax.vlines(
        #    config['T_cue'] + config['T_wait'] + config['T_movement'],
        #    ymin,
        #    ymax,
        #    linestyles='dashed',
        #    label='Movement',
        #)
        ax.set_title(f'{name} (aligned to cue')
    plt.suptitle('Aligned to cue')
    plt.tight_layout()
    plt.show()


def plot_response_times(valid_response_times):
    max_response_time = jnp.max(valid_response_times)
    # Plot the distribution
    beh_start_t = 0
    x_axis = jnp.linspace(0, max_response_time, 100)

    fig = plt.figure(figsize=(1.8, 1.8))
    plt.hist(valid_response_times, bins=20, color='blue', alpha=0.7, edgecolor='black',density=True
             )
    plt.xlabel('time after cue (s)')
    plt.ylabel('Density')
    #plt.title('Distribution of Response Times')
    plt.tight_layout()
    #set x axis lower limit to zero
    plt.xlim(0, max_response_time)
    plt.xticks([0, 3])
    plt.axvspan(3, x_axis[-1], color='green', alpha=0.2)
    plt.show()
    save_fig(fig, 'response_times_hist')
    # Sort the response times
    sorted_response_times = jnp.sort(valid_response_times)

    # Compute the cumulative proportion of responses
    cumulative_proportion = jnp.arange(1, len(sorted_response_times) + 1) / len(sorted_response_times)

    # Plot the cumulative psychometric curve
    fig = plt.figure(figsize=(1.8, 1.8))
    plt.plot(sorted_response_times, cumulative_proportion, marker='o', color='blue', alpha=0.7)
    plt.axvspan(3, 5, color='green', alpha=0.2)
    plt.xlim(0, 5)
    plt.xticks([0, 3])
    plt.yticks([0, 0.5, 1])
    plt.xlabel('time after cue (s)')
    plt.ylabel('Proportion of Responses\n(CDF)')
    plt.tight_layout()
    plt.show()
    save_fig(fig, 'response_times_cdf')


def truncate_colormap(cmap, minval=0.0, maxval=1.0, alpha=1, n=100):

    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_binned_responses(all_ys, all_xs, all_zs):
    cue_start_t = 0
    cue_end_t = (cue_start_t + config['T_cue']) / 100
    beh_start_t = config['T_wait'] / 100
    beh_end_t = (config['T_wait'] + config['T_movement']) / 100
    n_conditions = all_ys.shape[1]

    response_times = jnp.full((n_seeds, n_conditions), jnp.nan)  # Default to NaN if no response is detected

    for seed_idx in range(n_seeds):
        for condition_idx in range(n_conditions):
            cue_end = test_start_t[condition_idx] + config['T_cue']
            post_cue_activity = all_ys[seed_idx, condition_idx, cue_end:]  # Activity after the cue
            response_idx = jnp.argmax(post_cue_activity[:, 0] > 0.5)  # Find first timestep where y > 0.5
            if post_cue_activity[response_idx, 0] > 0.5:
                response_times = response_times.at[seed_idx, condition_idx].set(response_idx * 0.01)

    # Define the response time bins (left closed, right open)
    bin_boundaries = [2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
    bin_labels = [f'{bin_boundaries[i]}-{bin_boundaries[i + 1]}' for i in range(len(bin_boundaries) - 1)]

    # Initialize lists for binning the xs, ys, zs data
    binned_xs = [[[] for n in range(3)] for _ in bin_labels]
    binned_ys = [[] for _ in bin_labels]
    binned_zs = [[] for _ in bin_labels]
    binned_response_times = [[] for _ in bin_labels]

    # Assign each trial to a bin based on its response time

    for seed_idx in range(n_seeds):

        aligned_xs = [align_to_cue(all_x[seed_idx], test_start_t, new_T=500) for all_x in all_xs]
        aligned_zs = align_to_cue(all_zs[seed_idx], test_start_t, new_T=500)
        aligned_ys = align_to_cue(all_ys[seed_idx], test_start_t, new_T=500)

        for condition_idx in range(n_conditions): #for all cue start times
            response_time = response_times[seed_idx, condition_idx]

            # Find the corresponding bin for the current response time
            for bin_idx, (lower, upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):
                if lower <= response_time < upper:
                    for i in range(3):
                        binned_xs[bin_idx][i].append(aligned_xs[i][condition_idx])
                    binned_ys[bin_idx].append(aligned_ys[condition_idx])
                    binned_zs[bin_idx].append(aligned_zs[condition_idx])
                    binned_response_times[bin_idx].append(response_time)
                    break

    # Convert lists to arrays
    binned_xs = [[jnp.array(bin_data) for bin_data in bin_xs] for bin_xs in binned_xs]
    binned_ys = [jnp.array(bin_data) for bin_data in binned_ys]
    binned_zs = [jnp.array(bin_data) for bin_data in binned_zs]

    # Print the shapes
    print("Shape of binned_xs:", [bin_data.shape for bin_data in binned_xs[0]])
    print("Shape of binned_ys:", [bin_data.shape for bin_data in binned_ys])

    cmap = plt.cm.get_cmap('turbo')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(bin_labels) - 1)
    # Plot output activity (mean ± SEM) for each response time bin
    fig = plt.figure(figsize=(2, 2))
    # Plot the activity for each response time bin
    for bin_idx, bin_data in enumerate(binned_ys):
        if len(bin_data) == 0:  # Skip empty bins
            continue
        # Compute mean and SEM for the current bin
        mean_ys, sem_ys = compute_mean_sem(bin_data)  # Compute mean and SEM across trials

        # Plot each bin with mean ± SEM
        ax = plt.subplot(1, 1, 1)  # Plot on a single axis
        x_axis = (jnp.array(range(mean_ys.shape[0])) - 100) / 100
        color = cmap(norm(bin_idx))
        ax.plot(x_axis, mean_ys[:, 0], label=f'{bin_labels[bin_idx]}', c=color)

        # Plot the shaded region representing SEM
        ax.fill_between(
            x_axis,
            mean_ys[:, 0] - sem_ys[:, 0],
            mean_ys[:, 0] + sem_ys[:, 0],
            color=color,
            alpha=0.3,
        )

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Necessary for creating a color bar
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Response time from cue (s)", rotation=270, labelpad=5)
    cbar.set_ticks([0, len(bin_labels) - 1])
    cbar.set_ticklabels([str(bin_boundaries[0]), str(bin_boundaries[-1])])

    ax.set_xlabel('time after cue (s)')
    ax.set_ylabel('activity')
    ax.axvspan(cue_start_t, cue_end_t, color='red', alpha=0.2)
    ax.axvspan(beh_start_t, x_axis[-1], color='green', alpha=0.2)
    plt.tight_layout()
    plt.show()
    save_fig(fig, 'clue_aligned_binned_responses')

    # Plot activity in each brain area for different response time bins (mean ± SEM) using binned xs and zs
    # Define the brain areas to plot
    brain_areas = ['D1', 'D2', 'Cortex', 'Thalamus', 'SNc']  #, 'nm']
    brain_area_names = ['dSPN', 'iSPN', 'Cortex', 'Thalamus', 'SNc']  #, 'NM']
    cmap = plt.cm.get_cmap('turbo')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(bin_labels) - 1)
    # Loop through each brain area
    #fig, axs = plt.subplots(6, 1, figsize=(2, 4), sharex=True)
    fig, axs = plt.subplots(
        len(brain_areas),
        1,
        figsize=(2, 5), sharex=True)
    n_brain_areas = len(brain_areas)
    for idx, name in enumerate(brain_areas):
        ax = axs[idx]

        # Collect the activity data from all bins and align to cue
        for bin_idx in range(len(binned_xs)):
            xs_bin = binned_xs[bin_idx]  # tuple of n_trials * T * n_neurons
            zs_bin = binned_zs[bin_idx]

            # Get the brain area activity (aligning to the cue)
            area_activity = get_brain_area(name, xs=xs_bin, zs=zs_bin)  # trials * T * N

            # Compute mean and SEM for the current bin
            mean_area_activity = jnp.mean(area_activity, axis=-1)  # trials * T

            # Plot each bin with mean ± SEM
            mean_act, sem_act = compute_mean_sem(mean_area_activity)  # T
            x_axis = (jnp.array(range(mean_act.shape[0])) - 100) / 100
            mask = (x_axis > -0.5) & (x_axis < 5)
            x_axis = x_axis[mask]
            mean_act = mean_act[mask]
            sem_act = sem_act[mask]
            color = cmap(norm(bin_idx))
            ax.plot(x_axis, mean_act, label=f'{bin_labels[bin_idx]}', c=color)
            ax.fill_between(
                x_axis,
                mean_act - sem_act,
                mean_act + sem_act,
                alpha=0.3,
                color=color)
        if idx == n_brain_areas:
            ax.set_xlabel('time after cue (s)')
        ax.set_xticks([0, 3])
        # create a second axis on left, label it with the brain area
        ax2 = ax.twinx()
        nm = brain_area_names[idx]
        ax2.set_ylabel(f'{nm}', rotation=270, labelpad=10)
        # get rid of ticks and labels for the second axis
        ax2.set_yticks([])
        # Add vertical lines for cue, wait, and movement phases

        ax.axvspan(cue_start_t, cue_end_t, color='red', alpha=0.2)
        ax.axvspan(beh_start_t, x_axis[-1], color='green', alpha=0.2)

        if name == 'SNc':
            ax.set_xlabel('time after cue (s)')
        ax.set_ylabel('activity')
    '''
    # Add a horizontal colorbar in the last subplot
    cbar_ax = axs[-1]  # Select the last subplot for the colorbar
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Response Time Bins")
    cbar.set_ticks([0, len(bin_labels) - 1])
    cbar.set_ticklabels([bin_labels[0], bin_labels[-1]])
    '''
    plt.tight_layout()
    plt.show()
    #save as .svg and .png
    save_fig(fig, 'binned_responses_by_area')
def plot_opto_inh(opto_ys, opto_xs, opto_zs, newT=900):
    label_list = ['control', 'inh. dSPN', 'inh. iSPN']
    inh_ys = opto_ys[0:3]
    inh_xs = opto_xs[0:3]
    inh_zs = opto_zs[0:3]
    #colors should be black for control, green for dsPN, and red for iSPN
    colors = ['black', 'green', 'red']
    brain_areas = ['D1', 'D2', 'Cortex', 'Thalamus', 'SNc']
    brain_area_names = ['dSPN', 'iSPN', 'Cortex', 'Thalamus', 'SNc']

    '''
    #plot histogram of response times for each label_list
    plt.figure(figsize=(2, 2))
    for stim_idx, label in enumerate(label_list):
        ys = inh_ys[stim_idx]
        response_times = get_response_times_opto(ys, exclude_nan=False)
        plt.hist(response_times, bins=20, color=colors[stim_idx], alpha=0.7, edgecolor='black',density=True)
    plt.xlabel('Response time from cue (s)')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()
    plt.savefig('response_times_hist_opto_inh.png', dpi=900)
    plt.savefig('response_times_hist_opto_inh.svg')
    '''
    fig = plt.figure(figsize=(1.8, 1.8))
    max_response_time = 0
    for stim_idx, label in enumerate(label_list):
        ys = inh_ys[stim_idx]
        response_times = get_response_times_opto(ys, exclude_nan=True)
        #max_resp = jnp.max(response_times)
        #if max_resp > max_response_time:
        #    max_response_time = max_resp
        sorted_response_times = jnp.sort(response_times)
        cumulative_proportion = jnp.arange(1, len(sorted_response_times) + 1) / len(sorted_response_times)
        plt.plot(sorted_response_times, cumulative_proportion, marker='o', color=colors[stim_idx], alpha=0.7)
    #set x axis lower limit to zero
    plt.axvspan(3, 6, color='green', alpha=0.2)
    plt.xlim(0, 6)
    plt.xticks([0, 3, 6])
    plt.yticks([0, 0.5, 1])
    plt.xlabel('time after cue (s)')
    plt.ylabel('Proportion of Responses\n(CDF)')
    plt.tight_layout()
    plt.show()
    save_fig(fig, 'response_times_cdf_opto_inh')

    cue_start_t = 0
    cue_end_t = (cue_start_t + config['T_cue']) / 100
    beh_start_t = config['T_wait'] / 100
    beh_end_t = (config['T_wait'] + config['T_movement']) / 100
    ops = (opto_start - opto_tstart) / 100
    ope = (opto_end - opto_tstart) / 100

    resp_times = []
    for stim_idx, label in enumerate(label_list):
        ys = inh_ys[stim_idx]
        resp_times.append(get_response_times_opto(ys, exclude_nan=False))

    fig, axs = plt.subplots(5, 1, figsize=(2, 5), sharex=True)
    for idx, name in enumerate(brain_areas):
        ax = axs[idx]

        for stim_idx, label in enumerate(label_list):
            resps = resp_times[stim_idx]
            area_activity = get_brain_area(name, inh_xs[stim_idx], inh_zs[stim_idx]).mean(
                axis=-1)
            #get all indices where resps is nan, and remove the corresponding indices in the 0th dim of area_activity
            mask = jnp.isnan(resps)
            area_activity = area_activity[~mask]

            mean_activity, sem_activity = compute_mean_sem(area_activity)
            nbin = len(mean_activity)
            t = jnp.linspace(-opto_tstart / 100, (newT - opto_tstart) / 100, nbin)
            mask = (t > -0.5) & (t < 6)
            t = t[mask]
            mean_activity = mean_activity[mask]
            sem_activity = sem_activity[mask]
            ax.plot(t, mean_activity, c=colors[stim_idx], label=label)
            ax.fill_between(t, mean_activity - sem_activity,
                            mean_activity + sem_activity, color=colors[stim_idx], alpha=0.3)
            ax.set_ylabel('activity')
            #set x ticks to be 0, 3
            ax.set_xticks([0, 3, 6])
            #create a second axis on left, label it with the brain area
            ax2 = ax.twinx()
            nm = brain_area_names[idx]
            ax2.set_ylabel(f'{nm}', rotation=270, labelpad=10)
            #get rid of ticks and labels for the second axis
            ax2.set_yticks([])

        #if idx == 0:
        #    ax.legend(loc='upper right', bbox_to_anchor=(0.75, 2.4))
        ax.axvspan(cue_start_t, cue_end_t, color='red', alpha=0.2)
        ax.axvspan(beh_start_t, t[-1], color='green', alpha=0.2)
        ax.axvspan(ops, ope, color='dodgerblue', alpha=0.2)
        if idx == len(brain_areas)-1:
            ax.set_xlabel('time after cue (s)')
    plt.tight_layout()
    plt.show()
    save_fig(fig, 'opto_inh_demo')

def plot_opto_stim(opto_ys, opto_xs, opto_zs, newT=900):
    label_list = ['control', 'stim. dSPN', 'stim. iSPN']
    stim_ys = [opto_ys[0]] + opto_ys[3:5]
    stim_xs = [opto_xs[0]] + opto_xs[3:5]
    stim_zs = [opto_zs[0]] + opto_zs[3:5]
    #colors should be black for control, green for dsPN, and red for iSPN
    colors = ['black', 'green', 'red']
    brain_areas = ['D1', 'D2', 'Cortex', 'Thalamus', 'SNc']
    brain_area_names = ['dSPN', 'iSPN', 'Cortex', 'Thalamus', 'SNc']



    fig = plt.figure(figsize=(1.8, 1.8))
    max_response_time = 0
    for stim_idx, label in enumerate(label_list):
        ys = stim_ys[stim_idx]
        response_times = get_response_times_opto(ys, exclude_nan=True)
        sorted_response_times = jnp.sort(response_times)
        cumulative_proportion = jnp.arange(1, len(sorted_response_times) + 1) / len(sorted_response_times)
        plt.plot(sorted_response_times, cumulative_proportion, marker='o', color=colors[stim_idx], alpha=0.7)
    #set x axis lower limit to zero
    plt.xlim(0, 6)
    plt.xticks([0, 3])
    plt.yticks([0, 0.5, 1])
    plt.xlabel('time after cue (s)')
    plt.ylabel('Proportion of Responses\n(CDF)')
    plt.tight_layout()
    plt.show()
    save_fig(fig, 'response_times_cdf_opto_stim')

    cue_start_t = 0
    cue_end_t = (cue_start_t + config['T_cue']) / 100
    beh_start_t = config['T_wait'] / 100
    beh_end_t = (config['T_wait'] + config['T_movement']) / 100
    ops = (opto_start - opto_tstart) / 100
    ope = (opto_end - opto_tstart) / 100

    resp_times = []
    for stim_idx, label in enumerate(label_list):
        ys = stim_ys[stim_idx]
        resp_times.append(get_response_times_opto(ys, exclude_nan=False))


    fig, axs = plt.subplots(3, 1, figsize=(2, 3), sharex=True)
    for idx, name in enumerate(brain_areas):
        ax = axs[idx]

        for stim_idx, label in enumerate(label_list):
            resps = resp_times[stim_idx]
            area_activity = get_brain_area(name, stim_xs[stim_idx], stim_zs[stim_idx]).mean(
                axis=-1)
            #get all indices where resps is nan, and remove the corresponding indices in the 0th dim of area_activity
            mask = jnp.isnan(resps)
            area_activity = area_activity[~mask]

            mean_activity, sem_activity = compute_mean_sem(area_activity)
            nbin = len(mean_activity)
            t = jnp.linspace(-opto_tstart / 100, (newT - opto_tstart) / 100, nbin)
            mask = (t > -0.5) & (t < 5)
            t = t[mask]
            mean_activity = mean_activity[mask]
            sem_activity = sem_activity[mask]
            ax.plot(t, mean_activity, c=colors[stim_idx], label=label)
            ax.fill_between(t, mean_activity - sem_activity,
                            mean_activity + sem_activity, color=colors[stim_idx], alpha=0.3)
            ax.set_ylabel('activity (AU)')
            #set x ticks to be 0, 3
            ax.set_xticks([0, 3])
            #create a second axis on left, label it with the brain area
            ax2 = ax.twinx()
            nm = brain_area_names[idx]
            ax2.set_ylabel(f'{nm}', rotation=270, labelpad=10)
            #get rid of ticks and labels for the second axis
            ax2.set_yticks([])

        #if idx == 0:
        #    ax.legend(loc='upper right', bbox_to_anchor=(0.75, 2.4))
        ax.axvspan(cue_start_t, cue_end_t, color='red', alpha=0.2)
        ax.axvspan(beh_start_t, t[-1], color='green', alpha=0.2)
        ax.axvspan(ops, ope, color='dodgerblue', alpha=0.2)
        if idx == len(brain_areas)-1:
            ax.set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    save_fig(fig, 'opto_stim_demo')


def plot_opto(all_ys_list, all_xs_list, all_zs_list, newT=600):
    #inhibits = ['control', 'inh. dSPN', 'inh. iSPN']
    #stims = ['control', 'stim dSPN', 'stim iSPN']
    dp = ['control', 'inh.', 'stim.']
    ip = ['control', 'inh.', 'stim.']
    #label_lists = [inhibits, stims]
    label_lists = [dp, ip]

    inh_ys = all_ys_list[0:3]
    stim_ys = [all_ys_list[0]] + all_ys_list[3:5]
    all_ys_list = [inh_ys, stim_ys]

    inh_xs = all_xs_list[0:3]
    stim_xs = [all_xs_list[0]] + all_xs_list[3:5]
    all_xs_list = [inh_xs, stim_xs]

    inh_zs = all_zs_list[0:3]
    stim_zs = [all_zs_list[0]] + all_zs_list[3:5]
    all_zs_list = [inh_zs, stim_zs]

    # ['control', 'suppress direct pathway', 'suppress indirect pathway']
    # need a black color, a yellow green color, and a red-orange color
    #colors1 = ['black', 'olive', 'darkorange']
    colors1 = ['black', 'olive', 'darkcyan']
    # ['control', 'enhance direct pathway', 'enhance indirect pathway']
    # need a black color, a blue-green color, and a purple/violet color
    #colors2 = ['black', 'darkcyan', 'mediumvioletred']
    colors2 = ['black', 'darkorange', 'mediumvioletred']
    colors_list = [colors1, colors2]

    brain_areas = ['behavior', 'D1', 'D2', 'Cortex', 'Thalamus', 'SNc']  #, 'nm']
    brain_area_labs = ['behavior', 'dSPNs', 'iSPNs', 'Cortex', 'Thalamus', 'SNc']  # , 'nm']
    #titles = ['inhibition', 'stimulation']
    titles = ['dSPN opto', 'iSPN opto']
    cue_start_t = 0
    cue_end_t = (cue_start_t + config['T_cue']) / 100
    beh_start_t = ((config['T_wait'] + opto_tstart) - opto_tstart) / 100
    beh_end_t = cue_start_t + 4  #beh_start_t + #config['T_movement'] / 100
    ops = (opto_start - opto_tstart) / 100
    ope = (opto_end - opto_tstart) / 100
    # Plot output activity with error bars

    # Plot mean activity in each brain area with error bars
    fig, axs = plt.subplots(6, 2, figsize=(4, 6), sharex=True, sharey=False)
    for opidx, label_list in enumerate(label_lists):
        print(label_list)
        xs_sub = all_xs_list[opidx]
        zs_sub = all_zs_list[opidx]
        colors = colors_list[opidx]
        for idx, name in enumerate(brain_areas):
            ba_lab = brain_area_labs[idx]
            ax = axs[idx, opidx]
            if idx == 0:
                ax.title.set_text(f'{titles[opidx]}')
                ys_sub = all_ys_list[opidx]
                colors = colors_list[opidx]
                for stim_idx, label in enumerate(label_list):
                    ys_mean, ys_sem = compute_mean_sem(ys_sub[stim_idx][:, :, 0])  # Mean and SEM over seeds
                    nbin = len(ys_mean)
                    # make a time vector from with nbin indices from -1 to 6
                    t = jnp.linspace(-opto_tstart / 100, (newT - opto_tstart) / 100, nbin)
                    # get indices where t > -0.5 and t < 6
                    mask = (t > -0.5) & (t < 4)
                    t = t[mask]
                    ys_mean = ys_mean[mask]
                    ys_sem = ys_sem[mask]
                    ax.plot(t, ys_mean, c=colors[stim_idx], label=label)
                    ax.fill_between(t, ys_mean - ys_sem, ys_mean + ys_sem, color=colors[stim_idx], alpha=0.3)

            else:
                for stim_idx, label in enumerate(label_list):
                    area_activity = get_brain_area(name, xs_sub[stim_idx], zs_sub[stim_idx]).mean(
                        axis=-1)  # (num_seeds, ...)
                    mean_activity, sem_activity = compute_mean_sem(area_activity)  # Mean and SEM over seeds and trials
                    nbin = len(mean_activity)
                    t = jnp.linspace(-opto_tstart / 100, (newT - opto_tstart) / 100, nbin)
                    mask = (t > -0.5) & (t < 4)
                    t = t[mask]
                    mean_activity = mean_activity[mask]
                    sem_activity = sem_activity[mask]
                    ax.plot(t, mean_activity, c=colors[stim_idx], label=label)
                    ax.fill_between(t, mean_activity - sem_activity,
                                    mean_activity + sem_activity, color=colors[stim_idx], alpha=0.3)

            if opidx == 0:
                ax.set_ylabel('activity (AU)')
            if opidx == 1:
                #create a second axis on left, label it with the brain area
                ax2 = ax.twinx()
                ax2.set_ylabel(f'{ba_lab}', rotation=270, labelpad=15)
                #get rid of ticks and labels for the second axis
                ax2.set_yticks([])
            #set the y ticks to [0.0. 0.1, 0.2, 0.3, 0.4]
            #ax.set_yticks([0.0, 0.2, 0.4, 0.6])
            ax.axvspan(cue_start_t, cue_end_t, color='red', alpha=0.2)
            ax.axvspan(beh_start_t, beh_end_t, color='green', alpha=0.2)
            ax.axvspan(ops, ope, color='gray', alpha=0.2)
            #if idx == len(brain_areas)-1:
            if idx == 0:
                ax.legend(loc='upper right', bbox_to_anchor=(0.75, 2.4))
                #ax.legend(loc='lower center', bbox_to_anchor=(0.25, -0.5))
    axs[-1, 1].set_xlabel('Time (s)')
    axs[-1, 0].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    save_fig(fig, 'opto_demo')

def plot_ratio_rt_correlogram(d1d2_ratio, response_times):
    # plot a correlogram of the d1d2 ratio and response times,
    # assuming both are arrays of equal length
    #mark any outliers in d1d2 ratio as na
    #first, identify outliers in d2d2_ratio
    z_scores = abs((d1d2_ratio - d1d2_ratio.mean()) / d1d2_ratio.std())
    outliers = z_scores > 3
    d1d2_ratio = jnp.where(outliers, jnp.nan, d1d2_ratio)

    rt_z_scores = abs((response_times - response_times.mean()) / response_times.std())
    rt_outliers = rt_z_scores > 3
    response_times = jnp.where(rt_outliers, jnp.nan, response_times)
    #mark all entries in d1d2 ratio that are nan in response times as nan
    d1d2_ratio = jnp.where(jnp.isnan(response_times), jnp.nan, d1d2_ratio)
    #mark all entries in response times that are nan in d1d2 ratio as nan
    response_times = jnp.where(jnp.isnan(d1d2_ratio), jnp.nan, response_times)
    #remove all nan values from d1d2_ratio and response times
    d1d2_ratio = d1d2_ratio[~jnp.isnan(d1d2_ratio)]
    #log the d1d2_ratio
    #d1d2_ratio = jnp.log(d1d2_ratio)
    response_times = response_times[~jnp.isnan(response_times)]
    #convert response times and d1d2_ratio to numpy arrays
    response_times = np.array(response_times)
    d1d2_ratio = np.array(d1d2_ratio)

    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(2, 3))
    ax.scatter(response_times, d1d2_ratio, c='blue', alpha=0.1)
    #calculate and plot a linear regression line for the regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(response_times, d1d2_ratio)
    #create a p value string: if p_value is 0, print p<0.001, else print p={p_value}
    if p_value < 0.001:
        p_val_str = 'p<0.001'
    else:
        p_value = round(p_value, 3)
        p_val_str = f'p={p_value}'

    print(p_value)
    #calculate a line of best fit
    line = slope * response_times + intercept
    ax.plot(response_times, line, c='black', label=f'y={slope:.2f}x+{intercept:.2f}')
    #above the line, plot the r2 and p value
    text_str = f'R^2={r_value:.2f}\n{p_val_str}'
    ax.text(0.9, 0.9, text_str, ha='right', va='top', transform=ax.transAxes)
    #set the y limit from 0 to 5
    ax.set_ylim(2, 5)
    ax.set_yticks([2, 3, 4, 5])
    ax.set_ylabel('dSPN/iSPN activity')
    ax.set_xlabel('time after cue (s)')
    #ax.set_title('D1:D2 ratio vs. Response time')
    plt.tight_layout()
    plt.show()
    plt.savefig('d1d2_ratio_vs_rt.png', dpi=900)
    plt.savefig('d1d2_ratio_vs_rt.svg')

def save_fig(fig, name):
    svgname = svg_folder + '/' + name + '.svg'
    pngname = png_folder + '/' + name + '.png'
    fig.savefig(svgname)
    fig.savefig(pngname, dpi=900)