* Changed to sigmoid
* Changed grad norm
* Changed weight initialization
* Changed D1 / D2 ratio calculation, drop baseline subtracting
* Changed task target to 0.25 / 1
* Changed opto strength to 3 * stronger
* Changed activation function: tanh → sigmoid in `nln()`
* Changed gradient clipping: `clip(1.0)` → `clip_by_global_norm(1.0)`
* Changed g_bg and g_nm: 1.4 → 0.5
* Changed tau_x: 10 → 50, tau_z: 100 → 200
* Changed movement threshold in loss: 0.75 → 0.5
* Changed task target: added baseline of 0.25 (target is now 0.25–1.0 instead of 0–1)
* Changed D1/D2 ratio: now D1/(D1+D2) using firing rates (no baseline subtraction), previously D1−D2
* Changed `get_brain_area()`: added `as_rate` flag to apply sigmoid before returning activity
* Changed `get_slope()`: now applies nln() after aligning to cue, before baseline subtracting
* Changed opto strength: dMSN range [0, 0.5] → [0, 1.5], iMSN range [0, 0.25] → [0, 0.75]
* Changed `plot_opto()`: now takes `opto_ys` argument for response time CDFs
* Changed y-axis label: "dSPN-iSPN activity" → "dSPN / iSPN activity"
* Changed `bin_normal_data()`: filters NaN before binning, returns early if insufficient data