# GTA VI Twitch CCU forecast — akun 7 (gunfire22 / quiet-heron-15), rebuild

Competition: gta6-twitch-ccu-forecast-01. Forecast Twitch concurrent viewers in
the Grand Theft Auto VI category, hours 2-24 of release day (2026-11-19 UTC),
as five quantiles per hour.

Model: per-hour Theil-Sen log regression with a sharp spread. For each hour h,
fit log(viewers_h) = a_h + b_h * log(viewers_1) across the rolling-midnight
comparables, slope = median of pairwise slopes, intercept = median of
y_i - b*x_i; predict at GTA VI's hour-1 print. Spread = residual std scaled
by 0.75, clamped to [0.20, 1.00]. Quantiles: q_tau(h) = exp(pred_h + z_tau * s_h).

This rebuild replaces the account's first entry (rolling-median quantiles,
practice loss 0.259). Leave-one-out backtest over the 12 rolling comparables:
pinball loss 0.241 vs the house baseline's 0.357 — the best estimator in the
backtest, and the sharp spread is the winning move on the practice board
(sharp-spread OLS 0.157 vs plain OLS 0.162).

Run: `MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py`
