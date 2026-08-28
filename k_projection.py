"""
Philadelphia Phillies, Research and Information
Quantitative Analyst Associate Trial Project
Projecting Pitcher Strikeout Percentage

Projects each pitcher's K% for a target season from the player-season panel in
k_2026.csv (Season, Name, PlayerId, Team, K%, TBF, Stuff+, Age).

The full methodology is documented in output/methodology_2025.pdf, generated
separately by make_report.py. This file contains the analysis only.

USAGE
    python k_projection.py                 # project 2025, score, write outputs
    python k_projection.py --self-test     # prove the leakage guard holds

LEAKAGE GUARD
    The 2025 rows in k_2026.csv are never used to fit anything. They are read
    once, at the end, to score projections that were already written. Every
    fitted object (league curve, shrinkage constant, aging curve, Stuff+ prior
    and its calibration, ridge, LightGBM, recency weights) is estimated on
    2021-2024 only. `--self-test` verifies this by re-running the pipeline
    against a copy of the file with all 2025 rows deleted and asserting the
    projections are identical.

    The script is parameterised by target season: `--target-season T` re-runs
    the whole pipeline for any season the file can support. For target T it
    reads only Season <= T-1, enforced in `build_features` and asserted at
    runtime. The rolling backtest re-fits the same pipeline internally for each
    earlier season, including its recency-weight search (see `run_backtest`).

REPRODUCIBILITY
    Deterministic: seeds fixed, no network access, no data outside k_2026.csv.

CITATIONS  (per instruction 1f)
    [1] Tango, T. Marcel The Monkey Forecasting System.
        http://www.tangotiger.net/archives/stud0346.shtml
    [2] FanGraphs Sabermetrics Library, Projection Systems.
        https://library.fangraphs.com/principles/projections/
    [3] FanGraphs Sabermetrics Library, Stuff+, Location+, and Pitching+ Primer.
        https://library.fangraphs.com/pitching/stuff-location-and-pitching-primer/
    [4] FanGraphs Sabermetrics Library, Sample Size (Carleton stabilisation
        points; K% ~ 70 BF). https://library.fangraphs.com/principles/sample-size/
    [5] FanGraphs, Pitcher Aging Curves: Starters and Relievers.
        https://blogs.fangraphs.com/pitcher-aging-curves-starters-and-relievers/
    [6] Robinson, D. Understanding empirical Bayes estimation (using baseball
        statistics). http://varianceexplained.org/r/empirical_bayes_baseball/
    [7] scikit-learn Ridge / LightGBM API documentation.
        https://scikit-learn.org/stable/modules/linear_model.html
        https://lightgbm.readthedocs.io/
    [8] Anthropic Claude (Opus) was used as a coding and research assistant:
        literature review, code scaffolding and drafting. All modelling
        decisions, the variance decomposition, the validation design and the
        reported numbers were specified, run and verified by the author.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Default input/output live next to this file, not next to whatever directory
# the script happens to be launched from. A relative default would resolve
# against the caller's cwd, which fails outright when that cwd is read-only
# (macOS resolves a bare `output` under / to the signed system volume).
PROJECT_ROOT = Path(__file__).resolve().parent

SEED = 20260826
MIN_AGE_BUCKET_N = 10   # min paired seasons for an age bucket to be trusted
np.random.seed(SEED)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """All tunable constants in one place."""

    data_path: Path = PROJECT_ROOT / "data" / "k_2026.csv"
    out_dir: Path = PROJECT_ROOT / "output"
    target_season: int = 2025

    # --- how many prior seasons feed the weighted baseline -------------------
    n_lag_seasons: int = 3

    # --- position-player filter ---------------------------------------------
    # Position players mopping up in blowouts are in the raw FanGraphs dump.
    # Excluded from training, but still projected and flagged in the output.
    # Stuff+ separates the two populations cleanly on this panel: no pitcher
    # reaching 100 BF has Stuff+ under 71.4, and no row above 30 BF falls in the
    # 60-70 band at all. A floor of 70 therefore catches every position player
    # without touching a genuine pitcher. The usage ceiling is a second guard,
    # set well above any flagged row (max 54 BF) rather than at the boundary --
    # at 30 BF it let position players with heavier mop-up duty through.
    stuff_floor: float = 70.0        # Stuff+ below this => not a real pitcher
    pos_player_max_tbf: int = 100    # ...and only below a genuine workload

    # --- minimum usage to enter model fitting (not prediction) --------------
    min_tbf_train: int = 25

    # --- evaluation reporting thresholds ------------------------------------
    eval_thresholds: tuple = (0, 50, 100, 200)

    # --- recency weights: candidates searched in backtest --------------------
    # Marcel [1] uses 5/4/3 == 1.00/0.80/0.60 normalised; these are fitted
    # rather than assumed, following Steamer [2]. See `select_weights`.
    weight_grid: tuple = (
        (1.00, 0.80, 0.60),   # Marcel 5/4/3
        (1.00, 0.70, 0.45),
        (1.00, 0.60, 0.35),
        (1.00, 0.50, 0.25),
        (1.00, 0.85, 0.70),
        (1.00, 1.00, 1.00),   # unweighted career-to-date
    )

    seed: int = SEED

    def __post_init__(self):
        # A weight vector shorter than n_lag_seasons would leave the surplus
        # lags unmapped in `build_features`; pandas skips the resulting NaNs on
        # .sum(), so those seasons would be silently zero-weighted rather than
        # raising. Catch it here instead.
        bad = [w for w in self.weight_grid if len(w) != self.n_lag_seasons]
        if bad:
            raise ValueError(
                f"weight_grid entries must have exactly n_lag_seasons="
                f"{self.n_lag_seasons} elements; offending entries: {bad}")


# =============================================================================
# 1. LOAD AND VALIDATE
# =============================================================================

def load_data(cfg: Config) -> pd.DataFrame:
    """Read the panel, validate it, and attach derived columns.

    `utf-8-sig` is required: the file ships with a UTF-8 BOM.
    """
    df = pd.read_csv(cfg.data_path, encoding="utf-8-sig")

    expected = {"Season", "Name", "PlayerId", "Team", "K%", "TBF", "Stuff+", "Age"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"k_2026.csv missing expected columns: {missing}")

    df = df.rename(columns={"K%": "k_pct", "Stuff+": "stuff_plus"})

    # --- integrity checks ---------------------------------------------------
    assert df.duplicated(["Season", "PlayerId"]).sum() == 0, \
        "duplicate player-seasons present; multi-team rows would need collapsing"
    assert (df.TBF > 0).all(), "non-positive TBF present"
    assert df.k_pct.between(0, 1).all(), "K% outside [0,1]"
    assert not df.isna().any().any(), "unexpected nulls"

    # Reconstruct the strikeout COUNT. Everything downstream operates on
    # (K, TBF) rather than K%, which is what makes the binomial treatment and
    # the BF-weighted arithmetic correct.
    df["k_count"] = (df.k_pct * df.TBF).round().astype(int)

    # --- position-player flag ----------------------------------------------
    df["is_pos_player"] = (df.stuff_plus < cfg.stuff_floor) & (df.TBF < cfg.pos_player_max_tbf)

    return df.sort_values(["PlayerId", "Season"]).reset_index(drop=True)


# =============================================================================
# 2. LEAGUE ENVIRONMENT
# =============================================================================

def league_environment(df: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """BF-weighted league K% by season, using data at or before `cutoff` only."""
    d = df[(df.Season <= cutoff) & (~df.is_pos_player)]
    lg = (d.groupby("Season")
            .apply(lambda g: pd.Series({
                "lg_k_pct": np.average(g.k_pct, weights=g.TBF),
                "lg_stuff": np.average(g.stuff_plus, weights=g.TBF),
                "total_tbf": g.TBF.sum(),
            }), include_groups=False)
            .reset_index())
    return lg


def project_league_k(lg: pd.DataFrame, target_season: int) -> float:
    """Extrapolate league K% into the target season.

    Weighted linear trend, damped 50% toward the most recent observed value.
    """
    x = lg.Season.values.astype(float)
    y = lg.lg_k_pct.values
    if len(x) < 2:
        return float(y[-1])
    w = np.linspace(1.0, 2.0, len(x))            # recent seasons weighted higher
    slope, intercept = np.polyfit(x, y, 1, w=w)
    trend_pred = slope * target_season + intercept
    last = y[-1]
    return float(0.5 * trend_pred + 0.5 * last)


# =============================================================================
# 3. VARIANCE DECOMPOSITION -> REGRESSION CONSTANT
# =============================================================================

def estimate_shrinkage_k(df: pd.DataFrame, cutoff: int, min_tbf: int = 25) -> dict:
    """Derive the regression-to-the-mean constant k, in batters-faced units.

        Var(observed K%) = Var(true talent) + E[ p(1-p) / TBF ]
        k = p_bar (1 - p_bar) / Var(true talent)

    The beta prior's effective sample size [6], measured from the panel rather
    than assumed as in Marcel [1]. Returns k ~= 77 BF on 2021-2024.
    """
    d = df[(df.Season <= cutoff) & (~df.is_pos_player) & (df.TBF >= min_tbf)]
    w = d.TBF.values
    p_bar = np.average(d.k_pct, weights=w)
    var_obs = np.average((d.k_pct - p_bar) ** 2, weights=w)
    binom_noise = np.average(p_bar * (1 - p_bar) / d.TBF.values, weights=w)
    var_talent = var_obs - binom_noise
    if var_talent <= 0:
        raise RuntimeError("non-positive talent variance; check input data")
    k = p_bar * (1 - p_bar) / var_talent
    return {
        "p_bar": float(p_bar),
        "var_observed": float(var_obs),
        "binomial_noise": float(binom_noise),
        "var_talent": float(var_talent),
        "k_bf": float(k),
        "n": int(len(d)),
    }


def estimate_shrinkage_k_informed(df: pd.DataFrame, cutoff: int, cfg: Config,
                                  stuff_prior, calibration: tuple | None = None) -> dict:
    """Shrinkage constant for the Stuff+-informed prior.

    Same decomposition as `estimate_shrinkage_k`, applied to the residuals that
    Stuff+ does not explain. Var(residual talent) < Var(total talent), so this k
    is larger than the flat-prior k: a more informative prior earns more weight.
    Returns k ~= 136 BF against 77 for the flat prior.
    """
    d = df[(df.Season <= cutoff) & (~df.is_pos_player) & (df.TBF >= cfg.min_tbf_train)].copy()
    w = d.TBF.values.astype(float)
    p_bar = np.average(d.k_pct, weights=w)

    implied_logit = stuff_prior(d.stuff_plus.values, d.Age.values)
    if calibration is not None:
        a_cal, b_cal = calibration
        implied_logit = a_cal + b_cal * implied_logit
    implied_p = inv_logit(implied_logit)

    resid = d.k_pct.values - implied_p
    var_resid_obs = np.average(resid ** 2, weights=w)
    binom_noise = np.average(p_bar * (1 - p_bar) / d.TBF.values, weights=w)
    var_resid_talent = var_resid_obs - binom_noise
    if var_resid_talent <= 0:
        var_resid_talent = 1e-5
    k = p_bar * (1 - p_bar) / var_resid_talent

    var_total_talent = (np.average((d.k_pct - p_bar) ** 2, weights=w) - binom_noise)
    r2_stuff = 1.0 - var_resid_talent / var_total_talent if var_total_talent > 0 else np.nan

    return {"k_bf_informed": float(k),
            "var_resid_talent": float(var_resid_talent),
            "stuff_share_of_talent_var": float(r2_stuff)}


# =============================================================================
# 4. AGING CURVE
# =============================================================================

def fit_aging_curve(df: pd.DataFrame, cutoff: int, lg: pd.DataFrame,
                    min_tbf: int = 100, detrend: bool = True) -> pd.Series:
    """League-detrended delta-method aging curve: expected one-year change in K%.

    Pairs consecutive seasons per pitcher, weights each pair by the harmonic
    mean of its two TBF values, and subtracts that pair's league-wide change so
    environment drift is not attributed to aging [5].

    Survivorship-biased by construction, so it is applied as a one-year nudge
    rather than extrapolated. See the methodology report.
    """
    d = df[(df.Season <= cutoff) & (~df.is_pos_player)]
    cur = d[["Season", "PlayerId", "k_pct", "TBF", "Age"]]
    nxt = cur.copy()
    nxt["Season"] -= 1
    m = cur.merge(nxt, on=["Season", "PlayerId"], suffixes=("", "_next"))
    m = m[(m.TBF >= min_tbf) & (m.TBF_next >= min_tbf)].copy()

    lg_map = lg.set_index("Season").lg_k_pct.to_dict()
    m["lg_delta"] = m.Season.map(lambda s: lg_map.get(s + 1, np.nan) - lg_map.get(s, np.nan))
    m = m.dropna(subset=["lg_delta"])

    if not detrend:
        # The same curve with the league-drift subtraction switched off, on the
        # identical row set. Used only to quantify what detrending is worth --
        # undetrended, league drift reads as apparent aging.
        m["lg_delta"] = 0.0
    m["delta_adj"] = (m.k_pct_next - m.k_pct) - m.lg_delta
    m["w"] = 2.0 / (1.0 / m.TBF + 1.0 / m.TBF_next)

    # Aggregate into age buckets (not per-age means -- single-age cells are thin
    # in a five-season file). Buckets with too little support fall back to the
    # nearest supported bucket rather than to zero.
    edges = [19, 23, 25, 27, 29, 31, 33, 46]
    bucket_vals = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = m[(m.Age > lo) & (m.Age <= hi)]
        val = (float(np.average(g.delta_adj, weights=g.w))
               if len(g) >= MIN_AGE_BUCKET_N else np.nan)
        bucket_vals.append((lo, hi, val, len(g)))

    supported = [b for b in bucket_vals if not np.isnan(b[2])]
    if not supported:
        # Not enough paired seasons yet (happens on the earliest backtest folds).
        # Fall back to a flat, no-aging curve rather than inventing a shape.
        return pd.Series({a: 0.0 for a in range(18, 51)})

    curve = {}
    for lo, hi, val, n in bucket_vals:
        if np.isnan(val):
            nearest = min(supported, key=lambda b: abs((b[0] + b[1]) / 2 - (lo + hi) / 2))
            val = nearest[2]
        for age in range(lo + 1, hi + 1):
            curve[age] = float(val)

    lo_a, hi_a = min(curve), max(curve)
    for age in range(18, 51):
        if age not in curve:
            curve[age] = curve[lo_a] if age < lo_a else curve[hi_a]
    return pd.Series(curve).sort_index()


# =============================================================================
# 5. FEATURE ENGINEERING  (the single place the season cutoff is enforced)
# =============================================================================

def emp_logit(k: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Empirical logit, log((K + 0.5) / (TBF - K + 0.5)).

    The Haldane-Anscombe correction keeps the 306 player-seasons at exactly 0%
    or 100% K% finite.
    """
    return np.log((k + 0.5) / (n - k + 0.5))


def inv_logit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_stuff_prior(df: pd.DataFrame, cutoff: int, cfg: Config):
    """Fit the prior mean E[logit K% | Stuff+, age], BF-weighted, seasons <= cutoff.

    Used instead of a flat league average because Stuff+ becomes reliable in far
    fewer pitches than K% does [3][4]. Must be calibrated before use; see
    `fit_prior_calibration`.
    """
    from sklearn.linear_model import LinearRegression

    d = df[(df.Season <= cutoff) & (~df.is_pos_player) & (df.TBF >= cfg.min_tbf_train)].copy()
    X = np.column_stack([
        d.stuff_plus.values,
        d.stuff_plus.values ** 2,
        d.Age.values,
    ])
    y = emp_logit(d.k_count.values, d.TBF.values)
    model = LinearRegression().fit(X, y, sample_weight=d.TBF.values)

    # Centre so the prior reproduces league average at league-average stuff.
    def predict(stuff: np.ndarray, age: np.ndarray) -> np.ndarray:
        stuff = np.clip(stuff, 40.0, 160.0)   # guard against position-player tails
        Xp = np.column_stack([stuff, stuff ** 2, age])
        return model.predict(Xp)

    return predict


def build_features(df: pd.DataFrame, target_season: int, cfg: Config,
                   weights: tuple, k_bf: float | None, aging: pd.Series,
                   lg: pd.DataFrame, lg_target: float,
                   prior_mode: str = "stuff", use_aging: bool = True,
                   calibration: tuple | None = None) -> pd.DataFrame:
    """Construct one feature row per player eligible for a `target_season` projection.

    HARD CUTOFF: only Season <= target_season - 1 is read. Asserted below.
    """
    cutoff = target_season - 1
    assert len(weights) == cfg.n_lag_seasons, (
        f"weights has {len(weights)} entries but n_lag_seasons="
        f"{cfg.n_lag_seasons}; unmatched lags would be silently zero-weighted")
    hist = df[df.Season <= cutoff].copy()
    assert hist.Season.max() <= cutoff, "LEAKAGE: history extends past cutoff"

    # Keep the most recent `n_lag_seasons` seasons per player, weighting each by
    # its GAP from the cutoff so a missed season counts as stale information.
    hist["lag"] = cutoff - hist.Season
    hist = hist[hist.lag < cfg.n_lag_seasons].copy()
    hist["w"] = hist.lag.map({i: weights[i] for i in range(len(weights))}).astype(float)

    lg_map = lg.set_index("Season").lg_k_pct.to_dict()
    hist["lg_k"] = hist.Season.map(lg_map)

    stuff_prior = fit_stuff_prior(df, cutoff, cfg)

    # Flat and informed priors need different effective sample sizes; see
    # `estimate_shrinkage_k_informed`. Auto-derive only when the caller passes
    # None, so an explicit k_bf is always honoured (used by the ablation).
    if prior_mode == "stuff" and k_bf is None:
        k_bf = estimate_shrinkage_k_informed(
            df, cutoff, cfg, stuff_prior, calibration)["k_bf_informed"]
    if k_bf is None:
        k_bf = estimate_shrinkage_k(df, cutoff, cfg.min_tbf_train)["k_bf"]

    rows = []
    for pid, g in hist.groupby("PlayerId", sort=False):
        g = g.sort_values("Season")
        last = g.iloc[-1]

        wk = float((g.w * g.k_count).sum())          # recency-weighted strikeouts
        wbf = float((g.w * g.TBF).sum())             # recency-weighted batters faced
        if wbf <= 0:
            continue

        wbf_only = (g.w * g.TBF)
        stuff_wtd = float((wbf_only * g.stuff_plus).sum() / wbf)
        lg_ref = float((wbf_only * g.lg_k).sum() / wbf)   # environment the record was set in

        # Change in stuff between the two most recent observed seasons.
        stuff_delta = float(g.stuff_plus.iloc[-1] - g.stuff_plus.iloc[-2]) if len(g) >= 2 else 0.0

        age_target = int(last.Age + (target_season - last.Season))

        # ---- empirical-Bayes shrinkage toward the prior ---------------------
        # "stuff" -> Stuff+-informed prior;  "flat" -> league average (control)
        prior_logit = float(stuff_prior(np.array([stuff_wtd]), np.array([age_target]))[0])
        if calibration is not None:
            a_cal, b_cal = calibration
            prior_logit = a_cal + b_cal * prior_logit
        if prior_mode == "flat":
            prior_p = float(lg_target)
            prior_logit_flat = float(np.log(prior_p / (1 - prior_p)))
        else:
            prior_p = float(inv_logit(prior_logit))
            prior_logit_flat = prior_logit
        p_eb = (wk + k_bf * prior_p) / (wbf + k_bf)

        # ---- aging: accumulate one-year deltas from last observed age ------
        age_adj = 0.0
        if use_aging:
            for a in range(int(last.Age), age_target):
                age_adj += float(aging.get(a, 0.0))

        # ---- league environment: re-anchor into the target season ----------
        league_shift = lg_target - lg_ref

        p_final = p_eb + age_adj + league_shift
        p_final = float(np.clip(p_final, 0.01, 0.60))

        reliability = wbf / (wbf + k_bf)

        rows.append({
            "PlayerId": pid,
            "Name": last.Name,
            "last_season": int(last.Season),
            "age_target": age_target,
            "w_k": wk,
            "w_bf": wbf,
            "raw_w_rate": wk / wbf,
            "log_w_bf": np.log1p(wbf),
            "tbf_last": int(last.TBF),
            "last_rate": float(last.k_pct),
            "log_tbf_last": np.log1p(last.TBF),
            "stuff_wtd": stuff_wtd,
            "stuff_last": float(last.stuff_plus),
            "stuff_delta": stuff_delta,
            "prior_p": prior_p,
            "reliability": reliability,
            # How much better/worse than his stuff implies he has actually been.
            # Captures command, deception and extension, which Stuff+ misses.
            "stuff_residual": float(emp_logit(np.array([wk]), np.array([wbf]))[0] - prior_logit),
            "prior_logit_used": prior_logit_flat,
            "n_seasons": int(len(g)),
            "is_pos_player": bool(last.is_pos_player),
            "marcel_eb": p_final,          # <- the standalone EB projection
            "eb_logit": float(np.log(p_final / (1 - p_final))),
            "age_adj": age_adj,
            "league_shift": league_shift,
        })

    return pd.DataFrame(rows)


# =============================================================================
# 6. MODEL LADDER
# =============================================================================

FEATURES = [
    "eb_logit", "raw_w_rate", "log_w_bf", "log_tbf_last",
    "stuff_wtd", "stuff_last", "stuff_delta",
    "stuff_residual", "reliability", "age_target", "n_seasons",
]


def _design(feat: pd.DataFrame) -> np.ndarray:
    X = feat[FEATURES].copy()
    X["age_sq"] = X.age_target ** 2
    return X.values.astype(float)


def fit_ridge(feat_tr: pd.DataFrame, y_tr: np.ndarray, w_tr: np.ndarray, seed: int):
    """Ridge on the engineered features, in logit space, TBF-weighted.

    Ridge rather than OLS because the features are collinear by construction.
    Alpha selected by weighted CV within the training data only [7].
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-2, 3, 30))),
    ])
    pipe.fit(_design(feat_tr), y_tr, ridge__sample_weight=w_tr)
    return pipe


def fit_gbm(feat_tr: pd.DataFrame, y_tr: np.ndarray, w_tr: np.ndarray, seed: int):
    """LightGBM on the same features.

    Deliberately shallow: only ~2,000 paired rows are available. Included so the
    comparison against the simpler models is demonstrated rather than assumed.
    """
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.03, num_leaves=7,
        min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=5.0,
        random_state=seed, verbose=-1,
    )
    model.fit(_design(feat_tr), y_tr, sample_weight=w_tr)
    return model


def fit_prior_calibration(df: pd.DataFrame, cutoff: int, cfg: Config,
                          weights: tuple) -> tuple:
    """Calibrate the Stuff+ prior to predict the FOLLOWING season.

    Fits a + b*prior_logit against next season's empirical logit, BF-weighted,
    seasons <= cutoff. The contemporaneous prior is over-dispersed when applied
    forward (slope b ~= 0.75); without this correction the informed prior
    performs worse than a flat one. See the methodology report.
    """
    from sklearn.linear_model import LinearRegression

    seasons = sorted(df.Season.unique())
    inner = [t for t in seasons if seasons[0] < t <= cutoff]
    frames = []
    for t in inner:
        lg_i = league_environment(df, t - 1)
        lgt_i = project_league_k(lg_i, t)
        k_i = estimate_shrinkage_k(df, t - 1, cfg.min_tbf_train)["k_bf"]
        ag_i = fit_aging_curve(df, t - 1, lg_i)
        f = build_features(df, t, cfg, weights, None, ag_i, lg_i, lgt_i,
                           prior_mode="stuff", calibration=None)
        act = (df[df.Season == t][["PlayerId", "k_pct", "TBF", "k_count"]]
               .rename(columns={"k_pct": "y_true", "TBF": "tbf_actual",
                                "k_count": "k_actual"}))
        frames.append(f.merge(act, on="PlayerId", how="inner"))

    if not frames:
        return (0.0, 1.0)   # no paired data yet -> identity calibration
    tr = pd.concat(frames, ignore_index=True)
    tr = tr[(~tr.is_pos_player) & (tr.tbf_actual >= cfg.min_tbf_train)]
    if len(tr) < 100:
        return (0.0, 1.0)

    pl = np.log(np.clip(tr.prior_p.values, 1e-4, 1 - 1e-4) /
                (1 - np.clip(tr.prior_p.values, 1e-4, 1 - 1e-4)))
    y = emp_logit(tr.k_actual.values, tr.tbf_actual.values)
    reg = LinearRegression().fit(pl.reshape(-1, 1), y,
                                 sample_weight=tr.tbf_actual.values.astype(float))
    return (float(reg.intercept_), float(reg.coef_[0]))


# =============================================================================
# 7. PAIR CONSTRUCTION AND BACKTEST
# =============================================================================

@dataclass
class Pairs:
    """Feature rows for one target season, joined to that season's outcome."""
    target: int
    feat: pd.DataFrame            # every projectable player (used for prediction)
    scored: pd.DataFrame          # subset that actually pitched in `target`
    diagnostics: dict = field(default_factory=dict)


def make_pairs(df: pd.DataFrame, target: int, cfg: Config, weights: tuple,
               prior_mode: str = "stuff", use_aging: bool = True) -> Pairs:
    """Everything needed to project `target`, fitted strictly on Season <= target-1.

    League environment, shrinkage constant, aging curve and Stuff+ prior are all
    re-estimated at this cutoff. Nothing is carried in from a wider window.
    """
    cutoff = target - 1
    lg = league_environment(df, cutoff)
    lg_target = project_league_k(lg, target)
    k_est = estimate_shrinkage_k(df, cutoff, cfg.min_tbf_train)
    aging = fit_aging_curve(df, cutoff, lg)

    stuff_prior_fn = fit_stuff_prior(df, cutoff, cfg)
    calib = fit_prior_calibration(df, cutoff, cfg, weights) if prior_mode == "stuff" else None
    k_inf = estimate_shrinkage_k_informed(df, cutoff, cfg, stuff_prior_fn, calib)

    k_arg = None if prior_mode == "stuff" else k_est["k_bf"]
    feat = build_features(df, target, cfg, weights, k_arg, aging,
                          lg, lg_target, prior_mode=prior_mode, use_aging=use_aging,
                          calibration=calib)

    actual = (df[df.Season == target][["PlayerId", "k_pct", "TBF", "k_count"]]
              .rename(columns={"k_pct": "y_true", "TBF": "tbf_actual",
                               "k_count": "k_actual"}))
    scored = feat.merge(actual, on="PlayerId", how="inner")

    return Pairs(target=target, feat=feat, scored=scored,
                 diagnostics={"lg_target": lg_target, **k_est, **k_inf,
                              "prior_calibration": calib,
                              "n_projectable": len(feat), "n_scored": len(scored)})


def metrics(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> dict:
    """RMSE / MAE / R^2, reported both unweighted and batters-faced-weighted."""
    err = y_pred - y_true
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "rmse_bfw": float(np.sqrt(np.average(err ** 2, weights=w))),
        "mae_bfw": float(np.average(np.abs(err), weights=w)),
        "bias": float(err.mean()),
        "n": int(len(y_true)),
    }


def evaluate_models(train_pairs: list, test: Pairs, cfg: Config,
                    min_tbf_eval: int) -> pd.DataFrame:
    """Fit every rung of the ladder on `train_pairs`, score on `test`."""
    # ---- assemble training rows (real pitchers, meaningful usage) ----------
    tr = pd.concat([p.scored for p in train_pairs], ignore_index=True)
    tr = tr[(~tr.is_pos_player) & (tr.tbf_actual >= cfg.min_tbf_train)].copy()

    te = test.scored.copy()
    te = te[(~te.is_pos_player) & (te.tbf_actual >= min_tbf_eval)].copy()
    if len(te) == 0 or len(tr) < 50:
        return pd.DataFrame()

    y_tr = emp_logit(tr.k_actual.values, tr.tbf_actual.values)
    w_tr = tr.tbf_actual.values.astype(float)
    y_te = te.y_true.values
    w_te = te.tbf_actual.values.astype(float)

    preds = {}

    # --- M0: league average (the floor any model must clear) ---------------
    preds["M0_league_average"] = np.full(len(te), test.diagnostics["lg_target"])

    # --- M1: last season's K%, unregressed (naive persistence) -------------
    #     The bar a projection actually has to clear to be worth building.
    preds["M1_last_season_kpct"] = te.last_rate.values

    # --- M2: Marcel-style EB, flat league prior (control) ------------------
    #     supplied by the caller via the "flat" variant; see run_backtest.
    if "marcel_flat" in te.columns:
        preds["M2_marcel_flat_prior"] = te.marcel_flat.values

    # --- M3: empirical Bayes, Stuff+-informed prior (this project's core) ---
    preds["M3_eb_stuff_prior"] = te.marcel_eb.values

    # --- M4: ridge on engineered features ----------------------------------
    ridge = fit_ridge(tr, y_tr, w_tr, cfg.seed)
    preds["M4_ridge"] = inv_logit(ridge.predict(_design(te)))

    # --- M5: LightGBM -------------------------------------------------------
    gbm = fit_gbm(tr, y_tr, w_tr, cfg.seed)
    preds["M5_lightgbm"] = inv_logit(gbm.predict(_design(te)))

    # --- M6: equal-weight blend of the EB projection and ridge -------------
    #     ATC and FanGraphs' Depth Charts both work this way: blending
    #     decorrelated systems beats picking one [2].
    preds["M6_blend_eb_ridge"] = 0.5 * preds["M3_eb_stuff_prior"] + 0.5 * preds["M4_ridge"]

    out = []
    for name, p in preds.items():
        p = np.clip(p, 0.01, 0.60)
        out.append({"model": name, "target_season": test.target,
                    "min_tbf_eval": min_tbf_eval, **metrics(y_te, p, w_te)})
    return pd.DataFrame(out)


def build_cache(df: pd.DataFrame, cfg: Config, weights: tuple,
                targets: list) -> dict:
    """Feature/outcome pairs for each target season, both prior variants.

    The flat-prior control rides along as a column on the stuff-prior frame so
    one object carries both. Every pair is fitted strictly on Season <= t-1.
    """
    cache = {}
    for t in targets:
        c_stuff = make_pairs(df, t, cfg, weights, prior_mode="stuff")
        c_flat = make_pairs(df, t, cfg, weights, prior_mode="flat")
        flat = c_flat.scored[["PlayerId", "marcel_eb"]].rename(
            columns={"marcel_eb": "marcel_flat"})
        c_stuff.scored = c_stuff.scored.merge(flat, on="PlayerId", how="left")
        cache[t] = c_stuff
    return cache


def run_backtest(df: pd.DataFrame, cfg: Config, eval_seasons: list) -> tuple:
    """Rolling-origin backtest: for each season Y, fit only on transitions that
    completed before Y, project Y, then score against Y.

    The recency weights are re-selected INSIDE each fold, from targets <= Y-1
    only. Selecting them once across the whole training window and then
    reporting RMSE on those same seasons makes this table in-sample with respect
    to the weight search, which flatters every model that consumes the EB
    projection. The shipped weights in `fit_and_project` are still fitted on the
    full training window -- that is legitimate, because none of it reaches the
    target season -- so the fold weights and the shipped weights can differ.
    What this table measures is the procedure, not one fixed weight vector.

    Returns (backtest table, {season: weights used for that fold}).
    """
    if not eval_seasons:
        return pd.DataFrame(), {}

    seasons = sorted(df.Season.unique())
    first_target = seasons[0] + 1

    all_rows, fold_weights = [], {}
    for Y in eval_seasons:
        fold_targets = list(range(first_target, Y))
        if not fold_targets:
            continue
        # A target season with only one prior season gives every weight vector
        # the same projection, so it cannot discriminate: require two.
        w_fold, _ = select_weights(df, cfg, list(range(seasons[0] + 2, Y)))
        fold_weights[Y] = tuple(w_fold)
        cache = build_cache(df, cfg, w_fold, fold_targets + [Y])
        train = [cache[t] for t in fold_targets]
        for thr in cfg.eval_thresholds:
            res = evaluate_models(train, cache[Y], cfg, thr)
            if len(res):
                res["fold_weights"] = str(tuple(round(x, 2) for x in w_fold))
                all_rows.append(res)

    bt = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    return bt, fold_weights


def select_weights(df: pd.DataFrame, cfg: Config, tune_seasons: list) -> tuple:
    """Grid-search the recency weights instead of assuming Marcel's 5/4/3 [1][2].

    Criterion: BF-weighted RMSE of the standalone EB projection at TBF >= 100,
    averaged over the tuning seasons.
    """
    scores = []
    for w in cfg.weight_grid:
        errs = []
        for Y in tune_seasons:
            pr = make_pairs(df, Y, cfg, w, prior_mode="stuff")
            s = pr.scored
            s = s[(~s.is_pos_player) & (s.tbf_actual >= 100)]
            if len(s) < 30:
                continue
            errs.append(np.sqrt(np.average((s.marcel_eb - s.y_true) ** 2,
                                           weights=s.tbf_actual)))
        if errs:
            scores.append({"weights": w, "rmse_bfw": float(np.mean(errs))})
    tab = pd.DataFrame(scores)
    if tab.empty:
        # No tuning season carries enough paired data -- the earliest backtest
        # folds, or a target only one season past the start of the file. Fall
        # back to Marcel's published 5/4/3 rather than inventing a fit.
        return cfg.weight_grid[0], pd.DataFrame(columns=["weights", "rmse_bfw"])
    tab = tab.sort_values("rmse_bfw").reset_index(drop=True)
    return tab.iloc[0]["weights"], tab


# =============================================================================
# 8. UNCERTAINTY
# =============================================================================

def prediction_intervals(bt_residuals: np.ndarray, reliability: np.ndarray,
                         point: np.ndarray, tbf_proj: np.ndarray,
                         scale: float = 1.0) -> tuple:
    """80% prediction intervals.

    Combines talent uncertainty (shrinks as reliability rises, calibrated from
    backtest residuals) with binomial uncertainty in next season's observed rate.
    The second term dominates for low-usage pitchers.
    """
    sigma_res = float(np.std(bt_residuals)) * scale
    talent_sd = sigma_res * np.sqrt(np.clip(1.0 - reliability, 0.05, 1.0) / 0.5)
    binom_sd = np.sqrt(np.clip(point * (1 - point), 1e-6, None) / np.clip(tbf_proj, 1, None))
    total_sd = np.sqrt(talent_sd ** 2 + binom_sd ** 2)
    z = 1.2816  # 80% central interval
    return (np.clip(point - z * total_sd, 0.0, 1.0),
            np.clip(point + z * total_sd, 0.0, 1.0),
            total_sd)


def calibrate_interval_scale(cache: dict, resid: np.ndarray) -> float:
    """Scan a width scale on the backtest seasons and keep the one whose empirical
    coverage is closest to the nominal 80%.

    The binomial half of the interval scales as 1/sqrt(TBF), so the scan has to
    use `project_tbf` -- the same crude playing-time estimate available at
    projection time -- and not the target season's realised TBF, which is not
    knowable when the interval is issued. Calibrating on realised TBF tunes the
    width in a regime that never occurs in production.
    """
    frames = []
    for t in cache:
        s = cache[t].scored
        frames.append(s[(~s.is_pos_player) & (s.tbf_actual >= 25)])
    if not frames:
        return 1.0
    v = pd.concat(frames, ignore_index=True)
    best, best_gap = 1.0, 9e9
    for scale in np.arange(0.5, 2.01, 0.05):
        lo, hi, _ = prediction_intervals(resid, v.reliability.values,
                                         v.marcel_eb.values,
                                         project_tbf(v), scale=scale)
        cov = float(((v.y_true >= lo) & (v.y_true <= hi)).mean())
        gap = abs(cov - 0.80)
        if gap < best_gap:
            best, best_gap = float(scale), gap
    return best


def project_tbf(feat: pd.DataFrame) -> np.ndarray:
    """Crude expected batters faced next season, Marcel-style [1].

    Used only to size prediction intervals, never as a playing-time forecast.
    """
    return 0.5 * feat.tbf_last.values + 0.1 * (feat.w_bf.values - feat.tbf_last.values) + 60.0


# =============================================================================
# 9. FIGURES
# =============================================================================

def make_figures(df: pd.DataFrame, cfg: Config, bt: pd.DataFrame,
                 cache: dict, final: pd.DataFrame, aging: pd.Series,
                 lg: pd.DataFrame, k_est: dict, tag: str,
                 holdout_df: pd.DataFrame | None = None) -> list:
    """Diagnostic and explanatory figures for the methodology report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
    })
    NAVY, RED, GREY = "#284B8C", "#BA0C2F", "#8A8A8A"   # Phillies-ish palette
    paths = []
    od = cfg.out_dir

    # --- Fig 1: reliability of K% vs Stuff+ as a function of usage ----------
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    p = df[~df.is_pos_player][["Season", "PlayerId", "k_pct", "TBF", "stuff_plus"]]
    nx = p.copy(); nx["Season"] -= 1
    m = p.merge(nx, on=["Season", "PlayerId"], suffixes=("", "_n"))
    m = m[m.TBF_n >= 50]
    edges = [0, 25, 50, 100, 200, 400, 900]
    lbl, rk, rs, ns = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = m[(m.TBF > lo) & (m.TBF <= hi)]
        if len(g) < 15:
            continue
        lbl.append(f"{lo}-{hi}")
        rk.append(g.k_pct.corr(g.k_pct_n))
        rs.append(g.stuff_plus.corr(g.k_pct_n))
        ns.append(len(g))
    x = np.arange(len(lbl))
    ax.bar(x - 0.2, rk, 0.4, label="prior-year K%", color=NAVY)
    ax.bar(x + 0.2, rs, 0.4, label="prior-year Stuff+", color=RED)
    ax.set_xticks(x); ax.set_xticklabels(lbl)
    ax.set_xlabel("Batters faced in year Y"); ax.set_ylabel("correlation with K% in year Y+1")
    ax.set_title("A pitcher's own K% is only trustworthy with usage;\n"
                 "below ~25 BF, Stuff+ carries more signal about next year",
                 fontsize=10)
    top = max(max(rk), max(rs))
    for i, n in enumerate(ns):
        ax.text(i, top * 1.06, f"n={n}", ha="center", fontsize=7, color=GREY)
    ax.set_ylim(0, top * 1.16)
    ax.axhline(0, color="black", lw=0.8); ax.legend(frameon=False)
    fig.tight_layout(); f = od / f"fig1_reliability_{tag}.png"; fig.savefig(f); plt.close(fig)
    paths.append(f)

    # --- Fig 2: league-detrended aging curve --------------------------------
    # Plotted as the PER-YEAR step the model actually applies. A cumulative
    # version would compound delta-method survivorship bias across two decades
    # and imply an implausible career-long decline; in practice the projection
    # applies a single one-year step for almost every pitcher.
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ages = [a for a in aging.index if 22 <= a <= 38]
    vals = [aging[a] * 100 for a in ages]
    ax.bar(ages, vals, color=[NAVY if v >= -0.4 else RED for v in vals], width=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Age in projected season")
    ax.set_ylabel("one-year ΔK% applied (pct. points)")
    ax.set_title("Aging adjustment, league-detrended (delta method, BF-weighted)\n"
                 "flat through 27, then a clear break", fontsize=10)
    fig.tight_layout(); f = od / f"fig2_aging_{tag}.png"; fig.savefig(f); plt.close(fig)
    paths.append(f)

    # --- Fig 3: backtest model comparison -----------------------------------
    sub = bt[bt.min_tbf_eval == 100] if len(bt) else bt
    if len(sub):
        piv = sub.pivot_table(index="model", columns="target_season", values="rmse_bfw")
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        piv.plot(kind="bar", ax=ax, width=0.78,
                 color=[NAVY, RED, "#6D8FC4", "#D4A017"][:piv.shape[1]])
        ax.set_ylabel("BF-weighted RMSE (K%)"); ax.set_xlabel("")
        ax.set_title("Rolling-origin backtest, pitchers with ≥100 BF in target season", fontsize=10)
        ax.legend(title="target season", frameon=False, fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
        fig.tight_layout(); f = od / f"fig3_backtest_{tag}.png"; fig.savefig(f); plt.close(fig)
        paths.append(f)

    # --- Fig 4: calibration + accuracy on the HELD-OUT target season --------
    if holdout_df is not None and len(holdout_df) > 40:
        last_scored = cfg.target_season
        s = holdout_df[holdout_df.tbf_actual >= 100].copy()
        s["marcel_eb"] = s.k_pct_projected      # plot the shipped projection
    elif cache:
        last_scored = max(cache.keys())
        s = cache[last_scored].scored
        s = s[(~s.is_pos_player) & (s.tbf_actual >= 100)]
    else:
        # Neither a holdout frame nor a backtest cache: nothing to plot. Bind
        # last_scored anyway so the guarded block below cannot reference it
        # unbound if that guard is ever loosened.
        last_scored = cfg.target_season
        s = pd.DataFrame()
    if len(s) > 40:
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
        a = axes[0]
        a.scatter(s.marcel_eb, s.y_true, s=np.sqrt(s.tbf_actual) * 1.6,
                  alpha=0.4, color=NAVY, edgecolor="none")
        lim = [0.05, 0.42]
        a.plot(lim, lim, color=RED, lw=1.2, ls="--")
        a.set_xlim(lim); a.set_ylim(lim)
        a.set_xlabel(f"projected K% ({last_scored})"); a.set_ylabel(f"actual K% ({last_scored})")
        r = float(np.corrcoef(s.marcel_eb, s.y_true)[0, 1])
        rm = float(np.sqrt(((s.marcel_eb - s.y_true) ** 2).mean()))
        a.set_title(f"Held-out {last_scored}: projected vs actual (≥100 BF)\n"
                    f"n={len(s)}   r={r:.3f}   RMSE={rm*100:.2f} pp", fontsize=10)

        b = axes[1]
        s2 = s.copy(); s2["dec"] = pd.qcut(s2.marcel_eb, 10, labels=False, duplicates="drop")
        cal = s2.groupby("dec").apply(
            lambda g: pd.Series({"pred": np.average(g.marcel_eb, weights=g.tbf_actual),
                                 "act": np.average(g.y_true, weights=g.tbf_actual)}),
            include_groups=False)
        b.plot(cal.pred * 100, cal.act * 100, "o-", color=NAVY, ms=5)
        lo, hi = cal.pred.min() * 100, cal.pred.max() * 100
        b.plot([lo, hi], [lo, hi], color=RED, ls="--", lw=1.2)
        b.set_xlabel("mean projected K% (decile)"); b.set_ylabel("mean actual K% (decile)")
        b.set_title("Calibration by projection decile", fontsize=10)
        fig.tight_layout(); f = od / f"fig4_calibration_{tag}.png"; fig.savefig(f); plt.close(fig)
        paths.append(f)

    # --- Fig 5: league K% trend and the projected target ---------------------
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(lg.Season, lg.lg_k_pct * 100, "o-", color=NAVY, lw=2, label="observed league K%")
    tgt = cfg.target_season
    ax.plot([tgt], [final.attrs.get("lg_target", np.nan) * 100], "*",
            ms=15, color=RED, label=f"projected {tgt}")
    ax.set_xlabel("Season"); ax.set_ylabel("league K% (BF-weighted)")
    ax.set_title("League strikeout environment", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); f = od / f"fig5_league_{tag}.png"; fig.savefig(f); plt.close(fig)
    paths.append(f)

    return paths


# =============================================================================
# 10. MAIN
# =============================================================================

def run_ablation(df: pd.DataFrame, cfg: Config, weights: tuple) -> pd.DataFrame:
    """Disable each component in turn and re-score on the held-out target season."""
    cutoff = cfg.target_season - 1
    if cfg.target_season not in df.Season.values:
        return pd.DataFrame()

    lg = league_environment(df, cutoff)
    lg_t = project_league_k(lg, cfg.target_season)
    k_flat = estimate_shrinkage_k(df, cutoff, cfg.min_tbf_train)["k_bf"]
    aging = fit_aging_curve(df, cutoff, lg)
    calib = fit_prior_calibration(df, cutoff, cfg, weights)
    sp = fit_stuff_prior(df, cutoff, cfg)
    k_inf = estimate_shrinkage_k_informed(df, cutoff, cfg, sp, calib)["k_bf_informed"]

    act = (df[df.Season == cfg.target_season][["PlayerId", "k_pct", "TBF"]]
           .rename(columns={"k_pct": "y_true", "TBF": "tbf_actual"}))

    def variant(label, w=None, use_aging=True, prior_mode="stuff",
                use_calib=True, no_league=False, k_override=None):
        w = w or weights
        c = calib if (use_calib and prior_mode == "stuff") else None
        kb = k_override if k_override is not None else (None if prior_mode == "stuff" else k_flat)
        f = build_features(df, cfg.target_season, cfg, w, kb, aging, lg, lg_t,
                           prior_mode=prior_mode, use_aging=use_aging, calibration=c)
        if no_league:
            f["marcel_eb"] = f.marcel_eb - f.league_shift
        h = f.merge(act, on="PlayerId", how="inner")
        h = h[(~h.is_pos_player) & (h.tbf_actual >= 100)]
        m = metrics(h.y_true.values, np.clip(h.marcel_eb.values, .01, .60),
                    h.tbf_actual.values)
        return {"variant": label, **m}

    rows = [
        variant("FULL empirical-Bayes model"),
        variant("  no aging adjustment", use_aging=False),
        variant("  no league re-anchoring", no_league=True),
        variant("  no recency weighting (1/1/1)", w=(1.0, 1.0, 1.0)),
        variant("  flat prior (Stuff+ unused)", prior_mode="flat"),
        variant("  uncalibrated Stuff+ prior", use_calib=False),
        variant("  almost no shrinkage (k=5)", k_override=5.0),
        variant("  over-shrunk (k=400)", k_override=400.0),
    ]
    t = pd.DataFrame(rows)
    t["delta_bps"] = ((t.rmse - t.rmse.iloc[0]) * 10000).round(1)
    return t


def fit_and_project(df: pd.DataFrame, cfg: Config, verbose: bool = True) -> dict:
    """The complete pipeline: fit on Season <= target-1, project the target season.

    Returns every intermediate a caller might want to print, score or test.
    Critically, this function NEVER reads df[df.Season >= cfg.target_season].
    """
    cutoff = cfg.target_season - 1
    seasons = sorted(df.Season.unique())

    def say(*a):
        if verbose:
            print(*a)

    # ---------- leakage guard ------------------------------------------------
    n_future = int((df.Season > cutoff).sum())
    say(f"\n[2] Leakage guard: {n_future} rows at Season > {cutoff} exist in the file "
        f"and are excluded from every fitted object below.")

    lg = league_environment(df, cutoff)
    lg_target = project_league_k(lg, cfg.target_season)
    say("\n[3] League environment (BF-weighted K%), training window only:")
    for _, r in lg.iterrows():
        say(f"      {int(r.Season)}: {r.lg_k_pct*100:5.2f}%")
    say(f"      {cfg.target_season}: {lg_target*100:5.2f}%  <- projected (damped trend)")

    k_est = estimate_shrinkage_k(df, cutoff, cfg.min_tbf_train)
    say(f"\n[4] Variance decomposition (Season <= {cutoff}, TBF >= {cfg.min_tbf_train}):")
    say(f"      Var(observed K%) = {k_est['var_observed']:.5f}")
    say(f"      binomial noise   = {k_est['binomial_noise']:.5f}")
    say(f"      Var(true talent) = {k_est['var_talent']:.5f}")
    say(f"      => k = {k_est['k_bf']:.1f} batters faced of league-average prior")
    say(f"         (cf. ~70 BF stabilisation point for K% [4]; cf. Marcel's 1200 PA [1])")

    aging = fit_aging_curve(df, cutoff, lg)
    aging_raw = fit_aging_curve(df, cutoff, lg, detrend=False)

    tune_seasons = [s_ for s_ in seasons if seasons[0] + 2 <= s_ <= cutoff]
    weights, wtab = select_weights(df, cfg, tune_seasons)
    if len(wtab):
        say(f"\n[5] Recency weights fitted on {[int(x) for x in tune_seasons]} "
            f"(not assumed):")
    else:
        say(f"\n[5] Recency weights: no tuning season has two prior seasons, so "
            f"the search\n    cannot discriminate. Falling back to Marcel's "
            f"published 5/4/3 = {tuple(round(x, 2) for x in weights)}.")
    for _, r in wtab.iterrows():
        mark = "  <-- selected" if tuple(r.weights) == tuple(weights) else ""
        say(f"      {str(tuple(round(x,2) for x in r.weights)):22s} "
            f"BF-wtd RMSE {r.rmse_bfw:.5f}{mark}")

    eval_seasons = [s_ for s_ in seasons if seasons[0] + 2 <= s_ <= cutoff]
    say(f"\n[6] Rolling-origin backtest on target seasons "
        f"{[int(x) for x in eval_seasons]} ...")
    say("    Weights are re-selected inside each fold, so this table is "
        "out-of-sample\n    with respect to the weight search as well as the "
        "outcome:")
    bt, fold_weights = run_backtest(df, cfg, eval_seasons)
    for Y in sorted(fold_weights):
        say(f"      fold {Y}: weights {tuple(round(x, 2) for x in fold_weights[Y])} "
            f"selected on targets {list(range(seasons[0] + 2, Y)) or '-- (none; Marcel default)'}")

    if len(bt) and verbose:
        show = (bt[bt.min_tbf_eval == 100]
                .groupby("model")[["rmse", "rmse_bfw", "mae", "r2"]].mean()
                .sort_values("rmse_bfw"))
        say("\n    Mean across backtest seasons, pitchers with >=100 BF:")
        say("    " + "-" * 66)
        say(f"    {'model':26s} {'RMSE':>8s} {'RMSE_bfw':>9s} {'MAE':>8s} {'R2':>7s}")
        say("    " + "-" * 66)
        for name, r in show.iterrows():
            say(f"    {name:26s} {r.rmse:8.5f} {r.rmse_bfw:9.5f} {r.mae:8.5f} {r.r2:7.3f}")

    # ---------- final fit and projection ------------------------------------
    say(f"\n[7] Fitting final model and projecting {cfg.target_season} ...")
    final_pairs = make_pairs(df, cfg.target_season, cfg, weights, prior_mode="stuff")
    feat = final_pairs.feat.copy()

    # Training material for the shipped ridge and for the interval calibration,
    # built with the SHIPPED weights so its features match what `feat` carries.
    # The backtest above uses its own per-fold caches; reusing those here would
    # mix weight conventions between the ridge's training rows and its inputs.
    # Still strictly Season <= cutoff.
    cache = build_cache(df, cfg, weights, list(range(seasons[0] + 1, cutoff + 1)))

    feat["pred_eb"] = feat.marcel_eb
    if cache:
        train_all = pd.concat([cache[t].scored for t in cache], ignore_index=True)
        train_all = train_all[(~train_all.is_pos_player) &
                              (train_all.tbf_actual >= cfg.min_tbf_train)]
        y_tr = emp_logit(train_all.k_actual.values, train_all.tbf_actual.values)
        w_tr = train_all.tbf_actual.values.astype(float)

        ridge = fit_ridge(train_all, y_tr, w_tr, cfg.seed)
        feat["pred_ridge"] = inv_logit(ridge.predict(_design(feat)))
    else:
        # No completed season-to-season transition exists yet, so there is
        # nothing to fit a ridge on. Ship the EB projection alone.
        say("    no completed transitions available; shipping the EB projection alone.")
        feat["pred_ridge"] = feat.pred_eb

    # SHIPPED MODEL: 50/50 blend of the empirical-Bayes projection and ridge.
    # Chosen a priori on the reasoning FanGraphs Depth Charts and ATC use [2]:
    # blending decorrelated systems beats crowning a backtest winner when the
    # fold-to-fold separation is inside noise. Not chosen because it won on 2025.
    feat["k_pct_projected"] = np.clip(0.5 * feat.pred_eb + 0.5 * feat.pred_ridge, 0.01, 0.60)

    # ---------- uncertainty --------------------------------------------------
    resid = []
    for t in cache:
        sc = cache[t].scored
        sc = sc[(~sc.is_pos_player) & (sc.tbf_actual >= 100)]
        resid.append((sc.marcel_eb - sc.y_true).values)
    resid = np.concatenate(resid) if resid else np.array([0.03])

    feat["tbf_projected"] = project_tbf(feat)
    pi_scale = calibrate_interval_scale(cache, resid)
    say(f"    interval width scale calibrated on backtest seasons: {pi_scale:.2f}")
    lo, hi, sd = prediction_intervals(resid, feat.reliability.values,
                                      feat.k_pct_projected.values,
                                      feat.tbf_projected.values, scale=pi_scale)
    feat["pi80_low"], feat["pi80_high"], feat["pred_sd"] = lo, hi, sd
    feat.attrs["lg_target"] = lg_target

    return {"feat": feat, "bt": bt, "cache": cache, "lg": lg, "lg_target": lg_target,
            "k_est": k_est, "aging": aging, "aging_raw": aging_raw,
            "k_inf": final_pairs.diagnostics.get("k_bf_informed"),
            "k_by_threshold": {t: estimate_shrinkage_k(df, cutoff, t)["k_bf"]
                               for t in (25, 50, 100, 200)},
            "weights": weights, "wtab": wtab,
            "pi_scale": pi_scale, "resid": resid, "seasons": seasons, "cutoff": cutoff,
            "fold_weights": fold_weights}


EXPORT_COLS = ["PlayerId", "Name", "age_target", "last_season", "k_pct_projected",
               "pi80_low", "pi80_high", "pred_sd", "reliability", "tbf_projected",
               "w_bf", "raw_w_rate", "stuff_wtd", "stuff_delta", "stuff_residual",
               "age_adj", "league_shift", "pred_eb", "pred_ridge",
               "n_seasons", "is_pos_player"]


def build_export(feat: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return (feat[EXPORT_COLS]
            .rename(columns={"age_target": f"age_{cfg.target_season}"})
            .sort_values(["k_pct_projected", "PlayerId"], ascending=[False, True])
            .round(6)
            .reset_index(drop=True))


def self_test(cfg: Config) -> bool:
    """Prove the leakage guard: deleting the target season must change nothing.

    Runs the pipeline twice, once on the file as delivered and once on a copy
    with every target-season row removed, and asserts the projections match.
    """
    print("=" * 78)
    print("  SELF-TEST: does deleting the 2025 rows change the 2025 projections?")
    print("=" * 78)

    df_full = load_data(cfg)
    print(f"\n  run A: file as delivered      -> {len(df_full):,} rows "
          f"({int((df_full.Season >= cfg.target_season).sum())} of them from "
          f"{cfg.target_season})")
    res_a = fit_and_project(df_full, cfg, verbose=False)
    exp_a = build_export(res_a["feat"], cfg)

    df_cut = df_full[df_full.Season < cfg.target_season].copy()
    print(f"  run B: {cfg.target_season} rows deleted   -> {len(df_cut):,} rows")
    res_b = fit_and_project(df_cut, cfg, verbose=False)
    exp_b = build_export(res_b["feat"], cfg)

    same_shape = exp_a.shape == exp_b.shape
    print(f"\n  identical shape ............... {same_shape}  {exp_a.shape} vs {exp_b.shape}")
    if not same_shape:
        print("\n  FAIL: projection tables differ in shape.")
        return False

    num = exp_a.select_dtypes(include=[np.number]).columns
    max_diff = float(np.abs(exp_a[num].values - exp_b[num].values).max())
    ids_match = bool((exp_a.PlayerId.values == exp_b.PlayerId.values).all())
    print(f"  identical PlayerId ordering ... {ids_match}")
    print(f"  max abs numeric difference .... {max_diff:.2e}")

    ok = ids_match and max_diff < 1e-12
    print(f"\n  RESULT: {'PASS -- no 2025 information reaches any fitted object.' if ok else 'FAIL -- leakage detected.'}")
    print("=" * 78 + "\n")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None,
                    help="player-season panel to read "
                         "(default: <script dir>/data/k_2026.csv)")
    ap.add_argument("--target-season", type=int, default=2025)
    ap.add_argument("--out-dir", default=None,
                    help="directory to write outputs into "
                         "(default: <script dir>/output)")
    ap.add_argument("--self-test", action="store_true",
                    help="verify that deleting the target season changes nothing")
    args = ap.parse_args()

    # An explicit --data/--out-dir is resolved against the caller's cwd, as a
    # user would expect; omitting it falls back to the script-relative default.
    kw = {"target_season": args.target_season}
    if args.data:
        kw["data_path"] = Path(args.data).expanduser().resolve()
    if args.out_dir:
        kw["out_dir"] = Path(args.out_dir).expanduser().resolve()
    cfg = Config(**kw)

    if not cfg.data_path.exists():
        raise SystemExit(f"input panel not found: {cfg.data_path}\n"
                         f"pass --data /path/to/k_2026.csv")
    try:
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise SystemExit(f"cannot write to output directory {cfg.out_dir}: {e}\n"
                         f"pass --out-dir /some/writable/path")
    tag = str(cfg.target_season)

    if args.self_test:
        raise SystemExit(0 if self_test(cfg) else 1)

    print("=" * 78)
    print(f"  PITCHER K% PROJECTION  |  target season {cfg.target_season}")
    print("=" * 78)

    df = load_data(cfg)
    seasons = sorted(df.Season.unique())
    print(f"\n[1] Data: {len(df):,} player-seasons, {df.PlayerId.nunique():,} pitchers, "
          f"{seasons[0]}-{seasons[-1]}")
    print(f"    Training window: {seasons[0]}-{cfg.target_season - 1}")
    print(f"    Position-player rows flagged: {df.is_pos_player.sum()} "
          f"({df.is_pos_player.mean()*100:.1f}% of rows, "
          f"{df[df.is_pos_player].TBF.sum()/df.TBF.sum()*100:.2f}% of BF)")

    res = fit_and_project(df, cfg, verbose=True)
    feat, bt, cache = res["feat"], res["bt"], res["cache"]
    if len(bt):
        bt.to_csv(cfg.out_dir / f"backtest_{tag}.csv", index=False)

    export = build_export(feat, cfg)
    pred_path = cfg.out_dir / f"k_pct_projections_{tag}.csv"
    export.to_csv(pred_path, index=False)
    print(f"    -> {len(export):,} projections written to {pred_path}")
    print(f"       ({int(export.is_pos_player.sum())} flagged as position players: "
          f"excluded from training, projected for completeness)")

    # ---------- scoring against the held-out target season -------------------
    holdout = None
    h = None                     # stays None when the target season is absent
    if cfg.target_season in df.Season.values:
        act = (df[df.Season == cfg.target_season][["PlayerId", "k_pct", "TBF"]]
               .rename(columns={"k_pct": "y_true", "TBF": "tbf_actual"}))
        flat_pairs = make_pairs(df, cfg.target_season, cfg, res["weights"], prior_mode="flat")
        h = (feat.merge(act, on="PlayerId", how="inner")
                 .merge(flat_pairs.feat[["PlayerId", "marcel_eb"]]
                        .rename(columns={"marcel_eb": "pred_flat"}),
                        on="PlayerId", how="left"))
        h = h[~h.is_pos_player]

        print(f"\n[8] HELD-OUT SCORING -- {cfg.target_season} was never used to fit anything.")
        for thr in cfg.eval_thresholds:
            hh = h[h.tbf_actual >= thr]
            if len(hh) < 20:
                continue
            print(f"\n    pitchers with >= {thr} BF in {cfg.target_season}  (n = {len(hh)})")
            print("    " + "-" * 62)
            print(f"    {'model':24s} {'RMSE':>9s} {'RMSE_bfw':>9s} {'MAE':>8s} {'R2':>7s}")
            print("    " + "-" * 62)
            cand = {
                "M0_league_average": np.full(len(hh), res["lg_target"]),
                "M1_last_season_kpct": hh.last_rate.values,
                "M2_marcel_flat_prior": hh.pred_flat.values,
                "M3_eb_stuff_prior": hh.pred_eb.values,
                "M4_ridge": hh.pred_ridge.values,
                "M6_blend (SHIPPED)": hh.k_pct_projected.values,
            }
            store = {}
            for nm, pv in cand.items():
                mm = metrics(hh.y_true.values, np.clip(pv, .01, .60), hh.tbf_actual.values)
                store[nm] = mm
                print(f"    {nm:24s} {mm['rmse']:9.5f} {mm['rmse_bfw']:9.5f} "
                      f"{mm['mae']:8.5f} {mm['r2']:7.3f}")
            if thr == 100:
                cov = float(((hh.y_true >= hh.pi80_low) & (hh.y_true <= hh.pi80_high)).mean())
                print(f"    80% prediction-interval coverage: {cov*100:.1f}% (nominal 80%)")
                holdout = {"n": int(len(hh)), "pi80_coverage": cov, **store}

    abl = run_ablation(df, cfg, res["weights"])
    if len(abl):
        abl.to_csv(cfg.out_dir / f"ablation_{tag}.csv", index=False)
        print(f"\n[8b] Component ablation (held-out {cfg.target_season}, >=100 BF):")
        print("     " + "-" * 64)
        print(f"     {'variant':32s} {'RMSE':>8s} {'R2':>7s} {'Δ bps':>8s}")
        print("     " + "-" * 64)
        for _, r in abl.iterrows():
            print(f"     {r.variant:32s} {r.rmse:8.5f} {r.r2:7.3f} {r.delta_bps:+8.1f}")

    figs = make_figures(df, cfg, bt, cache, feat, res["aging"],
                        res["lg"], res["k_est"], tag,
                        holdout_df=h)
    print(f"\n[9] Figures: {', '.join(f.name for f in figs)}")

    manifest = {
        "target_season": cfg.target_season,
        "training_window": [int(seasons[0]), int(cfg.target_season - 1)],
        "seed": cfg.seed,
        "recency_weights": list(res["weights"]),
        # Weights the backtest actually used in each fold, re-selected from that
        # fold's history alone. They can differ from the shipped weights above,
        # which are fitted on the whole training window.
        "backtest_fold_weights": {int(y): list(w) for y, w in res["fold_weights"].items()},
        "shrinkage_k_bf": res["k_est"]["k_bf"],
        # Sensitivity of k to the usage floor it is measured over, and the
        # larger constant the Stuff+-informed prior earns. Both are quoted in
        # the methodology, so they are recorded rather than re-derived by hand.
        "shrinkage_k_bf_by_threshold": {int(t): float(v)
                                        for t, v in res["k_by_threshold"].items()},
        "shrinkage_k_bf_informed": float(res["k_inf"]) if res["k_inf"] else None,
        "variance_decomposition": res["k_est"],
        "league_k_projected": res["lg_target"],
        "league_k_observed": {int(r.Season): float(r.lg_k_pct) for _, r in res["lg"].iterrows()},
        "aging_curve_per_age": {int(a): float(v) for a, v in res["aging"].items() if 21 <= a <= 40},
        "aging_undetrended_per_age": {int(a): float(v) for a, v in res["aging_raw"].items()
                                      if 21 <= a <= 40},
        "interval_scale": res["pi_scale"],
        "n_projections": int(len(export)),
        "n_extreme_rate_seasons": int(((df.k_pct == 0) | (df.k_pct == 1)).sum()),
        "holdout_evaluation": holdout,
    }
    with open(cfg.out_dir / f"run_manifest_{tag}.json", "w") as fh:
        json.dump(manifest, fh, indent=2, default=float)
    print(f"[10] Manifest: {cfg.out_dir / f'run_manifest_{tag}.json'}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
