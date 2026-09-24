import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# ============================================================
# 1. Define calibration and out-of-sample periods
# ============================================================

# Calibration period:
# Historical period used to estimate the stochastic parameter models.
calibration_df = val_df.copy()

# Out-of-sample period:
# 2025 is kept completely separate from parameter calibration.
out_of_sample_df = df2025.ffill().bfill().copy()


# ============================================================
# 2. Align calibration residuals with calibration observations
# ============================================================

min_len = min(len(res_val), len(calibration_df))

calibration_df = calibration_df.iloc[:min_len].copy()
calibration_df["residual"] = np.asarray(res_val[:min_len])

# Remove observations for which the residual is unavailable
calibration_df = calibration_df.dropna(subset=["residual"]).copy()


# ============================================================
# 3. Residual-based stochastic targets
# ============================================================

# Absolute residual magnitude:
# used as a proxy for the conditional stochastic scale.
calibration_df["abs_residual"] = calibration_df["residual"].abs()

sigma_target = calibration_df["abs_residual"]


# ============================================================
# 4. Mean-reversion target using rolling AR(1)
# ============================================================

# Number of observations used to estimate local residual persistence.
# For hourly data, 168 = one week.
KAPPA_WINDOW = 168

residual = calibration_df["residual"]

# Rolling AR(1) coefficient:
#
#       r_t = phi_t * r_(t-1) + epsilon_t
#
# phi_t is estimated locally using a rolling window rather
# than from a single observation-to-observation ratio.

lagged_residual = residual.shift(1)

rolling_covariance = (
    residual.rolling(KAPPA_WINDOW)
    .cov(lagged_residual)
)

rolling_variance = (
    lagged_residual.rolling(KAPPA_WINDOW)
    .var()
)

calibration_df["phi"] = (
    rolling_covariance / rolling_variance
)

# Numerical stability:
# phi must be positive for the standard OU transformation
#
#       kappa = -ln(phi) / Delta_t
#
# Here Delta_t = 1 hour.
#
# Values <= 0 are not used because they do not correspond
# to a conventional positive OU mean-reversion speed.

calibration_df.loc[
    calibration_df["phi"] <= 0,
    "phi"
] = np.nan

calibration_df["phi"] = calibration_df["phi"].clip(
    lower=1e-3,
    upper=0.999
)

calibration_df["kappa_target"] = -np.log(
    calibration_df["phi"]
)

# Remove observations for which a rolling estimate
# cannot be obtained.
calibration_df = calibration_df.dropna(
    subset=["kappa_target"]
).copy()


# ============================================================
# 5. Jump identification
# ============================================================

# Residual-based jump criterion:
# an observation is classified as a jump when its absolute
# residual exceeds three standard deviations of the
# calibration-period absolute residual distribution.

jump_threshold = (
    3 * calibration_df["abs_residual"].std()
)

calibration_df["jump_flag"] = (
    calibration_df["abs_residual"] > jump_threshold
).astype(int)


# Jump-size target:
# conditional magnitude of residuals for identified jumps.
jump_size_target = calibration_df.loc[
    calibration_df["jump_flag"] == 1,
    "abs_residual"
]


# ============================================================
# 6. Feature preprocessing
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(with_mean=False),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

X_calibration = calibration_df[
    numeric_features + categorical_features
]

# Fit preprocessing ONLY on the calibration period.
preprocessor.fit(X_calibration)

X_calibration_transformed = preprocessor.transform(
    X_calibration
)


# ============================================================
# 7. Learn stochastic parameter models
# ============================================================

RF_TREES = 200
RF_MAX_DEPTH = 8
RF_SEED = 42


# ------------------------------------------------------------
# Diffusion-scale model
# ------------------------------------------------------------

sigma_model = RandomForestRegressor(
    n_estimators=RF_TREES,
    max_depth=RF_MAX_DEPTH,
    random_state=RF_SEED
)

sigma_model.fit(
    X_calibration_transformed,
    sigma_target
)


# ------------------------------------------------------------
# Jump-probability model
# ------------------------------------------------------------

lambda_model = RandomForestClassifier(
    n_estimators=RF_TREES,
    max_depth=RF_MAX_DEPTH,
    random_state=RF_SEED
)

lambda_model.fit(
    X_calibration_transformed,
    calibration_df["jump_flag"]
)


# ------------------------------------------------------------
# Jump-size model
# ------------------------------------------------------------

X_jump = calibration_df.loc[
    calibration_df["jump_flag"] == 1,
    numeric_features + categorical_features
]

if len(X_jump) > 0:

    X_jump_transformed = preprocessor.transform(X_jump)

    jump_size_model = RandomForestRegressor(
        n_estimators=RF_TREES,
        max_depth=RF_MAX_DEPTH,
        random_state=RF_SEED
    )

    jump_size_model.fit(
        X_jump_transformed,
        jump_size_target
    )

else:

    jump_size_model = None

    print(
        "No jump events were identified; "
        "the jump-size model was not fitted."
    )


# ------------------------------------------------------------
# Mean-reversion model
# ------------------------------------------------------------

kappa_model = RandomForestRegressor(
    n_estimators=RF_TREES,
    max_depth=RF_MAX_DEPTH,
    random_state=RF_SEED
)

kappa_model.fit(
    X_calibration_transformed,
    calibration_df["kappa_target"]
)


# ============================================================
# 8. Predict stochastic parameters for the 2025
#    out-of-sample period
# ============================================================

X_out_of_sample = out_of_sample_df[
    numeric_features + categorical_features
].dropna()

X_out_of_sample_transformed = preprocessor.transform(
    X_out_of_sample
)


# Diffusion scale
sigma_t = sigma_model.predict(
    X_out_of_sample_transformed
)

sigma_t = np.clip(
    sigma_t,
    0,
    None
)


# Conditional jump probability
lambda_t = lambda_model.predict_proba(
    X_out_of_sample_transformed
)[:, 1]

lambda_t = np.clip(
    lambda_t,
    0,
    1
)


# Conditional jump-size scale
if jump_size_model is not None:

    jump_size_t = jump_size_model.predict(
        X_out_of_sample_transformed
    )

    jump_size_t = np.clip(
        jump_size_t,
        0,
        None
    )

else:

    jump_size_t = np.zeros(
        len(X_out_of_sample_transformed)
    )


# Mean-reversion speed
kappa_t = kappa_model.predict(
    X_out_of_sample_transformed
)

kappa_t = np.clip(
    kappa_t,
    0,
    None
)


# ============================================================
# 9. Align deterministic forecast and stochastic parameters
# ============================================================

deterministic_out_of_sample = np.asarray(
    deterministic_future
)

n_steps = min(
    len(deterministic_out_of_sample),
    len(sigma_t),
    len(lambda_t),
    len(jump_size_t),
    len(kappa_t)
)

deterministic_out_of_sample = (
    deterministic_out_of_sample[:n_steps]
)

sigma_t = sigma_t[:n_steps]
lambda_t = lambda_t[:n_steps]
jump_size_t = jump_size_t[:n_steps]
kappa_t = kappa_t[:n_steps]


# ============================================================
# 10. Monte Carlo simulation
# ============================================================

def monte_carlo_variable_params(
    deterministic,
    sigma_t,
    lambda_t,
    jump_size_t,
    kappa_t,
    n_scenarios=200,
    seed=42
):
    """
    Generate stochastic residual scenarios using
    time-varying MRJD parameters.

    Parameters
    ----------
    deterministic : array-like
        Deterministic electricity-price forecast.

    sigma_t : array-like
        Time-varying diffusion scale.

    lambda_t : array-like
        Time-varying jump probability.

    jump_size_t : array-like
        Time-varying conditional jump-size scale.

    kappa_t : array-like
        Time-varying mean-reversion speed.

    n_scenarios : int
        Number of Monte Carlo paths.

    seed : int
        Random seed for reproducibility.
    """

    rng = np.random.default_rng(seed)

    n_steps = len(deterministic)

    scenarios = np.zeros(
        (n_scenarios, n_steps)
    )

    for s in range(n_scenarios):

        residual_path = np.zeros(n_steps)

        # Initial residual
        residual_path[0] = (
            sigma_t[0] * rng.normal()
        )

        for t in range(1, n_steps):

            # Mean reversion toward zero
            drift = (
                -kappa_t[t]
                * residual_path[t - 1]
            )

            # Continuous diffusion component
            diffusion = (
                sigma_t[t]
                * rng.normal()
            )

            # Jump component
            if rng.random() < lambda_t[t]:
                jump = (
                    jump_size_t[t]
                    * rng.normal()
                )
            else:
                jump = 0.0

            # Residual update
            residual_path[t] = (
                residual_path[t - 1]
                + drift
                + diffusion
                + jump
            )

        scenarios[s, :] = residual_path

    return scenarios


# ============================================================
# 11. Generate 2025 Monte Carlo scenarios
# ============================================================

scenarios = monte_carlo_variable_params(
    deterministic=deterministic_out_of_sample,
    sigma_t=sigma_t,
    lambda_t=lambda_t,
    jump_size_t=jump_size_t,
    kappa_t=kappa_t,
    n_scenarios=200,
    seed=42
)


# ============================================================
# 12. Combine deterministic forecast and stochastic residuals
# ============================================================

predicted_prices = (
    deterministic_out_of_sample[None, :]
    + scenarios
)
