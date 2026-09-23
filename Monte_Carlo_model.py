
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline

# --------------------------------
# Align validation and residuals
# --------------------------------
test_df = df2025.ffill().bfill().copy()
val_df = val_df.copy()

min_len = min(len(res_val), len(val_df))
res_val = np.array(res_val[:min_len])
val_df = val_df.iloc[:min_len].copy()

# --------------------------------
# Residual-based features
# --------------------------------
val_df['residual'] = res_val
val_df['residual_lag1'] = val_df['residual'].shift(1)
val_df = val_df.dropna(subset=['residual', 'residual_lag1']).copy()

# Kappa (speed of mean reversion)
val_df['phi'] = val_df['residual'] / val_df['residual_lag1']
val_df['phi'] = val_df['phi'].clip(-0.99, 0.99)
val_df['kappa_target'] = -np.log(np.abs(val_df['phi']))
val_df = val_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['kappa_target'])

# Sigma
val_df['abs_res'] = val_df['residual'].abs()
sigma_target = val_df['abs_res']

# Jump flags
jump_threshold = 3 * val_df['abs_res'].std()
val_df['jump_flag'] = (val_df['abs_res'] > jump_threshold).astype(int)
jump_std_target = val_df.loc[val_df['jump_flag'] == 1, 'abs_res']

# --------------------------------
# Unified Preprocessor
# --------------------------------

preproc = ColumnTransformer([
    ('num', StandardScaler(with_mean=False), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Fit preprocessor ONCE on validation data
X_val = val_df[numeric_features + categorical_features]
preproc.fit(X_val)

# Transform once
X_val_transformed = preproc.transform(X_val)

# --------------------------------
# Train stochastic parameter models
# --------------------------------
sigma_model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
sigma_model.fit(X_val_transformed, sigma_target)

lambda_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
lambda_model.fit(X_val_transformed, val_df['jump_flag'])

X_jump = val_df.loc[val_df['jump_flag'] == 1, numeric_features + categorical_features]
if len(X_jump) > 0:
    X_jump_transformed = preproc.transform(X_jump)
    jump_std_model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    jump_std_model.fit(X_jump_transformed, jump_std_target)
else:
    jump_std_model = None
    print("⚠️ No jump events found — jump_std_model skipped.")

kappa_model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
kappa_model.fit(X_val_transformed, val_df['kappa_target'])

# --------------------------------
# Predict stochastic parameters for 2025
# --------------------------------
X_test = test_df[numeric_features + categorical_features].dropna().copy()
X_test_transformed = preproc.transform(X_test)

sigma_t = sigma_model.predict(X_test_transformed)
lambda_t = lambda_model.predict_proba(X_test_transformed)[:, 1]

if jump_std_model is not None:
    jump_std_t = jump_std_model.predict(X_test_transformed)
else:
    jump_std_t = np.zeros(len(X_test_transformed))

kappa_t = kappa_model.predict(X_test_transformed)
predicted_prices = np.array([])
min_len = min(len(deterministic_future), len(kappa_t))
deterministic_future = deterministic_future
min_val=df['Day Ahead Auction (DK1)'].min()
max_val=df['Day Ahead Auction (DK1)'].max()

def monte_carlo_variable_params(deterministic, sigma_t, lambda_t, jump_std_t, kappa_t,
                                n_scenarios=200, seed=None):
    rng = np.random.default_rng(seed)

    n_steps = len(deterministic)
    scenarios = np.zeros((n_scenarios, n_steps))

    for s in range(n_scenarios):
        X = np.zeros(n_steps)
        X[0] = rng.normal(0, sigma_t[0])

        for t in range(1, n_steps-1):
            drift = -kappa_t[t] * X[t-1]
            diffusion = sigma_t[t] * rng.normal()
            jump = rng.normal(0, jump_std_t[t]) if rng.random() < lambda_t[t] else 0

            X[t] = X[t-1] + drift + diffusion + jump
            X[t] = np.clip(X[t], min_val, max_val)

        scenarios[s, :] = X

    return scenarios


scenarios = monte_carlo_variable_params(deterministic_future, sigma_t, lambda_t, jump_std_t, kappa_t)

predicted_prices = deterministic_future + scenarios
