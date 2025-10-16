# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import polars as pl
import statsmodels.formula.api as smf
import bambi as bmb
import arviz as az

# %%
# Estimate P_t = B0 + B1p_t-1 + b2 s11t + ut
# Change the sample period to be form 2-60
df = pl.read_csv("data/raw/dgp_determandstoch.csv", ignore_errors=True)
df = df.with_columns(p_stock_shift=pl.col("P_STOCH").shift())
results = smf.ols("P_STOCH ~ p_stock_shift + STEP11",data=df.to_pandas()).fit()
print(results.summary())

# %%
model = bmb.Model("P_STOCH ~ p_stock_shift + STEP11", data=df.to_pandas())
results = model.fit()
az.plot_trace(results)
az.summary(results)
