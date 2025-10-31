import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
import statsmodels.api as sm
import os
import warnings
import numpy as np


# %%


import pandas as pd

def run_lme(
    df=None,
    endog=None,
    exog=None,
    group=None,
    endog_col=None,
    exog_cols=None,
    group_col=None,
    add_const=True,
    dropna=True
):
    """
    Run a linear mixed-effects model with flexible input options.

    Parameters:
        df : pd.DataFrame, optional
            DataFrame containing data. Required if column names are used.
        endog : array-like, optional
            Dependent variable (Y). Overrides df + endog_col if provided.
        exog : DataFrame or array-like, optional
            Independent variables (X). Overrides df + exog_cols if provided.
        group : array-like, optional
            Grouping variable. Overrides df + group_col if provided.
        endog_col : str, optional
            Column name of the dependent variable in df.
        exog_cols : list of str, optional
            Column names of independent variables in df.
        group_col : str, optional
            Column name of the group variable in df.
        add_const : bool
            Whether to add a constant to exog.
        dropna : bool
            Whether to drop NA rows before fitting.

    Returns:
        model, fit_result
    """
    if df is not None:
        if endog is None and endog_col:
            endog = df[endog_col]
        if exog is None and exog_cols:
            exog = df[exog_cols]
        if group is None and group_col:
            group = df[group_col]

    if dropna:
        df_temp = pd.concat([endog, exog, group], axis=1).dropna()
        endog = df_temp.iloc[:, 0]
        exog = df_temp.iloc[:, 1:-1]
        group = df_temp.iloc[:, -1]

    if add_const:
        exog = sm.add_constant(exog, has_constant='add')

    model = MixedLM(endog, exog, groups=group)
    result = model.fit()
    return model, result


