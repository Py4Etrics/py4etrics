'''
Specification test for heteroscedasticity in Logit/Probit models

The tests are based on
(1) Wooldridge 2010, section 15.5.3
(2) https://www.statalist.org/forums/forum/general-stata-discussion/general/1292180-test-for-heteroskedasticity-in-logit-probit-models

Check consistency with Stata using the following

----- Stata code starts -----

// https://www.statalist.org/forums/forum/general-stata-discussion/general/1292180-test-for-heteroskedasticity-in-logit-probit-models
// https://www.stata.com/manuals13/p_predict.pdf
// https://www.stata.com/manuals13/rpredict.pdf

import delimited "/Users/tetsu/Documents/My_Simulation/Python/projects/wooldridge/raw_data/data_csv/mroz.csv"

probit inlf nwifeinc educ exper expersq age kidslt6 kidsge6

predict xbhat, xb
// the following give s the same resutl
// predict xbhat, index

// 'c' for continuous variables for interaction term
// '#' multiplication for interaction terms
probit inlf nwifeinc educ exper expersq age kidslt6 kidsge6 c.xbhat#c.nwifeinc c.xbhat#c.educ c.xbhat#c.exper c.xbhat#c.expersq c.xbhat#c.age c.xbhat#c.kidslt6 c.xbhat#c.kidsge6

test c.xbhat#c.nwifeinc c.xbhat#c.educ c.xbhat#c.exper c.xbhat#c.expersq c.xbhat#c.age c.xbhat#c.kidslt6 c.xbhat#c.kidsge6

// for reference ---------------------------------------------------------------
hetprobit inlf nwifeinc educ exper expersq age kidslt6 kidsge6, het(nwifeinc educ exper expersq age kidslt6 kidsge6)

----- Stata code ends -----

Created by Tetsugen Haruyama
'''


import numpy as np
import pandas as pd
import statsmodels.api as sm


def het_test_logit(results):
    """
    Wald Test for Logit
    -------------------
    H0: homoscedasticity
    HA: heteroscedasticity

    Parameters
    ----------
    results : Logit results instance


    Returns
    -------
    Wald test statistic
    p-value
    Degree of Freedom
        The number of restrictions, which are equivalent to the number of
        explanatory variables, excluding a constant term

    References
    ----------
    The test is based on
    (1) Wooldridge 2010, section 15.5.3
    (2) https://www.statalist.org/forums/forum/general-stata-discussion/general/1292180-test-for-heteroskedasticity-in-logit-probit-models
    """

    yhat = results.predict(which="linear")  # original fitted values
    exog_var = results.model.exog  # original exog
    exog_df = pd.DataFrame(exog_var)  # convert to DataFrame

    try:  # drop a column of a constant if any
        tt = exog_df.nunique()
        idx_1 = list(tt).index(1.0)
        exog_df = exog_df.drop(idx_1, axis=1)
    except ValueError:
        pass

    num_para = exog_df.shape[1]  # no of non-constant parameters

    # X = np.exp(yhat).reshape(len(yhat),1) * exog_df.values
    X = yhat.reshape(len(yhat), 1) * exog_df.values

    endog = results.model.endog
    exog = np.column_stack((results.model.exog, X))
    res_test = sm.Logit(endog, exog).fit(disp=False)

    A = np.identity(len(res_test.params))
    A = A[-num_para:, :]
    s = res_test.wald_test(A, scalar=True)
    return print('H0: homoscedasticity\nHA: heteroscedasticity\n',
                 f'\nWald test: {s.statistic:.3f}',
                 f'\np-value: {s.pvalue:.3f}',
                 f'\ndf freedom: {s.df_denom:.0f}'
                 )


# --------------------------------------------------------------


def het_test_probit(results):
    """
    Wald検定 for Probit
    ------------------
    H0: homoscedasticity
    HA: heteroscedasticity

    Parameters
    ----------
    results : Logit results instance

    Returns
    -------
    Wald test statistic
    p-value
    Degree of Freedom
        The number of restrictions, which are equivalent to the number of
        explanatory variables, excluding a constant term

    References
    ----------
    The test is based on
    (1) Wooldridge 2010, section 15.5.3
    (2) https://www.statalist.org/forums/forum/general-stata-discussion/general/1292180-test-for-heteroskedasticity-in-logit-probit-models
    """

    yhat = results.predict(which="linear")  # original fitted values
    exog_var = results.model.exog  # original exog
    exog_df = pd.DataFrame(exog_var)  # convert to DataFrame

    try:  # drop a column of a constant if any
        tt = exog_df.nunique()
        idx_1 = list(tt).index(1.0)
        exog_df = exog_df.drop(idx_1, axis=1)
    except ValueError:
        pass

    num_para = exog_df.shape[1]   # no of non-constant parameters

    # X = np.exp(yhat).reshape(len(yhat),1) * exog_df.values
    X = yhat.reshape(len(yhat), 1) * exog_df.values

    endog = results.model.endog
    exog = np.column_stack((results.model.exog, X))
    res_test = sm.Probit(endog, exog).fit(disp=False)

    A = np.identity(len(res_test.params))
    A = A[-num_para:, :]
    s = res_test.wald_test(A, scalar=True)
    return print('H0: homoscedasticity\nHA: heteroscedasticity\n',
                 f'\nWald test: {s.statistic:.3f}',
                 f'\np-value: {s.pvalue:.3f}',
                 f'\ndf freedom: {s.df_denom:.0f}'
                 )
