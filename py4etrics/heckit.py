"""
Heckman correction for sample selection bias (the Heckman procedure).

This is a modified version of the code available at
https://github.com/statsmodels/statsmodels/blob/92ea62232fd63c7b60c60bee4517ab3711d906e3/statsmodels/regression/Heckit.py

Created by Tetsu Haruyama
"""

import numpy as np
import statsmodels.api as sm
import statsmodels.base.model as base
from statsmodels.iolib import summary
from statsmodels.tools.numdiff import approx_fprime
from scipy.stats import norm
import warnings
import pandas as pd
from statsmodels.tools.decorators import cache_readonly, cache_writable    # Tetsu
from statsmodels.compat.python import lrange

class Heckit(base.LikelihoodModel):
    """
    Class for Heckit correction for sample selection bias model.

    Parameters
    ----------
    endog : 1darray
        Data for the dependent variable. Should be set to np.nan for
        censored observations.
    exog : 2darray
        Data for the regression (response) equation. If a constant
        term is desired, the user should directly add the constant
        column to the data before using it as an argument here.
    exog_select : 2darray
        Data for the selection equation. If a constant
        term is desired, the user should directly add the constant
        column to the data before using it as an argument here.
    **kwargs:
        missing=, which can be 'none', 'drop', or 'raise'

    Notes
    -----
    The selection equation should contain at least one variable that
    is not in the regression (response) equation, i.e. the selection
    equation should contain at least one instrument. However, this
    module should still work if the user chooses not to do this.
    """

    def __init__(self, endog, exog, exog_select, **kwargs):

        # check that Z has same index as X (and consequently Y through super().__init__)
        if pd.__name__ in type(endog).__module__ and pd.__name__ in type(exog).__module__:
            if not all(endog.index==exog_select.index):
                raise ValueError("Z indices need to be the same as X and Y indices")


        # shape checks
        if (len(endog) == len(exog)) and (len(endog) == len(exog_select)):
            pass
        else:
            raise ValueError("Y, X, and Z data shapes do not conform with each other.")

        try:
            if (endog.ndim == 1) and (exog.ndim <= 2) and (exog_select.ndim <= 2):
                pass
            else:
                raise ValueError("Y, X, and Z data shapes do not conform with each other.")
        except AttributeError:
            if (np.asarray(endog).ndim == 1) and (np.asarray(exog).ndim <= 2) and (np.asarray(exog_select).ndim <= 2):
                pass
            else:
                raise ValueError("Y, X, and Z data shapes do not conform with each other.")

        # give missing (treated) values in endog variable finite values so that super().__init__
        # does not strip them out -- they will be put back after the call to super().__init__
        treated = np.asarray(~np.isnan(endog))

        try:
            endog_nomissing = endog.copy()
            endog_nomissing[~treated] = -99999
        except (TypeError, AttributeError):
            endog_nomissing = [endog[i] if treated[i] else -99999 for i in range(len(treated))]

        # create 1-D array that will be np.nan for every row of exog_select that has any missing
        # values and a finite value otherwise for the call to super().__init__ so that it can
        # strip out rows where exog_select has missing data if missing option is set

        if np.asarray(exog_select).ndim==2:
            exog_select_1dnan_placeholder = \
                [np.nan if any(np.isnan(row)) else 1 for row in np.asarray(exog_select)]
        else:  # assume ==1
            exog_select_1dnan_placeholder = [np.nan if np.isnan(row) else 1 for row in np.asarray(exog_select)]

        if pd.__name__ in type(endog).__module__:
            exog_select_1dnan_placeholder = pd.Series(exog_select_1dnan_placeholder, index=endog.index)
        else:
            exog_select_1dnan_placeholder = np.array(exog_select_1dnan_placeholder)

        # create array of sequential row positions so that rows of exog_select that have missing
        # data can be identified after call to super().__init__
        obsno = np.array(list(range(len(endog))))

        # call super().__init__
        super(Heckit, self).__init__(
            endog_nomissing, exog=exog,
            exog_select_1dnan_placeholder=exog_select_1dnan_placeholder, obsno=obsno,
            treated=treated,
            **kwargs)

        # put np.nan back into endog for treated rows
        self.endog = self.data.endog = \
            np.asarray(
                [self.endog[i] if self.treated[i] else np.nan for i in range(len(self.treated))]
                )

        # strip out rows stripped out by call to super().__init__ in Z variable
        self.exog_select = np.asarray([np.asarray(exog_select)[obs] for obs in self.obsno])

        # store variable names of exog_select
        try:
            self.exog_select_names = exog_select.columns.tolist()
        except AttributeError:
            self.exog_select_names = None

        # delete attributes created by the call to super().__init__ that are no longer needed
        del self.exog_select_1dnan_placeholder
        del self.obsno


        # store observation counts
        self.nobs_total = len(endog)
        self.nobs_uncensored = self.nobs = np.sum(self.treated)
        self.nobs_censored = self.nobs_total - self.nobs_uncensored


    def initialize(self):
        self.wendog = self.endog
        self.wexog = self.exog
        self._df_model = None   # Tetsu
        self.rank = None    # Tetsu
        self._df_resid = None # Tetsu


    def whiten(self, data):
        """
        Model whitener for Heckit correction model does nothing.
        """
        return data


    def get_datamats(self):
        Y = np.asarray(self.endog)
        Y = Y[self.treated]

        X = np.asarray(self.exog)
        X = X[self.treated,:]
        if X.ndim==1:
            X = np.atleast_2d(X).T

        Z = np.asarray(self.exog_select)
        if Z.ndim==1:
            Z = np.atleast_2d(Z).T

        return Y, X, Z


    # Tetsu ------ starts -----
    @property
    def df_model(self):
        """
        The model degree of freedom, defined as the rank of the regressor
        matrix minus 1 if a constant is included.
        """
        X = np.asarray(self.exog)
        X = X[self.treated,:]
        if self._df_model is None:
            if self.rank is None:
                self.rank = np.linalg.matrix_rank(X)
                self._df_model = float(self.rank - self.k_constant)
        return self._df_model


    @df_model.setter
    def df_model(self, value):
        self._df_model = value

    @property
    def df_resid(self):
        """
        The residual degree of freedom, defined as the number of observations
        minus the rank of the regressor matrix.
        """
        X = np.asarray(self.exog)
        X = X[self.treated,:]

        if self._df_resid is None:
            if self.rank is None:
                self.rank = np.linalg.matrix_rank(X)
            self._df_resid = self.nobs - self.rank
        return self._df_resid

    @df_resid.setter
    def df_resid(self, value):
        self._df_resid = value

    # Tetsu ----- ends -----





    def fit(self, method='twostep',
            cov_type_1='nonrobust', cov_type_2='nonrobust',  #  Tetsu ################
             start_params_mle=None, method_mle=None, maxiter_mle=None, **kwargs_mle):
        """
        Fit the Heckit selection model.

        Parameters
        ----------
        method : str
            Can only be "2step", which uses Heckit's two-step method.
        start_params_mle: 1darray
            If using MLE, starting parameters.
        method_mle: str
            If using MLE, the MLE estimation method.
        maxiter_mle: scalar
            If using MLE, the maximum number of iterations for MLE estimation.
        **kwargs_mle:
            Other arguments to pass to optimizer for MLE estimation.

        Returns
        -------
        A HeckitResults class instance.

        See Also
        ---------
        HeckitResults


        """

        ## Show warning to user if estimation is by two-step but MLE arguments were also provided
        if method=='twostep':
            if start_params_mle is not None or method_mle is not None or maxiter_mle is not None or \
                len(kwargs_mle.keys())>0:
                warnings.warn('The user chose to estimate the Heckit model by Two-Step,' + \
                    ' but MLE arguments were provided. Extraneous MLE arguments will be ignored.')

        results = self._fit_twostep(cov_type_1=cov_type_1,cov_type_2=cov_type_2)  # Tetsu : cov_type_1=cov_type_1,cov_type_2=cov_type_2
        return results


    def _fit_twostep(self,cov_type_1='nonrobust',cov_type_2='nonrobust'):  # Tetsu : cov_type_1=cov_type_1,cov_type_2=cov_type_2
        ########################################################################
        # PRIVATE METHOD
        # Fits using Heckit two-step from Heckit (1979).
        ########################################################################

        ## prep data
        Y, X, Z = self.get_datamats()

        ## Step 1
        step1model = sm.Probit(self.treated, Z)
        step1res = step1model.fit(disp=False,cov_type=cov_type_1)  # Tets cov_type=cov_type_1
        step1_fitted = np.atleast_2d(step1res.fittedvalues).T
        step1_varcov = step1res.cov_params()

        inverse_mills = norm.pdf(step1_fitted)/norm.cdf(step1_fitted)

        ## Step 2
        W = np.hstack((X, inverse_mills[self.treated] ) )
        step2model = sm.OLS(Y, W)
        step2res = step2model.fit(cov_type=cov_type_2)    # Tetsu : cov_type_2=cov_type

        params = step2res.params[:-1]
        betaHat_inverse_mills = step2res.params[-1]


        ## Compute standard errors
        # Compute estimated error variance of censored regression
        delta = np.multiply(inverse_mills, inverse_mills + step1_fitted)[self.treated]

        sigma2Hat = step2res.resid.dot(step2res.resid) / self.nobs_uncensored + \
            (betaHat_inverse_mills**2 * sum(delta)) / self.nobs_uncensored
        sigma2Hat = sigma2Hat[0]
        sigmaHat = np.sqrt(sigma2Hat)
        rhoHat = betaHat_inverse_mills / sigmaHat

        # compute standard errors of beta estimates of censored regression
        delta_1d = delta.T[0]

        Q = rhoHat**2 * ((W.T*delta_1d).dot(Z[self.treated])).dot(step1_varcov).dot((Z[self.treated].T*delta_1d).dot(W))

        WT_W_inv = np.linalg.inv(W.T.dot(W))
        WT_R = W.T*(1 - rhoHat**2 * delta_1d)
        normalized_varcov_all = WT_W_inv.dot(WT_R.dot(W)+Q).dot(WT_W_inv)
        del WT_W_inv
        del WT_R

        del delta_1d

        normalized_varcov = normalized_varcov_all[:-1,:-1]

        varcov_all = sigma2Hat * normalized_varcov_all
        varcov = varcov_all[:-1,:-1]

        stderr_all = np.sqrt(np.diag(varcov_all))
        stderr = stderr_all[:-1]
        stderr_betaHat_inverse_mills = stderr_all[-1]


        ## store results
        results = HeckitResults(self, params, normalized_varcov, sigma2Hat,
            select_res=step1res,
            params_inverse_mills=betaHat_inverse_mills, stderr_inverse_mills=stderr_betaHat_inverse_mills,
            var_reg_error=sigma2Hat, corr_eqnerrors=rhoHat,
            method='twostep', cov_type_1=cov_type_1, cov_type_2=cov_type_2)  #  cov_type_1=cov_type_1, cov_type_2=cov_type_2

        return results


    def predict(self, params, exog=None):
        """
        Return linear predicted values from a design matrix.

        Parameters
        ----------
        exog : array-like
            Design / exogenous data
        params : array-like, optional after fit has been called
            Parameters of a linear model

        Returns
        -------
        An array of fitted values

        Notes
        -----
        If the model has not yet been fit, params is not optional.
        """
        if exog is None:
            exog = self.exog
        return np.dot(exog, params)

        if self._results is None and params is None:
            raise ValueError("If the model has not been fit, then you must specify the params argument.")
        if self._results is not None:
            return np.dot(exog, self._results.params)
        else:
            return np.dot(exog, params)


class HeckitResults(base.LikelihoodModelResults):
    """
    Class to represent results/fits for Heckit model.

    Attributes
    ----------
    select_res : ProbitResult object
        The ProbitResult object created when estimating the selection equation.
    params_inverse_mills : scalar
        Parameter estimate of the coef on the inverse Mills term in the second step.
    stderr_inverse_mills : scalar
        Standard error of the parameter estimate of the coef on the inverse Mills
        term in the second step.
    var_reg_error : scalar
        Estimate of the "sigma" term, i.e. the error variance estimate of the
        regression (response) equation
    corr_eqnerrors : scalar
        Estimate of the "rho" term, i.e. the correlation estimate of the errors between the
        regression (response) equation and the selection equation
    method : string
        The method used to produce the estimates, i.e. 'twostep', 'mle'
    """

    def __init__(self, model, params, normalized_cov_params=None, scale=1.,
        select_res=None,
        params_inverse_mills=None, stderr_inverse_mills=None,
        var_reg_error=None, corr_eqnerrors=None,
        method=None, cov_type_1=None, cov_type_2=None):  # Tetsu : cov_type_1=None, cov_type_2=None

        super(HeckitResults, self).__init__(model, params,
                                                normalized_cov_params,
                                                scale)

        self.select_res = select_res
        self.params_inverse_mills = params_inverse_mills
        self.stderr_inverse_mills = stderr_inverse_mills
        self.var_reg_error = var_reg_error
        self.corr_eqnerrors = corr_eqnerrors
        self.method = method


    # Tetsu ----- starts ------
        self.cov_type_1 = cov_type_1
        self.cov_type_2 = cov_type_2
        self.df_model = model.df_model
        self.df_resid = model.df_resid
        dropnan = np.asarray(~np.isnan(self.model.endog))
        self.endog = self.model.endog[dropnan]
        self.exog = self.model.exog[dropnan]
        self.wendog = self.endog
        self.wexog = self.exog
        self.pinv_wexog = np.linalg.pinv(self.exog)

        if not hasattr(self, 'use_t'):
            self.use_t = False
            # self.use_t = True

        if not hasattr(self.select_res, 'use_t'):
            self.select_res.use_t = False

    @cache_readonly
    def nobs(self):
        """Number of observations n."""
        # return float(self.model.nobs_total)
        return float(len(self.endog))

    @cache_readonly
    def fittedvalues(self):
        """The predicted values for the original (unwhitened) design."""
        # return self.model.predict(self.params, self.model.exog)
        return self.model.predict(self.params, self.exog)

    @cache_readonly
    def resid(self):
        """The residuals of the model."""
        return self.endog - self.model.predict(self.params, self.exog)

    @cache_readonly
    def wresid(self):
        """
        The residuals of the transformed/whitened regressand and regressor(s)
        """
        return self.wendog - self.model.predict(self.params, self.wexog)

    @cache_readonly
    def ssr(self):
        """Sum of squared (whitened) residuals."""
        wresid = self.wresid
        return np.dot(wresid, wresid)

    @cache_readonly
    def centered_tss(self):
        """The total (weighted) sum of squares centered about the mean."""
        model = self.model
        weights = getattr(model, 'weights', None)
        sigma = getattr(model, 'sigma', None)
        if weights is not None:
            # mean = np.average(model.endog, weights=weights)
            mean = np.average(self.endog, weights=weights)
            return np.sum(weights * (self.endog - mean)**2)

        elif sigma is not None:
            # Exactly matches WLS when sigma is diagonal
            iota = np.ones_like(self.endog)
            iota = model.whiten(iota)
            mean = self.wendog.dot(iota) / iota.dot(iota)
            err = self.endog - mean
            err = model.whiten(err)
            return np.sum(err**2)
        else:
            centered_endog = self.wendog - self.wendog.mean()
            return np.dot(centered_endog, centered_endog)

    @cache_readonly
    def uncentered_tss(self):
        """
        Uncentered sum of squares.  Sum of the squared values of the
        (whitened) endogenous response variable.
        """
        wendog = self.wendog
        return np.dot(wendog, wendog)

    @cache_readonly
    def ess(self):
        """Explained sum of squares. If a constant is present, the centered
        total sum of squares minus the sum of squared residuals. If there is no
        constant, the uncentered total sum of squares is used."""

        if self.k_constant:
            return self.centered_tss - self.ssr
        else:
            return self.uncentered_tss - self.ssr

    @cache_readonly
    def rsquared(self):
        """
        R-squared of a model with an intercept.  This is defined here
        as 1 - `ssr`/`centered_tss` if the constant is included in the
        model and 1 - `ssr`/`uncentered_tss` if the constant is
        omitted.
        """
        if self.k_constant:
            return 1 - self.ssr/self.centered_tss
        else:
            return 1 - self.ssr/self.uncentered_tss

    @cache_readonly
    def rsquared_adj(self):
        """
        Adjusted R-squared.  This is defined here as 1 -
        (`nobs`-1)/`df_resid` * (1-`rsquared`) if a constant is
        included and 1 - `nobs`/`df_resid` * (1-`rsquared`) if no
        constant is included.
        """
        return 1 - (np.divide(self.nobs - self.k_constant, self.df_resid)
                    * (1 - self.rsquared))

    @cache_readonly
    def mse_model(self):
        """
        Mean squared error the model. This is the explained sum of
        squares divided by the model degrees of freedom.
        """
        return self.ess/self.df_model


    @cache_readonly
    def mse_resid(self):
        """
        Mean squared error of the residuals.  The sum of squared
        residuals divided by the residual degrees of freedom.
        """
        return self.ssr/self.df_resid


    @cache_readonly
    def mse_total(self):
        """
        Total mean squared error.  Defined as the uncentered total sum
        of squares divided by n the number of observations.
        """
        if self.k_constant:
            return self.centered_tss / (self.df_resid + self.df_model)
        else:
            return self.uncentered_tss / (self.df_resid + self.df_model)


    @cache_readonly
    def fvalue(self):
        """F-statistic of the fully specified model.  Calculated as the mean
        squared error of the model divided by the mean squared error of the
        residuals."""
        if hasattr(self, 'cov_type_1') and self.cov_type_1 != 'nonrobust':
            # with heteroscedasticity or correlation robustness
            k_params = self.normalized_cov_params.shape[0]
            mat = np.eye(k_params)
            const_idx = self.model.data.const_idx
            # TODO: What if model includes implicit constant, e.g. all
            #       dummies but no constant regressor?
            # TODO: Restats as LM test by projecting orthogonalizing
            #       to constant?
            if self.model.data.k_constant == 1:
                # if constant is implicit, return nan see #2444
                if const_idx is None:
                    return np.nan

                idx = lrange(k_params)
                idx.pop(const_idx)
                mat = mat[idx]  # remove constant
                if mat.size == 0:  # see  #3642
                    return np.nan
            ft = self.f_test(mat)
            # using backdoor to set another attribute that we already have
            self._cache['f_pvalue'] = ft.pvalue
            return ft.fvalue
        else:
            # for standard homoscedastic case
            return self.mse_model/self.mse_resid


    @cache_readonly
    def f_pvalue(self):
        """p-value of the F-statistic"""
        from scipy import stats
        return stats.f.sf(self.fvalue, self.df_model, self.df_resid)


    def _HCCM(self, scale):
        H = np.dot(self.pinv_wexog,
                   scale[:, None] * self.pinv_wexog.T)
        return H

    @cache_readonly
    def cov_HC1(self):
        """
        See statsmodels.RegressionResults
        """

        self.het_scale = self.nobs/(self.df_resid)*(self.wresid**2)
        cov_HC1 = self._HCCM(self.het_scale)
        return cov_HC1

    @cache_readonly
    def HC1_se(self):
        """
        See statsmodels.RegressionResults
        """
        return np.sqrt(np.diag(self.cov_HC1))

    # Tetsu ----- ends -----



    def summary(self, yname=None, xname=None, zname=None, title=None, alpha=.05):
        """Summarize the Heckit model Results

        Parameters
        -----------
        yname : string, optional
            Default is `y`
        xname : list of strings, optional
            Default is `x_##` for ## in p the number of regressors
            in the regression (response) equation.
        zname : list of strings, optional
            Default is `z_##` for ## in p the number of regressors
            in the selection equation.
        title : string, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary
            results

        """

        ## Put in Z name detected from data if none supplied, unless that too could not be
        ## inferred from data, then put in generic names
        if zname is None and self.model.exog_select_names is not None:
            zname=self.model.exog_select_names
        elif zname is None and self.model.exog_select_names is None:
            try:
                zname = ['z' + str(i) for i in range(len(self.model.exog_select[0]))]
                zname[0]  = 'z0_or_zconst'
            except TypeError:
                zname = 'z0_or_zconst'

        try:  # for Python 3
            if isinstance(zname, str):
                zname = [zname]
        except NameError:  # for Python 2
            if isinstance(zname, basestring):
                zname = [zname]


        ## create summary object
        # instantiate the object
        smry = summary.Summary()

        top_left = [('Dep. Variable:', None),
                    ('Model:', None),
                    ('Method:', ['Heckman Two-Step']),
                    ('Date:', None),
                    ('Time:', None),
                    ('No. Total Obs.:', ["%#i" % self.model.nobs_total]),
                    ('No. Censored Obs.:', ["%#i" % self.model.nobs_censored]),
                    ('No. Uncensored Obs.:', ["%#i" % self.model.nobs_uncensored]),
                    ]


        top_right = [('R-squared:', ["%#8.3f" % self.rsquared]),
                     ('Adj. R-squared:', ["%#8.3f" % self.rsquared_adj]),
                     ('F-statistics:', ["%#8.3f" % self.fvalue]),
                     ('Prob (F-statistic):', ["%#8.3f" % self.f_pvalue]),  # Tetsu added
                     ('Cov in 1st Stage:', [self.cov_type_1]),  # Tetsu added
                     ('Cov in 2nd Stage:', [self.cov_type_2]),  # Tetsu added
                     ]

        if title is None:
            title = self.model.__class__.__name__ + ' ' + "Regression Results"

        smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                          yname=yname, xname=xname, title=title)

        # add the Heckit-corrected regression table
        smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha,
                             use_t=self.use_t)

        # add the selection equation estimates table
        smry.add_table_params(self.select_res, yname=yname, xname=zname, alpha=alpha,
                             use_t=self.select_res.use_t)

        # add the estimate to the inverse Mills estimate (z-score)
        smry.add_table_params(
            base.LikelihoodModelResults(None, np.atleast_1d(self.params_inverse_mills),
            normalized_cov_params=np.atleast_1d(self.stderr_inverse_mills**2), scale=1.),
            yname=None, xname=['IMR (Lambda)'], alpha=alpha,
            use_t=False)

        # add point estimates for rho and sigma
        diagn_left = [('rho:', ["%#6.3f" % self.corr_eqnerrors]),
                      ('sigma:', ["%#6.3f" % np.sqrt(self.var_reg_error)]),
                      ]

        diagn_right = [
                       ]

        smry.add_table_2cols(self, gleft=diagn_left, gright=diagn_right,
                          yname=yname, xname=xname,
                          title="")

        # add text at end
        smry.add_extra_txt(['First table are the estimates for the regression (response) equation.',
            'Second table are the estimates for the selection equation.',
            'Third table is the estimate for the coef of the inverse Mills ratio (Heckman\'s Lambda).'])

        ## return
        return smry
