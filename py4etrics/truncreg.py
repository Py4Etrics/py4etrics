"""
Created by Tetsu Haruyama
"""

import numpy as np
from scipy.stats import truncnorm
import statsmodels.api as sm
from py4etrics.base_for_models import GenericLikelihoodModel_TobitTruncreg

class Truncreg(GenericLikelihoodModel_TobitTruncreg):
    """
    Method 1:
    Truncreg(endog, exog, left=<-np.inf>, right=<np.inf>).fit()
    endog = dependent variable
    exog = independent variable (add constant if needed)
    left = the threshold value for left-truncation (default:-np.inf)
    right = the threshold value for right-truncation (default:np.inf)

    Method 2:
    formula = 'y ~ 1 + x'
    Truncreg(formula, left=<-np.inf>, right=<np.inf>, data=<DATA>).fit()

    Note:
    Left-truncated Regression if 'left' only is set.
    Right-truncated Regression if 'right' only is set.
    Left- and Right-truncated Regression if 'left' and 'right' both are set.

    """

    def __init__(self, endog, exog, left=None, right=None, **kwds):
        super(Truncreg, self).__init__(endog, exog, **kwds)

        if left == None:
            left = -np.inf
        self.left = left

        if right == None:
            right = np.inf
        self.right = right

    def loglikeobs(self, params):
        s = params[-1]
        beta = params[:-1]

        def _truncreg(y,x,left,right,beta,s):
            Xb = np.dot(x, beta)
            _l = (left - Xb)/np.exp(s)
            _r = (right - Xb)/np.exp(s)
            return truncnorm.logpdf(y,a=_l,b=_r,loc=Xb,scale=np.exp(s))

        return _truncreg(self.endog, self.exog,
                         self.left, self.right, beta, s)


    def fit(self, cov_type='nonrobust', start_params=None, maxiter=10000, maxfun=10000, **kwds):
        # add sigma for summary
        if 'Log(Sigma)' not in self.exog_names:
            self.exog_names.append('Log(Sigma)')
        else:
            pass
        # initial guess
        res_ols = sm.OLS(self.endog, self.exog).fit()
        params_ols = res_ols.params
        sigma_ols = np.log(np.std(res_ols.resid))
        if start_params == None:
            start_params = np.append(params_ols, sigma_ols)

        return super(Truncreg, self).fit(cov_type=cov_type, start_params=start_params,
                                     maxiter=maxiter, maxfun=maxfun, **kwds)

# EOF
