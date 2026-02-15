Path outline for now:
- Began with basic MCMC w/ regime integration, transitions based purely on current regime
- implimented logistic regression & integrated predictors that proved worthwhile in prev. S&P500 project
- CURRENT: change from logistic regression to autoregression, note this is now a "hidden Markov model", can thus integrate gibbs sampling & then hopefully bayesian analysis -- better reflects higher dimensionality
- next(?): tendency to predict large falls in equities and regime stickiness, altering regimes to depend on SD before targeting potential model insensitivity