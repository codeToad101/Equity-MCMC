#abstracted iteraction of prev. models
import numpy as np
import arviz as az
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from arch import arch_model
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import t
from statsmodels.tsa.stattools import acf

class IndexSimulator:
    def __init__(self, prices: pd.Series):
        self.prices = prices
        self.returns = prices.pct_change().dropna()
        self.log_returns = np.log(prices / prices.shift(1)).dropna().values * 100
        self.HMC_r_obs = None
        self.HMC_trace = None
        self.garch_res = None
        self.transition_matrix = None


        mean_return = self.returns.mean()
        std_return = self.returns.std()
        up_threshold = mean_return + std_return
        down_threshold = mean_return - std_return

        self.Markov_states = self.returns.apply(lambda x: self.classify_state(x, up_threshold, down_threshold)) #change
        prev_states = self.Markov_states.shift(1)

        # Create a transition matrix and Normalize to get probabilities
        states = ["Down", "Stagnant", "Up"]
        transition_counts = pd.crosstab(prev_states, self.Markov_states)
        transition_counts = transition_counts.reindex(index=states, columns=states, fill_value=0)
        row_sums = transition_counts.sum(axis=1).replace(0, 1) #alt is transition_counts = transition_counts + 1e-6
        self.transition_matrix = transition_counts.div(row_sums, axis=0)
        self.equilibrium_matrix = self.find_equilibrium(self.transition_matrix)
    
    def classify_state(self, return_value, up_thresh, down_thresh):
        if return_value > up_thresh:
            return 'Up'
        elif return_value < down_thresh:
            return 'Down'
        else:
            return 'Stagnant'

    def find_equilibrium(self, P):
        """
        Compute stationary distribution of a Markov transition matrix P.
        Assumes P is row-stochastic (rows sum to 1).
        """

        # Ensure rows sum to 1 (defensive normalization)
        #P = P / P.sum(axis=1, keepdims=True)
        P = P.div(P.sum(axis=1), axis=0).fillna(0)

        # Compute eigenvalues/eigenvectors of P^T
        eigvals, eigvecs = np.linalg.eig(P.T)

        # Find eigenvector corresponding to eigenvalue 1
        idx = np.argmin(np.abs(eigvals - 1))
        stationary = np.real(eigvecs[:, idx])

        # Normalize to sum to 1
        stationary = stationary / stationary.sum()

        # Ensure non-negative (numerical safeguard)
        stationary = np.maximum(stationary, 0)
        stationary = stationary / stationary.sum()

        return stationary

    # Model fitting
    def fit_hmc_sv(self, draws=2000, tune=2000):
        # Build PyMC model, sample posterior
        T = len(self.log_returns)

        with pm.Model() as model:

            # ----- Priors -----
            mu = pm.Normal("mu", mu=0, sigma=1) #assumed mean/SD of return
            #mu = pm.Normal("mu", mu=0, sigma=5)

            u = pm.Beta("u", alpha=1, beta=1)
            phi = pm.Deterministic("phi", 2 * u - 1) #uniform (-1, 1)

            #movement of volatility
            #sigma_eta = pm.HalfNormal("sigma_eta", sigma=1.5) #for non-volatile stocks
            sigma_eta = pm.HalfNormal("sigma_eta", sigma=0.2) #for volatile stocks

            #mu_h = pm.Normal("mu_h", mu=5, sigma=2) #shift in h[0] to reflect scaled data

            # ----- Latent log volatility -----
            #mu_h = pm.Normal("mu_h", mu=0, sigma=5)
            #c = pm.Deterministic("c", mu_h * (1 - phi))

            h = pm.AR( #h_t evolves as AR(1), centered at 0 for computational capability
                "h",
                rho=phi,
                sigma=sigma_eta,
                constant=False, #from c = false to c for recenter attempt
                shape=T
            )

            #h = pm.Deterministic("h", h_raw + mu_h)

            # ----- Observation volatility -----
            sigma_t = pm.Deterministic("sigma_t", pt.exp(h / 2)) #gets direct vol

            nu_raw = pm.Exponential("nu_raw", 1/10) #assumes mild fat tails
            nu = pm.Deterministic("nu", nu_raw + 2) #safety, neeps nu > 2

            # ----- Likelihood -----
            self.HMC_r_obs = pm.StudentT( #realistic distribution for tails
                "r_obs",
                nu=nu,
                mu=mu,
                sigma=sigma_t,
                observed=self.log_returns
            )
            self.HMC_trace = pm.sample(
                draws=draws, #posteriar samples
                tune=tune,
                target_accept=0.95, #slower runtime, should improve accuracy
                chains=4, #need to rerun as 4!
                cores=2
            )
            summary = az.summary(self.HMC_trace, var_names=["mu", "phi", "sigma_eta", "nu"])
            print(summary[["r_hat", "ess_bulk"]])

            # divs = self.HMC_trace.sample_stats["diverging"].sum().values
            # print("Divergences:", divs)
            #pass
    
    def fit_garch(self):
        # Fit arch_model(returns)
        # Fit GARCH(1,1) with Student-t errors
        garch = arch_model(
            self.log_returns,
            mean='Constant',      # could change later
            vol='GARCH',
            p=1,
            q=1,
            dist='t'
        )

        self.garch_res = garch.fit(disp="off")
        #pass
    
    # Monte Carlo simulation
    def simulate_hmc(self, n_paths=500, horizon=252):
        # Use posterior draws
        posterior = self.HMC_trace.posterior
        h_sim = np.zeros(horizon)
        r_sim = np.zeros(horizon)

        all_paths = []
        for i in range(n_paths):
            draw_idx = np.random.randint(len(posterior["phi"].values.flatten())) #for unbiased sampling doy

            h_sim = np.zeros(horizon)
            r_sim = np.zeros(horizon)
            price_sim = np.zeros(horizon)
            price_sim[0] = self.prices.iloc[-1]

            phi_draw = posterior["phi"].values.flatten()[draw_idx]
            sigma_eta_draw = posterior["sigma_eta"].values.flatten()[draw_idx]
            mu_draw = posterior["mu"].values.flatten()[draw_idx]
            #mu_h_draw = posterior["mu_h"].values.flatten()[draw_idx]
            nu_draw = posterior["nu"].values.flatten()[draw_idx]
    
            for n in range(1, horizon):
                h_sim[n] = phi_draw * h_sim[n-1] + sigma_eta_draw * np.random.randn()
                h_sim[n] = np.clip(h_sim[n], -10, 10) #for vol. stocks
                #h_sim[n] = mu_h_draw + phi_draw * (h_sim[n-1] - mu_h_draw) + sigma_eta_draw * np.random.randn() #centered at mu_draw rather than 0, more fair
                z = t.rvs(df=nu_draw)
                z = np.clip(z, -10, 10) #for vol. stocks
                z = z / np.sqrt(nu_draw / (nu_draw - 2)) #scales z val according to prev. scales
                r_sim[n] = mu_draw + np.exp(h_sim[n] / 2) * z #altered random (Gaussian dist) to match T
                price_sim[n] = price_sim[n-1] * np.exp(r_sim[n] / 100)
    
            #plt.plot(price_sim)
            all_paths.append(price_sim.copy())

        sim_array = np.column_stack(all_paths)   # Convert to array

        return pd.DataFrame(sim_array)   # Convert to DataFrame :p
        #pass
    
    def simulate_garch(self, n_paths=500, horizon=252):
        # Forward recursion
        #mc sim

        params = self.garch_res.params

        mu = params['mu']
        omega = params['omega']
        alpha = params['alpha[1]']
        beta = params['beta[1]']
         # Cap alpha + beta to avoid near-IGARCH
        if alpha + beta > 0.97:
            scale = 0.97 / (alpha + beta)
            alpha *= scale
            beta *= scale
        nu = params['nu']  # degrees of freedom
        last_vol = np.asarray(self.garch_res.conditional_volatility)[-1]
        last_resid = np.asarray(self.garch_res.resid)[-1]

        initial_price = self.prices.iloc[-1]  # Initial stock price
        states = ["Down", "Stagnant", "Up"]

        # Initial state probabilities
        initial_state = self.Markov_states.iloc[-1]
        initial_state_probabilities = [1 if state == initial_state else 0 for state in states]

        # Simulating the paths
        simulated_paths = []
        for sim in range(n_paths):
            simulated_returns = []
            current_state = np.random.choice(states, p=initial_state_probabilities)
            sigma_t = last_vol
            r_prev = last_resid
    
            for day in range(horizon):
                #Simulate return based on the current state using Student's t-distribution
                # Update conditional variance
                sigma2_t = omega + alpha * (r_prev ** 2) + beta * (sigma_t ** 2)
                sigma_t = np.sqrt(sigma2_t)

                # Draw Student-t innovation
                #z_t = np.random.standard_t(df=nu)
                #z_t = np.clip(z_t, -10, 10)  # ±10σ tail cap
                z_t = np.random.standard_t(df=nu) * np.sqrt((nu - 2) / nu) #shocks standardized s.t. they have unit variance (less severe results)

                # Optional regime mean shift
                # regime_shift = {
                #     "Up": 0.05,
                #     "Down": -0.05,
                #     "Stagnant": 0.0
                # }

                # Generate return
                #daily_return = mu + sigma_t * z_t
                #daily_return = mu + regime_shift[current_state] + sigma_t * z_t
                innovation = sigma_t * z_t
                daily_return = mu + innovation
                r_prev = innovation

                # Store residual for next iteration
                r_prev = daily_return - mu

                # df_s, mu_s, sigma_s = state_params[current_state]
                # daily_return = t.rvs(df_s, loc=mu_s, scale=sigma_s)
        
                simulated_returns.append(daily_return / 100) #mis-scaled? needs to be just daily return...? i think no...
                # Transition to the next state based on the transition matrix
                current_state = np.random.choice(states, p=self.transition_matrix.loc[current_state, states].values)

    
            # Cumulative price based on returns
            simulated_prices = initial_price * np.exp(np.cumsum(simulated_returns))
            simulated_paths.append(simulated_prices)

        # Convert the list of paths to a DataFrame
        return pd.DataFrame(simulated_paths).T #simulated_paths_df

        #pass
    
    def simulate_empirical(self, n_paths=500, horizon=252):
        # Bootstrap historical returns
        initial_price = self.prices.iloc[-1]
        states = ["Down", "Stagnant", "Up"]

        # Pre-group returns by state
        state_returns = {
            state: self.returns[self.Markov_states == state].values
            for state in states
        }

        simulated_paths = []

        for _ in range(n_paths):
            simulated_returns = []

            # Start from last observed state
            current_state = self.Markov_states.iloc[-1]

            for _ in range(horizon):
                # Sample return conditioned on state
                state_pool = state_returns[current_state]

                # Safety: fallback if empty
                if len(state_pool) == 0:
                    sampled_return = np.random.choice(self.returns.values)
                else:
                    sampled_return = np.random.choice(state_pool)

                simulated_returns.append(sampled_return)

                # Transition to next state
                probs = self.transition_matrix.loc[current_state, states].values
                current_state = np.random.choice(states, p=probs)

            price_path = initial_price * np.cumprod(1 + np.array(simulated_returns))
            simulated_paths.append(price_path)

        simulated_paths_df = pd.DataFrame(simulated_paths).T

        return simulated_paths_df
        #pass
    
    # Metrics
    def compute_metrics(self, simulated_paths_df):
        # Mean, std, kurtosis, tail percentiles, ACF
        simulated_returns = simulated_paths_df.pct_change().dropna()
        sim_returns_array = simulated_returns.values.T
        #sim_returns_flat = simulated_returns.values.reshape(-1, simulated_returns.values.shape[-1])

        # Path-level stats
        sim_means = sim_returns_array.mean(axis=1)
        sim_stds = sim_returns_array.std(axis=1)
        flattened_sim_kurtosis = pd.Series(sim_returns_array.flatten()).kurtosis()
        path_sim_kurtosis = np.mean([
            pd.Series(path).kurtosis() for path in sim_returns_array
        ])
        #sim_kurtosis = [pd.Series(sim).kurtosis() for sim in sim_returns_flat]

        # Observed stats
        obs_mean = self.returns.mean()
        obs_std = self.returns.std()
        obs_kurt = pd.Series(self.returns).kurtosis()

        # Tail Levels (VaR)
        levels = [1, 5] # 1% and 5% tail risk
        obs_var = np.percentile(self.returns, levels)
        sim_var = np.percentile(sim_returns_array, levels)

        # --- Expected Shortfall (ES) Calculation ---
        # Mean of returns that are worse than the VaR threshold
        obs_es = {
            f"ES_{lvl}": self.returns[self.returns <= obs_var[i]].mean() 
            for i, lvl in enumerate(levels)
        }
        sim_es = {
            f"ES_{lvl}": sim_returns_array[sim_returns_array <= sim_var[i]].mean() 
            for i, lvl in enumerate(levels)
        }

        # ACF
        obs_acf = acf(self.returns**2, nlags=20)
        sim_acfs = [acf(sim**2, nlags=20) for sim in sim_returns_array]
        mean_sim_acf = np.mean(sim_acfs, axis=0)

        # Tail percentiles
        percentiles = [1, 5, 95, 99]
        obs_pct = np.percentile(self.returns, percentiles)
        #sim_pct = np.percentile(sim_returns_array, percentiles, axis=1).mean(axis=1)
        sim_pct = np.percentile(sim_returns_array.flatten(), percentiles)

        return {
            "mean": {"observed": obs_mean, "simulated": sim_means.mean()},
            "std": {"observed": obs_std, "simulated": sim_stds.mean()},
            "flattened kurtosis": {"observed": obs_kurt, "simulated": np.mean(flattened_sim_kurtosis)},
            "path kurtosis": {"observed": obs_kurt, "simulated": np.mean(path_sim_kurtosis)},
            "tail_risk": {
                "levels": levels,
                "VaR_obs": obs_var,
                "VaR_sim": sim_var,
                "ES_obs": obs_es,
                "ES_sim": sim_es
            },
            "distributions": {
                "sim_means": sim_means,
                "sim_stds": sim_stds
            },
            "percentiles": {
               "levels": percentiles,
                "observed": obs_pct,
                "simulated": sim_pct
            },
            "acf": {
                "observed": obs_acf,
                "simulated": mean_sim_acf
            }
        }
    
    # Visualization
    def plot_paths(self, simulated_paths_df, title="Monte Carlo Sim", save_path=None):
        # Percentile bands + optional overlay of historical path
        plt.figure(figsize=(10, 6))
        plt.plot(simulated_paths_df, color="lightblue", alpha=0.1)

        # Add 5% and 95% percentile lines
        percentiles_5 = simulated_paths_df.quantile(0.05, axis=1)
        percentiles_95 = simulated_paths_df.quantile(0.95, axis=1)

        plt.plot(percentiles_5, color="red", linestyle="--", label="5th Percentile")
        plt.plot(percentiles_95, color="green", linestyle="--", label="95th Percentile")

        # Add labels and title
        #ticker = "PREIX" #demo ticker remember
        plt.title(title)
        plt.xlabel("Days")
        plt.ylabel("Price")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        # plt.legend()
        # plt.show()
        #pass
    
    def plot_acf(self, metrics, title="ACF Comparison", save_path=None):
        obs_acf = metrics["acf"]["observed"]
        sim_acf = metrics["acf"]["simulated"]

        plt.figure()
        plt.plot(obs_acf, label="Observed")
        plt.plot(sim_acf, label="Simulated Mean")
        plt.legend()
        plt.title(title)
        plt.xlabel("Lag")
        plt.ylabel("ACF")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        #plt.show()
    
    def plot_mean_std_hist(self, metrics, title="", save_path=None):
        sim_means = metrics["distributions"]["sim_means"]
        sim_stds = metrics["distributions"]["sim_stds"]
        sim_means = sim_means[np.isfinite(sim_means)]
        sim_stds = sim_stds[np.isfinite(sim_stds)]

        if len(sim_means) == 0:
            print("All simulated means invalid — skipping plot")
            return

        obs_mean = metrics["mean"]["observed"]
        obs_std = metrics["std"]["observed"]

        plt.figure(figsize=(12, 5))

        # Mean histogram
        plt.subplot(1, 2, 1)
        plt.hist(sim_means, bins=30, alpha=0.7)
        plt.axvline(obs_mean, linestyle='--', label='Observed Mean')
        plt.title(f"{title} - Mean Distribution")
        plt.legend()

        # Std histogram
        plt.subplot(1, 2, 2)
        plt.hist(sim_stds, bins=30, alpha=0.7)
        plt.axvline(obs_std, linestyle='--', label='Observed Std')
        plt.title(f"{title} - Std Distribution")
        plt.legend()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def posterior_predictive_checks_hmc(self, test_returns, n_particles=2000):
        """
        Run sequential posterior predictive checks using particles from HMC posterior.
        Returns coverage, interval widths, PIT, and log predictive likelihoods.
        """
        posterior = self.HMC_trace.posterior

        # Flatten chains and select particles
        phi_draws = posterior["phi"].values.flatten()
        sigma_eta_draws = posterior["sigma_eta"].values.flatten()
        mu_draws = posterior["mu"].values.flatten()
        #mu_h_draws = posterior["mu_h"].values.flatten()
        nu_draws = posterior["nu"].values.flatten()
        h_current = posterior["h"].values[:, :, -1].flatten()  # last latent h

        idx = np.random.choice(len(phi_draws), size=n_particles, replace=True)
        phi = phi_draws[idx]
        sigma_eta = sigma_eta_draws[idx]
        mu = mu_draws[idx]
        #mu_h = mu_h_draws[idx]
        nu = nu_draws[idx]
        h_particles = h_current[idx]

        coverage_count = 0
        lower_bounds = []
        upper_bounds = []
        log_likelihoods = []
        pit_values = []

        for actual_return in test_returns:
            # --- PREDICT STEP ---
            noise = np.random.randn(n_particles)
            #h_particles = mu_h + phi * (h_particles - mu_h) + sigma_eta * noise
            h_particles = phi * h_particles + sigma_eta * noise #recall, not centered at mu
            h_particles = np.clip(h_particles, -10, 5) #for vol. stocks

            z = t.rvs(df=nu, size=n_particles)
            z = z / np.sqrt(nu / (nu - 2))  # normalize variance
            r_particles = mu + np.exp(h_particles / 2) * z

            # PIT
            pit = np.mean(r_particles <= actual_return)
            pit_values.append(pit)

            # 95% interval
            lower = np.percentile(r_particles, 2.5)
            upper = np.percentile(r_particles, 97.5)
            lower_bounds.append(lower)
            upper_bounds.append(upper)

            if lower <= actual_return <= upper:
                coverage_count += 1

            # Likelihood & log predictive likelihood
            scale = np.exp(h_particles / 2)
            likelihoods = t.pdf(actual_return, df=nu, loc=mu, scale=scale)
            likelihoods += 1e-12  # avoid zeros
            log_likelihoods.append(np.log(np.mean(likelihoods)))

            # Resample particles based on likelihood
            weights = likelihoods / np.sum(likelihoods)
            resample_idx = np.random.choice(np.arange(n_particles), size=n_particles, p=weights)
            h_particles = h_particles[resample_idx]
            phi = phi[resample_idx]
            sigma_eta = sigma_eta[resample_idx]
            mu = mu[resample_idx]
            nu = nu[resample_idx]

        return {
            "coverage": coverage_count / len(test_returns),
            "avg_interval_width": np.mean(np.array(upper_bounds) - np.array(lower_bounds)),
            "log_likelihood": np.mean(log_likelihoods),
            "pit_mean": np.mean(pit_values),
            "pit_var": np.var(pit_values),
            "pit_values": pit_values,
            "intervals": {"lower": lower_bounds, "upper": upper_bounds}
        }

    def posterior_predictive_checks_garch(self, test_returns, n_sim=2000):
        #sequential posterior predictive checks using fitted GARCH model.
        #returns = coverage, interval widths, PIT, and log predictive likelihoods.
        params = self.garch_res.params
        mu = params['mu']
        omega = params['omega']
        alpha = params['alpha[1]']
        beta = params['beta[1]']
        nu = params['nu']

        sigma_t = self.garch_res.conditional_volatility[-1]
        r_prev = self.garch_res.resid[-1]

        coverage_count = 0
        lower_bounds = []
        upper_bounds = []
        log_likelihoods = []
        pit_values = []

        for actual_return in test_returns:
            sigma2_t = omega + alpha * (r_prev ** 2) + beta * (sigma_t ** 2)
            sigma_t = np.sqrt(sigma2_t)

            # predictive distribution
            z = np.random.standard_t(df=nu, size=n_sim)
            z = z / np.sqrt(nu / (nu - 2))  # normalize variance
            r_draws = mu + sigma_t * z

            lower = np.percentile(r_draws, 2.5)
            upper = np.percentile(r_draws, 97.5)
            lower_bounds.append(lower)
            upper_bounds.append(upper)

            if lower <= actual_return <= upper:
                coverage_count += 1

            likelihood = t.pdf(actual_return, df=nu, loc=mu, scale=sigma_t)
            log_likelihoods.append(np.log(likelihood + 1e-12))

            pit = np.mean(r_draws <= actual_return)
            pit_values.append(pit)

            # update residual for next step
            r_prev = actual_return - mu

        return {
            "coverage": coverage_count / len(test_returns),
            "avg_interval_width": np.mean(np.array(upper_bounds) - np.array(lower_bounds)),
            "log_likelihood": np.mean(log_likelihoods),
            "pit_mean": np.mean(pit_values),
            "pit_var": np.var(pit_values),
            "pit_values": pit_values,
            "intervals": {"lower": lower_bounds, "upper": upper_bounds}
        }
    
    def plot_pit_hist(self, pit_values, title="PIT Histogram", save_path=None):
        plt.figure(figsize=(6, 4))
        plt.hist(pit_values, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        plt.title(title)
        plt.xlabel("PIT")
        plt.ylabel("Frequency")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_interval_coverage(self, lower_bounds, upper_bounds, actual_returns, title="Predictive Intervals", save_path=None):
        #predictive intervals overlayed w/ actual returns
        plt.figure(figsize=(10, 4))
        plt.fill_between(
            np.arange(len(actual_returns)),
            lower_bounds,
            upper_bounds,
            color="lightgray",
            alpha=0.5,
            label="95% Interval"
        )
        plt.plot(actual_returns, color="red", linewidth=1, label="Actual Returns")
        plt.title(title)
        plt.xlabel("Day")
        plt.ylabel("Return")
        plt.legend()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()