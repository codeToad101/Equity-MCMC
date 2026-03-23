import yfinance as yf
from dynamicMC import IndexSimulator
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf_report(results, image_paths, filename="report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Monte Carlo Simulation Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    for ticker, model_data in results.items():
        elements.append(Paragraph(f"{ticker}", styles["Heading1"]))
        elements.append(Spacer(1, 10))

        # =========================
        # SIMULATION SECTION
        # =========================
        elements.append(Paragraph("Simulation Results", styles["Heading2"]))

        for model_name, metrics in model_data["simulation"].items():
            elements.append(Paragraph(f"{model_name}", styles["Heading3"]))

            mean_obs = metrics["mean"]["observed"]
            mean_sim = metrics["mean"]["simulated"]

            std_obs = metrics["std"]["observed"]
            std_sim = metrics["std"]["simulated"]

            kurt_obs = metrics["kurtosis"]["observed"]
            kurt_sim = metrics["kurtosis"]["simulated"]

            elements.append(Paragraph(
                f"Mean (obs vs sim): {mean_obs:.6f} vs {mean_sim:.6f}",
                styles["Normal"]
            ))
            elements.append(Paragraph(
                f"Std Dev (obs vs sim): {std_obs:.6f} vs {std_sim:.6f}",
                styles["Normal"]
            ))
            elements.append(Paragraph(
                f"Kurtosis (obs vs sim): {kurt_obs:.4f} vs {kurt_sim:.4f}",
                styles["Normal"]
            ))

            elements.append(Spacer(1, 6))

            # Images
            key = (ticker, model_name)
            if key in image_paths:
               elements.append(Image(image_paths[key], width=400, height=250))
               elements.append(Spacer(1, 12))

            acf_key = (ticker, f"{model_name} ACF")
            if acf_key in image_paths:
                elements.append(Image(image_paths[acf_key], width=400, height=250))
                elements.append(Spacer(1, 12))

            hist_key = (ticker, f"{model_name} HIST")
            if hist_key in image_paths:
                elements.append(Image(image_paths[hist_key], width=400, height=250))
                elements.append(Spacer(1, 12))

        elements.append(Spacer(1, 12))

        # =========================
        # POSTERIOR SECTION
        # =========================
        elements.append(Paragraph("Posterior Predictive Checks", styles["Heading2"]))

        for model_name, posterior in model_data["posterior"].items():
            elements.append(Paragraph(f"{model_name}", styles["Heading3"]))

            elements.append(Paragraph(
                f"Coverage: {posterior['coverage']:.4f}",
                styles["Normal"]
            ))
            elements.append(Paragraph(
                f"Avg Interval Width: {posterior['avg_interval_width']:.6f}",
                styles["Normal"]
            ))
            elements.append(Paragraph(
                f"Avg Log Likelihood: {posterior['log_likelihood']:.6f}",
                styles["Normal"]
            ))
            elements.append(Paragraph(
                f"PIT Mean / Var: {posterior['pit_mean']:.4f} / {posterior['pit_var']:.4f}",
                styles["Normal"]
            ))

            elements.append(Spacer(1, 6))

            # PIT histogram
            pit_key = (ticker, f"{model_name} PIT")
            if pit_key in image_paths:
                elements.append(Image(image_paths[pit_key], width=400, height=250))
                elements.append(Spacer(1, 12))

            # Interval plot
            interval_key = (ticker, f"{model_name} Interval")
            if interval_key in image_paths:
                elements.append(Image(image_paths[interval_key], width=400, height=250))
                elements.append(Spacer(1, 12))

        elements.append(Spacer(1, 20))

    doc.build(elements)
 

tickers = {
    "PREIX": "T. Rowe S&P 500 Index Fund",
    "SPY": "S&P 500",
    "QQQ": "NASDAQ 100",
    "DIA": "Dow Jones",
    "IWM": "Russell 2000"
}

# data = yf.download("SPY", period="10y")
# print(data.columns)

results = {}
image_paths = {}
posterior = {}

for ticker, name in tickers.items():
    print(f"\n--- Running {name} ({ticker}) ---")

    data = yf.download(ticker, period="10y", auto_adjust=True)
    prices = data.xs(ticker, axis=1, level='Ticker')['Close'].dropna()
    #prices = data["Adj Close"].dropna()

    sim = IndexSimulator(prices)

    # Fit models
    sim.fit_hmc_sv()
    sim.fit_garch()

    # Simulate
    hmc_paths = sim.simulate_hmc()
    garch_paths = sim.simulate_garch()
    emp_paths = sim.simulate_empirical()

    # Metrics
    hmc_metrics = sim.compute_metrics(hmc_paths)
    garch_metrics = sim.compute_metrics(garch_paths)
    emp_metrics = sim.compute_metrics(emp_paths)

    #init posterior checks
    test_returns = sim.log_returns[-252:]  # last 252 days

    hmc_posterior = sim.posterior_predictive_checks_hmc(test_returns)
    garch_posterior = sim.posterior_predictive_checks_garch(test_returns)

    results[ticker] = {
        "simulation": {
            "HMC": hmc_metrics,
            "GARCH": garch_metrics,
            "EMP": emp_metrics
        },
        "posterior": {
            "HMC": hmc_posterior,
            "GARCH": garch_posterior
        }
    }

    # Save plots instead of showing
    hmc_MCsim_path = f"outputs/images/{ticker}_hmc_MCsim.png"
    garch_MCsim_path = f"outputs/images/{ticker}_garch_MCsim.png"
    emp_MCsim_path = f"outputs/images/{ticker}_emp_MCsim.png"

    # Plots (optional)
    sim.plot_paths(hmc_paths, title=f"{ticker} HMC Monte Carlo Paths", save_path=hmc_MCsim_path)
    sim.plot_paths(garch_paths, title=f"{ticker} GARCH Monte Carlo Paths", save_path=garch_MCsim_path)
    sim.plot_paths(emp_paths, title=f"{ticker} EMP Monte Carlo Paths", save_path=emp_MCsim_path)
    image_paths[(ticker, "HMC")] = hmc_MCsim_path
    image_paths[(ticker, "GARCH")] = garch_MCsim_path
    image_paths[(ticker, "EMP")] = emp_MCsim_path

    # Save plots instead of showing
    hmc_acf_path = f"outputs/images/{ticker}_hmc_acf.png"
    garch_acf_path = f"outputs/images/{ticker}_garch_acf.png"
    emp_acf_path = f"outputs/images/{ticker}_emp_acf.png"

    # Plots (optional)
    sim.plot_acf(hmc_metrics, title=f"{ticker} HMC ACF", save_path=hmc_acf_path)
    sim.plot_acf(garch_metrics, title=f"{ticker} GARCH ACF", save_path=garch_acf_path)
    sim.plot_acf(emp_metrics, title=f"{ticker} EMP ACF", save_path=emp_acf_path)
    image_paths[(ticker, "HMC ACF")] = hmc_acf_path
    image_paths[(ticker, "GARCH ACF")] = garch_acf_path
    image_paths[(ticker, "EMP ACF")] = emp_acf_path

    hmc_hist_path = f"outputs/images/{ticker}_hmc_hist.png"
    garch_hist_path = f"outputs/images/{ticker}_garch_hist.png"
    emp_hist_path = f"outputs/images/{ticker}_emp_hist.png"

    sim.plot_mean_std_hist(hmc_metrics, title=f"{ticker} HMC", save_path=hmc_hist_path)
    sim.plot_mean_std_hist(garch_metrics, title=f"{ticker} GARCH", save_path=garch_hist_path)
    sim.plot_mean_std_hist(emp_metrics, title=f"{ticker} EMP", save_path=emp_hist_path)

    image_paths[(ticker, "HMC HIST")] = hmc_hist_path
    image_paths[(ticker, "GARCH HIST")] = garch_hist_path
    image_paths[(ticker, "EMP HIST")] = emp_hist_path

    # Example usage after running posterior checks
    hmc_pit_path = f"outputs/images/{ticker}_hmc_pit.png"
    garch_pit_path = f"outputs/images/{ticker}_garch_pit.png"

    sim.plot_pit_hist(hmc_posterior["pit_values"], title=f"{ticker} HMC PIT", save_path=hmc_pit_path)
    sim.plot_pit_hist(garch_posterior["pit_values"], title=f"{ticker} GARCH PIT", save_path=garch_pit_path)

    hmc_interval_path = f"outputs/images/{ticker}_hmc_intervals.png"
    garch_interval_path = f"outputs/images/{ticker}_garch_intervals.png"

    sim.plot_interval_coverage(
        hmc_posterior["intervals"]["lower"],
        hmc_posterior["intervals"]["upper"],
        test_returns,
        title=f"{ticker} HMC Predictive Intervals",
        save_path=hmc_interval_path
    )

    sim.plot_interval_coverage(
        garch_posterior["intervals"]["lower"],
        garch_posterior["intervals"]["upper"],
        test_returns,
        title=f"{ticker} GARCH Predictive Intervals",
        save_path=garch_interval_path
    )

    # Add to image_paths for PDF
    image_paths[(ticker, "HMC PIT")] = hmc_pit_path
    image_paths[(ticker, "GARCH PIT")] = garch_pit_path
    image_paths[(ticker, "HMC Interval")] = hmc_interval_path
    image_paths[(ticker, "GARCH Interval")] = garch_interval_path

    generate_pdf_report(results, image_paths)

print("\nDone.")