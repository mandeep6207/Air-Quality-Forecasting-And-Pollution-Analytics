"""
eda_visualizations.py
---------------------
Generates and saves all EDA plots to the visuals/ directory.

Plots produced
--------------
1. aqi_distribution.png       – AQI histogram with KDE overlay
2. correlation_heatmap.png    – Pearson correlation heatmap of numeric features
3. top_polluted_cities.png    – Top 15 cities by mean AQI (bar chart)
4. monthly_trends.png         – Monthly average AQI trend (line chart)
5. aqi_bucket_distribution.png – AQI bucket count plot
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from utils import get_logger, save_figure, set_plot_style, project_root, ensure_dir

logger = get_logger(__name__)

ROOT = project_root()
VISUALS_DIR = ROOT / "visuals"
CLEAN_DATA_PATH = ROOT / "data" / "cleaned_air_quality.csv"

POLLUTANT_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene", "AQI",
]

AQI_BUCKET_ORDER = [
    "Good", "Satisfactory", "Moderate",
    "Poor", "Very Poor", "Severe",
]

BUCKET_COLORS = {
    "Good": "#2ecc71",
    "Satisfactory": "#a8e063",
    "Moderate": "#f1c40f",
    "Poor": "#e67e22",
    "Very Poor": "#e74c3c",
    "Severe": "#8e44ad",
}


# ── Individual plot functions ─────────────────────────────────────────────────

def plot_aqi_distribution(df: pd.DataFrame) -> None:
    """Histogram + KDE of AQI values."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    n_bins = 60
    counts, edges, patches = ax.hist(
        df["AQI"], bins=n_bins, color="#3498db", edgecolor="white",
        alpha=0.75, label="Count",
    )

    # Simple KDE overlay using numpy
    aqi_vals = df["AQI"].dropna().values
    kde_x = np.linspace(aqi_vals.min(), aqi_vals.max(), 300)
    bw = 1.06 * aqi_vals.std() * len(aqi_vals) ** (-0.2)
    kde_y = np.array([
        np.mean(np.exp(-0.5 * ((kde_x[i] - aqi_vals) / bw) ** 2) / (bw * np.sqrt(2 * np.pi)))
        for i in range(len(kde_x))
    ])
    ax2 = ax.twinx()
    ax2.plot(kde_x, kde_y, color="#e74c3c", linewidth=2, label="KDE")
    ax2.set_ylabel("Density", color="#e74c3c")
    ax2.tick_params(axis="y", labelcolor="#e74c3c")
    ax2.set_ylim(0)
    ax2.spines["top"].set_visible(False)

    ax.set_xlabel("AQI Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Air Quality Index (AQI)", fontweight="bold", pad=12)

    # Slight x-axis padding for aesthetics
    try:
        x_min, x_max = aqi_vals.min(), aqi_vals.max()
        ax.set_xlim(max(0, x_min - 5), x_max + 5)
    except Exception:
        pass

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    save_figure(fig, VISUALS_DIR / "aqi_distribution.png")
    logger.info("Saved aqi_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Pearson correlation heatmap for pollutant columns + AQI."""
    set_plot_style()
    corr = df[POLLUTANT_COLS].corr()
    n = len(POLLUTANT_COLS)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)
    ax.set_title("Correlation Heatmap – Pollutants & AQI", fontweight="bold", pad=12)

    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            # Use a slightly lower threshold for white text to improve
            # readability on moderately strong correlations.
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(val) > 0.5 else "black")

    fig.tight_layout()
    save_figure(fig, VISUALS_DIR / "correlation_heatmap.png")
    logger.info("Saved correlation_heatmap.png")


def plot_top_polluted_cities(df: pd.DataFrame, top_n: int = 15) -> None:
    """Horizontal bar chart of top-N cities by mean AQI."""
    set_plot_style()
    city_aqi = (
        df.groupby("City")["AQI"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.Reds(np.linspace(0.4, 0.9, top_n))
    bars = ax.barh(city_aqi.index[::-1], city_aqi.values[::-1],
                   color=cmap[::-1], edgecolor="white")

    for bar, val in zip(bars, city_aqi.values[::-1]):
        ax.text(val + 2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=9)

    ax.set_xlabel("Mean AQI")
    ax.set_title(f"Top {top_n} Most Polluted Cities (Mean AQI)", fontweight="bold", pad=12)
    fig.tight_layout()
    save_figure(fig, VISUALS_DIR / "top_polluted_cities.png")
    logger.info("Saved top_polluted_cities.png")


def plot_monthly_trends(df: pd.DataFrame) -> None:
    """Line chart of monthly average AQI across all years."""
    set_plot_style()
    df = df.copy()
    df["Month"] = pd.to_datetime(df["Date"]).dt.month
    monthly = df.groupby("Month")["AQI"].mean()

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly.index, monthly.values, marker="o", linewidth=2.5,
            color="#e74c3c", markersize=7, markerfacecolor="white",
            markeredgewidth=2)
    ax.fill_between(monthly.index, monthly.values, alpha=0.15, color="#e74c3c")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean AQI")
    ax.set_title("Monthly Average AQI Trend", fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    fig.tight_layout()
    save_figure(fig, VISUALS_DIR / "monthly_trends.png")
    logger.info("Saved monthly_trends.png")


def plot_aqi_bucket_distribution(df: pd.DataFrame) -> None:
    """Count plot of AQI bucket categories."""
    set_plot_style()
    present = [b for b in AQI_BUCKET_ORDER if b in df["AQI_Bucket"].values]
    counts = df["AQI_Bucket"].value_counts().reindex(present).fillna(0)
    colors = [BUCKET_COLORS.get(b, "#95a5a6") for b in present]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(present, counts.values, color=colors, edgecolor="white", width=0.6)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 30,
                f"{int(val):,}", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("AQI Bucket")
    ax.set_ylabel("Number of Records")
    ax.set_title("AQI Bucket Distribution", fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    save_figure(fig, VISUALS_DIR / "aqi_bucket_distribution.png")
    logger.info("Saved aqi_bucket_distribution.png")


# ── Master function ───────────────────────────────────────────────────────────

def run_eda(clean_path: Path = CLEAN_DATA_PATH) -> None:
    """Load cleaned data and generate all EDA plots."""
    ensure_dir(VISUALS_DIR)
    df = pd.read_csv(clean_path, parse_dates=["Date"])
    logger.info("Loaded cleaned data: %s", df.shape)

    plot_aqi_distribution(df)
    plot_correlation_heatmap(df)
    plot_top_polluted_cities(df)
    plot_monthly_trends(df)
    plot_aqi_bucket_distribution(df)

    logger.info("All EDA plots saved to %s", VISUALS_DIR)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_eda()
