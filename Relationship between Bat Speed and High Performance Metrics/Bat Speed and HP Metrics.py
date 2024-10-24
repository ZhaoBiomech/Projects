# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm
import scipy.stats as sp

# Get data
hp = pd.read_csv(
    "C:\\Users\\Willi\\OneDrive\\Desktop\\Data Science Learning\\Baseball Data Project\\HP Data\\hp_obp.csv"
)
print(hp.columns.tolist())

# Remove all entries that do not include hitting information
hp_hit = hp.dropna(subset=["bat_speed_mph"])

# Rename columns for formatting and clean data
hp_hit.columns = hp_hit.columns.str.replace(r"[\[\]]", "", regex=True)
hp_hit = hp_hit.rename(
    columns={"jump_height_(imp-mom)_cm_mean_cmj": "jump_height_cmj"}
)
hp_hit = hp_hit.rename(
    columns={"jump_height_(imp-mom)_cm_mean_sj": "jump_height_sj"}
)
hp_hit = hp_hit.dropna(
    subset=[
        "jump_height_cmj",
        "body_weight_lbs",
        "peak_power_w_mean_cmj",
        "concentric_peak_force_n_mean_cmj",
        "peak_power_w_mean_sj",
        "jump_height_sj",
        "peak_vertical_force_n_max_imtp",
    ]
)
# Plot the univariate distribution of batspeed split by playing level
sns.displot(
    hp_hit, x="bat_speed_mph", hue="playing_level", kind="kde", fill="True"
)

# Plot the bivariate distribution of bat speed + body weight split by playing level
sns.displot(
    hp_hit,
    x="body_weight_lbs",
    y="bat_speed_mph",
    hue="playing_level",
    kind="kde",
    fill="True",
)

# Retrieve variables of interest
hp_vars = [
    "body_weight_lbs",
    "peak_power_w_mean_cmj",
    "concentric_peak_force_n_mean_cmj",
    "peak_power_w_mean_sj",
    "jump_height_cmj",
    "jump_height_sj",
    "peak_vertical_force_n_max_imtp",
]


def customize_axes_and_save(x, ax, slope, intercept, r_squared):
    """
    Function to help customize graph aesthetic (x labels, title, etc.)
    """
    plt.title(f"Bat Speed (mph) and {var}")
    plt.ylabel("Bat Speed (mph)")
    plt.xlabel(var)

    # Display regression equation and R-squared on the plot
    ax.text(
        0.05,
        0.95,
        f"y = {intercept:.2f} + {slope:.2f}x\nR² = {r_squared:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
    )

    plt.savefig(f"{var}.png", dpi=400, bbox_inches="tight")


# Plot the relationship between variables of interest and bat speed
for x in hp_vars:
    var = x.replace("_", "").title()
    y = sns.lmplot(
        data=hp_hit,
        x=x,
        y="bat_speed_mph",
        height=4,
        aspect=2,
        scatter_kws={"color": "black", "edgecolor": "grey"},
        line_kws={"color": "black"},
    )
    ax = y.ax

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = sp.linregress(
        hp_hit[x], hp_hit["bat_speed_mph"]
    )

    # Calculate R-squared
    r_squared = r_value**2

    customize_axes_and_save(x, ax, slope, intercept, r_squared)

# OLS regression to evaluate variables of interest
reg_hp = sm.ols(
    formula="bat_speed_mph ~ body_weight_lbs + peak_power_w_mean_cmj + concentric_peak_force_n_mean_cmj + peak_power_w_mean_sj + jump_height_cmj + jump_height_sj + peak_vertical_force_n_max_imtp",
    data=hp_hit,
).fit()
print(reg_hp.summary())

# Plot the relationship between variables of interest and bat speed split by playing level
for x in hp_vars:
    var = x.replace("_", "").title()
    y = sns.lmplot(
        data=hp_hit,
        x=x,
        y="bat_speed_mph",
        hue="playing_level",
        height=4,
        aspect=2,
    )
    plt.ylim(0, None)
