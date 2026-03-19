# ============================================================
# A/B Testing Dashboard — Landing Page Experiment
# Run with: python dashboard.py
# Then open: http://127.0.0.1:8050 in your browser
# ============================================================

# ── Core imports ────────────────────────────────────────────
import numpy as np                                         # Numerical operations and Bayesian simulation
import pandas as pd                                        # Data loading and manipulation
from statsmodels.stats.proportion import (
    proportion_effectsize,
    proportions_ztest,
    confint_proportions_2indep,
)                                                          # Frequentist testing utilities
from statsmodels.stats.power import NormalIndPower         # Sample size planning
import plotly.graph_objects as go                          # Custom Plotly charts
import plotly.express as px                                # Quick Plotly charts
from dash import Dash, html, dcc, dash_table               # Dash core components
import dash_bootstrap_components as dbc                    # Bootstrap layout system


# ── 1. LOAD & CLEAN DATA ────────────────────────────────────
df_raw = pd.read_csv("data/ab_data.csv")                        # Load the raw experiment log

# Remove exact duplicate rows
df = df_raw.drop_duplicates().copy()                       # Drop fully identical rows

# Remove mismatched assignment / exposure rows
mismatch_mask = (
    ((df["group"] == "treatment") & (df["landing_page"] != "new_page")) |
    ((df["group"] == "control")   & (df["landing_page"] != "old_page"))
)
df = df.loc[~mismatch_mask].copy()                         # Keep only clean assignment rows

# Keep one row per user
df = df.drop_duplicates(subset="user_id", keep="first").copy()

# Parse timestamps
df["timestamp"] = pd.to_datetime(df["timestamp"])          # Convert string to datetime


# ── 2. CORE METRICS ─────────────────────────────────────────
control_conversions   = int(df.loc[df["group"] == "control",   "converted"].sum())
control_users         = int(df.loc[df["group"] == "control",   "user_id"].nunique())
treatment_conversions = int(df.loc[df["group"] == "treatment", "converted"].sum())
treatment_users       = int(df.loc[df["group"] == "treatment", "user_id"].nunique())

control_rate   = control_conversions   / control_users    # Control conversion rate
treatment_rate = treatment_conversions / treatment_users  # Treatment conversion rate
observed_lift  = treatment_rate - control_rate            # Absolute lift (negative = worse)
relative_lift  = observed_lift  / control_rate            # Relative lift vs control

# Frequentist z-test
z_stat, p_value = proportions_ztest(
    [treatment_conversions, control_conversions],
    [treatment_users, control_users],
    alternative="larger"
)

# 95% Confidence interval
ci_low, ci_high = confint_proportions_2indep(
    count1=treatment_conversions, nobs1=treatment_users,
    count2=control_conversions,   nobs2=control_users,
    compare="diff", alpha=0.05, method="wald"
)

# Bayesian posterior simulation — same seed and draw count as notebook
np.random.seed(42)
posterior_draws = 200_000                                  # Matches notebook exactly

control_posterior   = np.random.beta(
    1 + control_conversions,
    1 + control_users   - control_conversions,
    posterior_draws
)
treatment_posterior = np.random.beta(
    1 + treatment_conversions,
    1 + treatment_users - treatment_conversions,
    posterior_draws
)

prob_treatment_better          = float((treatment_posterior > control_posterior).mean())
# Use same variable names as notebook
expected_loss_choose_treatment = float(np.maximum(control_posterior - treatment_posterior, 0).mean())
expected_loss_choose_control   = float(np.maximum(treatment_posterior - control_posterior, 0).mean())
# Alias for dashboard display
expected_loss_launch           = expected_loss_choose_treatment
expected_loss_keep             = expected_loss_choose_control
loss_ratio                     = expected_loss_launch / expected_loss_keep  # How many × riskier to launch

# Experiment dates
experiment_start = df["timestamp"].min().strftime("%b %d, %Y")
experiment_end   = df["timestamp"].max().strftime("%b %d, %Y")
total_users      = df["user_id"].nunique()

# Sample size planning
effect_size_h  = proportion_effectsize(0.12, 0.12 * 1.02)
pa             = NormalIndPower()
req_per_group  = int(float(pa.solve_power(
    effect_size=abs(float(effect_size_h)), power=0.80,
    alpha=0.05, ratio=1.0, alternative="larger"
))) + 1
met_sample     = min(control_users, treatment_users) >= req_per_group


# ── 3. BUILD FIGURES ────────────────────────────────────────

PLOT_BG    = "rgba(0,0,0,0)"                               # Transparent chart backgrounds
GRID_COLOR = "rgba(255,255,255,0.06)"                      # Subtle grid lines
FONT_COLOR = "#e2e8f0"                                     # Light text on dark background
C_CONTROL  = "#60a5fa"                                     # Blue for control
C_TREAT    = "#f87171"                                     # Red for treatment

COMMON_LAYOUT = dict(
    paper_bgcolor=PLOT_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(family="DM Mono, monospace", color=FONT_COLOR, size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
)


# ── Fig 1: Daily conversion rate over time ──────────────────
df["event_date"] = df["timestamp"].dt.date
daily = (
    df.groupby(["event_date", "group"])
      .agg(users=("user_id", "nunique"), conversions=("converted", "sum"))
      .reset_index()
)
daily["rate"] = daily["conversions"] / daily["users"]

fig_daily = go.Figure()
for grp, color, dash in [("control", C_CONTROL, "solid"), ("treatment", C_TREAT, "dot")]:
    d = daily[daily["group"] == grp]
    fig_daily.add_trace(go.Scatter(
        x=d["event_date"], y=d["rate"],
        name=grp.capitalize(), mode="lines+markers",
        line=dict(color=color, width=2, dash=dash),
        marker=dict(size=5),
    ))
fig_daily.update_layout(
    title="Daily Conversion Rate Over Time",
    yaxis_tickformat=".2%",
    **COMMON_LAYOUT
)


# ── Fig 2: Bayesian posteriors ───────────────────────────────
ctrl_density,  ctrl_bins  = np.histogram(control_posterior,   bins=300, density=True)
trt_density,   trt_bins   = np.histogram(treatment_posterior, bins=300, density=True)
ctrl_mid = (ctrl_bins[:-1] + ctrl_bins[1:]) / 2
trt_mid  = (trt_bins[:-1]  + trt_bins[1:])  / 2

fig_bayes = go.Figure()
fig_bayes.add_trace(go.Scatter(
    x=ctrl_mid, y=ctrl_density, name="Control",
    mode="lines", line=dict(color=C_CONTROL, width=2),
    fill="tozeroy", fillcolor="rgba(96,165,250,0.25)",
))
fig_bayes.add_trace(go.Scatter(
    x=trt_mid, y=trt_density, name="Treatment",
    mode="lines", line=dict(color=C_TREAT, width=2),
    fill="tozeroy", fillcolor="rgba(248,113,113,0.25)",
))
fig_bayes.update_layout(
    title="Bayesian Posterior Distributions",
    xaxis_tickformat=".2%",
    **COMMON_LAYOUT
)


# ── Fig 3: Confidence interval ───────────────────────────────
fig_ci = go.Figure()
fig_ci.add_trace(go.Scatter(
    x=[observed_lift], y=["Treatment − Control"],
    mode="markers", marker=dict(size=14, color=C_CONTROL, symbol="diamond"),
    error_x=dict(
        type="data", symmetric=False,
        array=[ci_high - observed_lift],
        arrayminus=[observed_lift - ci_low],
        color=C_CONTROL, thickness=2, width=8,
    ),
    name="Lift estimate",
))
fig_ci.add_vline(x=0, line_dash="dash", line_color="#f59e0b", line_width=2)
fig_ci.update_layout(
    title="95% Confidence Interval for Lift",
    xaxis_tickformat=".3%",
    xaxis_title="Lift (Treatment − Control)",
    **COMMON_LAYOUT
)


# ── Fig 4: Day-of-week segmentation ─────────────────────────
df["day_of_week"] = df["timestamp"].dt.day_name()
weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
seg = (
    df.groupby(["day_of_week", "group"])
      .agg(users=("user_id","nunique"), conversions=("converted","sum"))
      .reset_index()
)
seg["rate"] = seg["conversions"] / seg["users"]
seg["day_of_week"] = pd.Categorical(seg["day_of_week"], categories=weekday_order, ordered=True)
seg = seg.sort_values("day_of_week")

fig_seg = go.Figure()
for grp, color, dash in [("control", C_CONTROL, "solid"), ("treatment", C_TREAT, "dot")]:
    d = seg[seg["group"] == grp]
    fig_seg.add_trace(go.Scatter(
        x=d["day_of_week"].astype(str), y=d["rate"],
        name=grp.capitalize(), mode="lines+markers",
        line=dict(color=color, width=2, dash=dash),
        marker=dict(size=7),
    ))
fig_seg.update_layout(
    title="Conversion Rate by Day of Week",
    yaxis_tickformat=".2%",
    **COMMON_LAYOUT
)


# ── Fig 5: Expected loss comparison bar ─────────────────────
fig_loss = go.Figure()
fig_loss.add_trace(go.Bar(
    x=["Launch Treatment", "Keep Control"],
    y=[expected_loss_launch, expected_loss_keep],
    marker_color=[C_TREAT, C_CONTROL],
    text=[f"{expected_loss_launch:.4%}", f"{expected_loss_keep:.4%}"],
    textposition="outside",
    textfont=dict(color=FONT_COLOR),
))
fig_loss.update_layout(
    title="Expected Loss by Decision",
    yaxis_tickformat=".4%",
    yaxis_title="Expected Conversion Loss",
    showlegend=False,
    **COMMON_LAYOUT
)


# ── Fig 6: Group size bar ────────────────────────────────────
fig_groups = go.Figure()
fig_groups.add_trace(go.Bar(
    x=["Control", "Treatment"],
    y=[control_users, treatment_users],
    marker_color=[C_CONTROL, C_TREAT],
    text=[f"{control_users:,}", f"{treatment_users:,}"],
    textposition="outside",
    textfont=dict(color=FONT_COLOR),
))
fig_groups.update_layout(
    title="Users per Experiment Group",
    yaxis_title="Number of Users",
    showlegend=False,
    **COMMON_LAYOUT
)


# ── 4. BUSINESS IMPACT TABLE DATA ───────────────────────────
monthly_visitors = 50_000
avg_order_value  = 35.00
impl_cost        = 20_000.00
incr_orders_mo   = monthly_visitors * observed_lift
revenue_mo       = incr_orders_mo * avg_order_value
revenue_yr       = revenue_mo * 12
break_even       = impl_cost / revenue_mo if revenue_mo > 0 else float("inf")

impact_rows = [
    {"Metric": "Control Conversion Rate",          "Value": f"{control_rate:.4%}",    "Status": "neutral"},
    {"Metric": "Treatment Conversion Rate",         "Value": f"{treatment_rate:.4%}",  "Status": "bad"},
    {"Metric": "Absolute Lift",                     "Value": f"{observed_lift:.4%}",   "Status": "bad"},
    {"Metric": "Relative Lift",                     "Value": f"{relative_lift:.2%}",   "Status": "bad"},
    {"Metric": "Z-Statistic",                       "Value": f"{z_stat:.4f}",          "Status": "bad"},
    {"Metric": "One-Sided p-Value",                 "Value": f"{p_value:.4f}",         "Status": "bad"},
    {"Metric": "95% CI Lower Bound",                "Value": f"{ci_low:.4%}",          "Status": "bad"},
    {"Metric": "95% CI Upper Bound",                "Value": f"{ci_high:.4%}",         "Status": "neutral"},
    {"Metric": "Bayesian Win Probability",          "Value": f"{prob_treatment_better:.2%}", "Status": "bad"},
    {"Metric": "Expected Loss — Launch Treatment",  "Value": f"{expected_loss_launch:.4%}",  "Status": "bad"},
    {"Metric": "Expected Loss — Keep Control",      "Value": f"{expected_loss_keep:.4%}",    "Status": "neutral"},
    {"Metric": "Loss Ratio (Launch vs Keep)",       "Value": f"{loss_ratio:.1f}×",     "Status": "bad"},
    {"Metric": "Projected Monthly Revenue Impact",  "Value": f"${revenue_mo:,.2f}",    "Status": "bad"},
    {"Metric": "Projected Annual Revenue Impact",   "Value": f"${revenue_yr:,.2f}",    "Status": "bad"},
    {"Metric": "Implementation Cost",               "Value": f"${impl_cost:,.2f}",     "Status": "neutral"},
    {"Metric": "Break-Even Period",                 "Value": "Not reached" if np.isinf(break_even) else f"{break_even:.1f} months", "Status": "bad"},
]
impact_df = pd.DataFrame(impact_rows)


# ── 5. HELPER: KPI CARD ─────────────────────────────────────
def kpi_card(label, value, color="#e2e8f0", subtitle=None):
    """Return a styled Bootstrap card for a single KPI metric."""
    return dbc.Card(
        dbc.CardBody([
            html.P(label, style={
                "fontSize": "0.7rem", "textTransform": "uppercase",
                "letterSpacing": "0.12em", "color": "#94a3b8",
                "marginBottom": "6px", "fontFamily": "DM Mono, monospace",
            }),
            html.P(value, style={
                "fontSize": "1.6rem", "fontWeight": "700",
                "color": color, "margin": "0",
                "fontFamily": "DM Mono, monospace", "lineHeight": "1",
            }),
            html.P(subtitle or "", style={
                "fontSize": "0.65rem", "color": "#64748b",
                "marginTop": "4px", "marginBottom": "0",
                "fontFamily": "DM Mono, monospace",
            }),
        ], style={"padding": "16px 20px"}),
        style={
            "background": "rgba(255,255,255,0.04)",
            "border": "1px solid rgba(255,255,255,0.08)",
            "borderRadius": "12px",
            "backdropFilter": "blur(6px)",
        }
    )


# ── 6. CHART CARD WRAPPER ────────────────────────────────────
def chart_card(fig, height=340):
    """Wrap a Plotly figure in a styled card."""
    return dbc.Card(
        dbc.CardBody(
            dcc.Graph(figure=fig, config={"displayModeBar": False},
                      style={"height": f"{height}px"}),
            style={"padding": "8px"},
        ),
        style={
            "background": "rgba(255,255,255,0.03)",
            "border": "1px solid rgba(255,255,255,0.07)",
            "borderRadius": "12px",
        }
    )


# ── 7. DASH APP ──────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.SLATE,                                  # Dark Bootstrap base theme
        "https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap",
    ],
    title="A/B Test Dashboard",
)

# ── Verdict badge ────────────────────────────────────────────
verdict_color   = "#ef4444" if prob_treatment_better < 0.5 else "#22c55e"
verdict_text    = "⛔  DO NOT LAUNCH" if prob_treatment_better < 0.5 else "✅  LAUNCH APPROVED"
verdict_subtext = (
    f"Treatment wins probability is only {prob_treatment_better:.1%}. "
    f"Launching is {loss_ratio:.0f}× riskier than keeping the control page. "
    f"Expected loss if launched: {expected_loss_launch:.4%} vs {expected_loss_keep:.4%} if kept."
    if prob_treatment_better < 0.5 else
    f"Treatment wins probability is {prob_treatment_better:.1%}. Evidence supports launch."
)

# ── Layout ───────────────────────────────────────────────────
app.layout = html.Div(
    style={
        "minHeight": "100vh",
        "background": "linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%)",
        "fontFamily": "DM Mono, monospace",
        "padding": "32px 40px",
    },
    children=[

        # ── Header ──────────────────────────────────────────
        html.Div([
            html.Div("A/B TEST RESULTS", style={
                "fontSize": "0.7rem", "letterSpacing": "0.25em",
                "color": "#60a5fa", "marginBottom": "6px",
                "fontFamily": "DM Mono, monospace",
            }),
            html.H1("Landing Page Experiment", style={
                "fontSize": "clamp(1.8rem, 3vw, 2.8rem)",
                "fontWeight": "800", "color": "#f1f5f9",
                "fontFamily": "Syne, sans-serif",
                "margin": "0 0 8px 0", "lineHeight": "1.1",
            }),
            html.P(
                f"Experiment window: {experiment_start} → {experiment_end}  ·  "
                f"Total users: {total_users:,}  ·  "
                f"Sample target met: {'Yes ✓' if met_sample else 'No ✗'}",
                style={"color": "#64748b", "fontSize": "0.78rem",
                       "marginBottom": "0"},
            ),
        ], style={"marginBottom": "28px"}),

        # ── Verdict banner ───────────────────────────────────
        html.Div([
            html.Div(verdict_text, style={
                "fontSize": "1.1rem", "fontWeight": "700",
                "color": verdict_color, "fontFamily": "Syne, sans-serif",
                "marginBottom": "4px",
            }),
            html.Div(verdict_subtext, style={
                "fontSize": "0.78rem", "color": "#cbd5e1",
            }),
        ], style={
            "background": f"rgba({','.join(str(int(verdict_color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))}, 0.08)",
            "border": f"1px solid {verdict_color}33",
            "borderLeft": f"4px solid {verdict_color}",
            "borderRadius": "10px",
            "padding": "16px 22px",
            "marginBottom": "28px",
        }),

        # ── KPI Row ──────────────────────────────────────────
        dbc.Row([
            dbc.Col(kpi_card("Control Rate",   f"{control_rate:.3%}",   "#60a5fa"), md=2),
            dbc.Col(kpi_card("Treatment Rate",  f"{treatment_rate:.3%}", "#f87171"), md=2),
            dbc.Col(kpi_card("Absolute Lift",   f"{observed_lift:.3%}",  "#f87171",
                             subtitle="Treatment underperforms"), md=2),
            dbc.Col(kpi_card("Bayesian Win %",  f"{prob_treatment_better:.1%}", "#f87171",
                             subtitle="Need ≥ 95% to launch"), md=2),
            dbc.Col(kpi_card("p-Value",         f"{p_value:.4f}",        "#f87171",
                             subtitle="Need < 0.05"), md=2),
            dbc.Col(kpi_card("Loss Ratio",      f"{loss_ratio:.0f}×",    "#fbbf24",
                             subtitle="Riskier to launch"), md=2),
        ], className="g-3", style={"marginBottom": "24px"}),

        # ── Row 2: Daily trend + Bayesian posteriors ─────────
        dbc.Row([
            dbc.Col(chart_card(fig_daily),  md=7),
            dbc.Col(chart_card(fig_bayes),  md=5),
        ], className="g-3", style={"marginBottom": "24px"}),

        # ── Row 3: CI + Segmentation ─────────────────────────
        dbc.Row([
            dbc.Col(chart_card(fig_ci),   md=5),
            dbc.Col(chart_card(fig_seg),  md=7),
        ], className="g-3", style={"marginBottom": "24px"}),

        # ── Row 4: Expected loss + Group sizes ───────────────
        dbc.Row([
            dbc.Col(chart_card(fig_loss,   height=300), md=6),
            dbc.Col(chart_card(fig_groups, height=300), md=6),
        ], className="g-3", style={"marginBottom": "24px"}),

        # ── Row 5: Business impact table ─────────────────────
        dbc.Card([
            dbc.CardBody([
                html.H5("Business Impact Summary", style={
                    "color": "#e2e8f0", "fontFamily": "Syne, sans-serif",
                    "fontWeight": "700", "marginBottom": "16px",
                    "fontSize": "1rem",
                }),
                dash_table.DataTable(
                    data=impact_df[["Metric", "Value"]].to_dict("records"),
                    columns=[
                        {"name": "Metric", "id": "Metric"},
                        {"name": "Value",  "id": "Value"},
                    ],
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "rgba(255,255,255,0.06)",
                        "color": "#94a3b8",
                        "fontWeight": "500",
                        "fontSize": "0.7rem",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.1em",
                        "border": "none",
                        "fontFamily": "DM Mono, monospace",
                        "padding": "10px 16px",
                    },
                    style_cell={
                        "backgroundColor": "transparent",
                        "color": "#e2e8f0",
                        "fontSize": "0.82rem",
                        "fontFamily": "DM Mono, monospace",
                        "border": "none",
                        "borderBottom": "1px solid rgba(255,255,255,0.05)",
                        "padding": "10px 16px",
                        "textAlign": "left",
                    },
                    style_data_conditional=[
                        # Highlight bad rows in muted red
                        {
                            "if": {"filter_query": f'{{Metric}} = "{row["Metric"]}"'},
                            "color": "#fca5a5",
                        }
                        for row in impact_rows if row["Status"] == "bad"
                    ],
                    page_size=20,
                    sort_action="native",
                ),
            ], style={"padding": "20px 24px"}),
        ], style={
            "background": "rgba(255,255,255,0.03)",
            "border": "1px solid rgba(255,255,255,0.07)",
            "borderRadius": "12px",
            "marginBottom": "24px",
        }),

        # ── Footer ───────────────────────────────────────────
        html.Div(
            f"Analysis based on Udacity A/B Testing Dataset · "
            f"{total_users:,} users · Jan 2017 · "
            f"Frequentist + Bayesian + Segmentation framework",
            style={
                "textAlign": "center", "color": "#334155",
                "fontSize": "0.7rem", "paddingTop": "8px",
                "fontFamily": "DM Mono, monospace",
            }
        ),
    ]
)


# ── 8. RUN ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n✓  Dashboard ready → http://127.0.0.1:8050\n")
    app.run(debug=False, port=8050)                        # Launch the Dash server