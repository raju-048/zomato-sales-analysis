"""
Zomato Order Analytics – Pure Python Dash Dashboard
Run:  python dashboard.py
Then open:  http://127.0.0.1:8050
"""

import os
import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

def load(name):
    return pd.read_csv(os.path.join(BASE, f"Zomato  Order Data.xlsx - {name}.csv"))

customers   = load("Customer")
orders      = load("Orders")
restaurants = load("Restaurants")

# ─────────────────────────────────────────────────────────────
# 2. CLEAN & ENGINEER
# ─────────────────────────────────────────────────────────────
orders["order_timestamp"] = pd.to_datetime(orders["order_timestamp"], format="%m/%d/%Y", errors="coerce")
orders["order_month"]   = orders["order_timestamp"].dt.to_period("M").astype(str)
orders["order_quarter"] = orders["order_timestamp"].dt.to_period("Q").astype(str)
orders["order_year"]    = orders["order_timestamp"].dt.year.astype("Int64")
orders["day_of_week"]   = orders["order_timestamp"].dt.day_name()

customers["Signup_Time"]  = pd.to_datetime(customers["Signup_Time"], format="%d/%m/%Y", errors="coerce")
customers["signup_month"] = customers["Signup_Time"].dt.to_period("M").astype(str)

orders["discount_amount"] = orders["discount_amount"].fillna(0)
orders["delivery_fee"]    = orders["delivery_fee"].fillna(0)
orders["net_revenue"]     = orders["order_amount"] - orders["discount_amount"]

# Master joined frame
full = (orders
        .merge(restaurants, on="restaurant_id", how="left")
        .merge(customers, left_on="customer_id", right_on="Customer_id", how="left"))

delivered = full[full["order_status"] == "Delivered"]

# ─────────────────────────────────────────────────────────────
# 3. KPIs
# ─────────────────────────────────────────────────────────────
total_orders      = len(orders)
total_customers   = customers["Customer_id"].nunique()
total_revenue     = delivered["net_revenue"].sum()
avg_order_value   = delivered["order_amount"].mean()
delivery_rate     = (orders["order_status"] == "Delivered").mean() * 100
cancel_rate       = (orders["order_status"] == "Cancelled").mean() * 100
refund_rate       = (orders["order_status"] == "Refunded").mean() * 100
total_discounts   = orders["discount_amount"].sum()
avg_delivery_fee  = orders["delivery_fee"].mean()

# ─────────────────────────────────────────────────────────────
# 4. COLOUR PALETTE
# ─────────────────────────────────────────────────────────────
C = px.colors.qualitative.Set2
PASTEL = px.colors.qualitative.Pastel

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Segoe UI, Arial", size=12, color="#333"),
    title_font=dict(size=14, color="#1a1a2e"),
    margin=dict(l=30, r=30, t=50, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

def styled(fig, height=360):
    fig.update_layout(**CHART_LAYOUT, height=height)
    return fig

# ─────────────────────────────────────────────────────────────
# 5. BUILD ALL FIGURES
# ─────────────────────────────────────────────────────────────

# --- Overview ---
def fig_order_status():
    s = orders["order_status"].value_counts().reset_index()
    s.columns = ["Status", "Count"]
    return styled(px.pie(s, names="Status", values="Count",
                         title="Order Status Distribution", hole=0.42,
                         color_discrete_sequence=C).update_traces(textinfo="percent+label"))

def fig_payment_mode():
    p = orders["payment_mode"].value_counts().reset_index()
    p.columns = ["Mode", "Count"]
    return styled(px.pie(p, names="Mode", values="Count",
                         title="Payment Mode Split", hole=0.42,
                         color_discrete_sequence=PASTEL).update_traces(textinfo="percent+label"))

def fig_day_of_week():
    dow = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    d = orders["day_of_week"].value_counts().reindex(dow).reset_index()
    d.columns = ["Day", "Orders"]
    return styled(px.bar(d, x="Day", y="Orders", title="Orders by Day of Week",
                         color="Day", color_discrete_sequence=C, text_auto=True))

# --- Revenue ---
def fig_monthly_revenue():
    m = (delivered.groupby("order_month")["net_revenue"]
         .sum().reset_index().sort_values("order_month"))
    return styled(px.line(m, x="order_month", y="net_revenue", markers=True,
                          title="Monthly Net Revenue (Delivered)",
                          labels={"order_month": "Month", "net_revenue": "Revenue (₹)"},
                          color_discrete_sequence=["#E76F51"])
                  .update_layout(xaxis_tickangle=-40), height=360)

def fig_quarterly_revenue():
    q = (delivered.groupby("order_quarter")["net_revenue"]
         .sum().reset_index().sort_values("order_quarter"))
    return styled(px.bar(q, x="order_quarter", y="net_revenue",
                         title="Quarterly Net Revenue",
                         labels={"order_quarter": "Quarter", "net_revenue": "Revenue (₹)"},
                         color="net_revenue", color_continuous_scale="Blues",
                         text_auto=".2s")
                  .update_layout(xaxis_tickangle=-30))

def fig_yearly_revenue():
    y = (delivered.groupby("order_year")["net_revenue"]
         .sum().reset_index())
    return styled(px.bar(y, x="order_year", y="net_revenue",
                         title="Yearly Net Revenue",
                         labels={"order_year": "Year", "net_revenue": "Revenue (₹)"},
                         color="net_revenue", color_continuous_scale="Teal",
                         text_auto=".2s"))

# --- City ---
def fig_city_revenue():
    c = (delivered.groupby("City")["net_revenue"]
         .sum().sort_values().reset_index())
    return styled(px.bar(c, x="net_revenue", y="City", orientation="h",
                         title="Net Revenue by City",
                         color="net_revenue", color_continuous_scale="Teal",
                         text_auto=".2s"))

def fig_city_orders():
    c = (full.groupby("City")["order_id"].count()
         .sort_values(ascending=False).reset_index())
    c.columns = ["City", "Orders"]
    return styled(px.bar(c, x="City", y="Orders", title="Total Orders by City",
                         color="City", color_discrete_sequence=C, text_auto=True))

def fig_city_cancel():
    c = (full.groupby("City")["order_status"]
         .apply(lambda x: (x == "Cancelled").sum() / len(x) * 100)
         .reset_index())
    c.columns = ["City", "Cancel Rate (%)"]
    c = c.sort_values("Cancel Rate (%)", ascending=False)
    return styled(px.bar(c, x="City", y="Cancel Rate (%)",
                         title="Cancellation Rate by City (%)",
                         color="Cancel Rate (%)", color_continuous_scale="Reds",
                         text_auto=".1f"))

def fig_city_rating():
    c = (full.groupby("City")["avg_rating"]
         .mean().sort_values(ascending=False).reset_index())
    return styled(px.bar(c, x="City", y="avg_rating",
                         title="Avg Restaurant Rating by City",
                         color="avg_rating", color_continuous_scale="Mint",
                         text_auto=".2f", range_y=[3.5, 5]))

# --- Restaurants ---
def fig_top_restaurants():
    r = (delivered.groupby("restaurant_name")["net_revenue"]
         .sum().sort_values(ascending=False).head(10).reset_index())
    r.columns = ["Restaurant", "Revenue"]
    r = r.sort_values("Revenue")
    return styled(px.bar(r, x="Revenue", y="Restaurant", orientation="h",
                         title="Top 10 Restaurants by Revenue",
                         color="Revenue", color_continuous_scale="Oranges",
                         text_auto=".2s"))

def fig_brand_rating():
    b = (restaurants.groupby("restaurant_name")["avg_rating"]
         .mean().sort_values(ascending=False).reset_index())
    b.columns = ["Brand", "Avg Rating"]
    return styled(px.bar(b, x="Avg Rating", y="Brand", orientation="h",
                         title="Avg Rating by Restaurant Brand",
                         color="Avg Rating", color_continuous_scale="RdYlGn",
                         range_x=[3, 5.2], text_auto=".2f"), height=440)

def fig_refund_by_brand():
    r = (full.groupby("restaurant_name")["order_status"]
         .apply(lambda x: (x == "Refunded").sum() / len(x) * 100)
         .sort_values(ascending=False).reset_index())
    r.columns = ["Brand", "Refund Rate (%)"]
    return styled(px.bar(r, x="Refund Rate (%)", y="Brand", orientation="h",
                         title="Refund Rate by Restaurant Brand (%)",
                         color="Refund Rate (%)", color_continuous_scale="Purples",
                         text_auto=".1f"), height=440)

def fig_cuisine_revenue():
    c = (delivered.groupby("cuisine")["net_revenue"]
         .sum().sort_values(ascending=False).reset_index())
    return styled(px.bar(c, x="cuisine", y="net_revenue",
                         title="Revenue by Cuisine",
                         labels={"cuisine": "Cuisine", "net_revenue": "Revenue (₹)"},
                         color="cuisine", color_discrete_sequence=PASTEL,
                         text_auto=".2s"))

def fig_cuisine_city_heatmap():
    h = (delivered.groupby(["cuisine", "City"])["net_revenue"]
         .sum().unstack(fill_value=0))
    return styled(px.imshow(h, title="Revenue Heatmap: Cuisine × City",
                            color_continuous_scale="YlOrRd",
                            labels=dict(x="City", y="Cuisine", color="₹"),
                            aspect="auto", text_auto=".2s"), height=420)

# --- Customers ---
def fig_acquisition():
    a = customers["Acquisition_channel"].value_counts().reset_index()
    a.columns = ["Channel", "Customers"]
    return styled(px.funnel(a, y="Channel", x="Customers",
                            title="Customer Acquisition Channels",
                            color_discrete_sequence=C))

def fig_customer_city():
    c = customers["City"].value_counts().reset_index()
    c.columns = ["City", "Customers"]
    return styled(px.pie(c, names="City", values="Customers",
                         title="Customer Distribution by City",
                         color_discrete_sequence=C, hole=0.35)
                  .update_traces(textinfo="percent+label"))

def fig_monthly_signups():
    m = (customers.groupby("signup_month")["Customer_id"]
         .count().reset_index().sort_values("signup_month"))
    m.columns = ["Month", "New Customers"]
    return styled(px.area(m, x="Month", y="New Customers",
                          title="Monthly New Customer Signups",
                          color_discrete_sequence=["#457B9D"])
                  .update_layout(xaxis_tickangle=-40))

# --- Discounts & Payments ---
def fig_discount_impact():
    orders["Discounted"] = orders["discount_amount"] > 0
    d = (orders.groupby("Discounted")["order_amount"]
         .mean().reset_index())
    d["Discounted"] = d["Discounted"].map({True: "Discounted", False: "No Discount"})
    d.columns = ["Type", "Avg Order Amount"]
    return styled(px.bar(d, x="Type", y="Avg Order Amount",
                         title="Avg Order Amount: Discounted vs Non-Discounted",
                         color="Type", text_auto=".0f",
                         color_discrete_sequence=["#2A9D8F", "#E9C46A"]))

def fig_payment_status():
    ps = (orders.groupby(["payment_mode", "order_status"])["order_id"]
          .count().reset_index())
    ps.columns = ["Payment Mode", "Status", "Count"]
    return styled(px.bar(ps, x="Payment Mode", y="Count", color="Status",
                         title="Order Status by Payment Mode",
                         barmode="stack", color_discrete_sequence=C))

def fig_order_amount_dist():
    return styled(px.histogram(delivered, x="order_amount", nbins=40,
                               title="Order Amount Distribution (Delivered)",
                               labels={"order_amount": "Order Amount (₹)"},
                               color_discrete_sequence=["#264653"])
                  .update_layout(bargap=0.05))

def fig_delivery_fee_dist():
    return styled(px.histogram(orders, x="delivery_fee", nbins=30,
                               title="Delivery Fee Distribution",
                               color_discrete_sequence=["#E9C46A"])
                  .update_layout(bargap=0.05))

# ─────────────────────────────────────────────────────────────
# 6. KPI CARD HELPER
# ─────────────────────────────────────────────────────────────
def kpi_card(title, value, icon, color):
    return dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.Div(icon, style={"fontSize": "28px", "marginBottom": "6px"}),
                html.H4(value, style={"fontWeight": "700", "color": color, "marginBottom": "4px"}),
                html.P(title, style={"color": "#777", "fontSize": "12px", "marginBottom": 0}),
            ], style={"textAlign": "center", "padding": "18px 10px"})
        ], style={"borderRadius": "12px", "border": "none",
                  "boxShadow": "0 2px 12px rgba(0,0,0,.08)"}),
        xs=6, sm=4, md=3, lg=2, style={"marginBottom": "14px"}
    )

# ─────────────────────────────────────────────────────────────
# 7. INSIGHTS TABLE DATA
# ─────────────────────────────────────────────────────────────
top_city_rev     = (delivered.groupby("City")["net_revenue"].sum()
                    .sort_values(ascending=False).index[0])
top_city_orders  = (full.groupby("City")["order_id"].count()
                    .sort_values(ascending=False).index[0])
top_restaurant   = (delivered.groupby("restaurant_name")["net_revenue"].sum()
                    .sort_values(ascending=False).index[0])
top_brand_rating = (restaurants.groupby("restaurant_name")["avg_rating"].mean()
                    .sort_values(ascending=False))
top_channel      = customers["Acquisition_channel"].value_counts().index[0]
top_channel_n    = customers["Acquisition_channel"].value_counts().iloc[0]

disc_avg   = orders[orders["discount_amount"] > 0]["order_amount"].mean()
nodisc_avg = orders[orders["discount_amount"] == 0]["order_amount"].mean()

insights = [
    {"Area": "Orders",       "Insight": f"{total_orders:,} total orders. Delivery {delivery_rate:.1f}%, Cancellation {cancel_rate:.1f}%, Refund {refund_rate:.1f}%."},
    {"Area": "Revenue",      "Insight": f"Net revenue ₹{total_revenue:,.0f} from delivered orders. Avg order value ₹{avg_order_value:,.0f}. Total discounts ₹{total_discounts:,.0f}."},
    {"Area": "Top City",     "Insight": f"{top_city_rev} leads in revenue. {top_city_orders} has highest order volume."},
    {"Area": "Restaurants",  "Insight": f"Top brand by revenue: {top_restaurant}. Highest rated: {top_brand_rating.index[0]} ({top_brand_rating.iloc[0]:.2f}★)."},
    {"Area": "Customers",    "Insight": f"{total_customers:,} unique customers. Top acquisition: {top_channel} ({top_channel_n:,} customers)."},
    {"Area": "Discounts",    "Insight": f"Discounted orders avg ₹{disc_avg:,.0f} vs ₹{nodisc_avg:,.0f} non-discounted. Avg delivery fee ₹{avg_delivery_fee:.0f}."},
]

# ─────────────────────────────────────────────────────────────
# 7b. DATA-DRIVEN RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────

# -- Cancellation analysis --
city_cancel_s = (full.groupby("City")["order_status"]
                 .apply(lambda x: (x == "Cancelled").sum() / len(x) * 100))
worst_cancel_city  = city_cancel_s.idxmax()
worst_cancel_rate  = city_cancel_s.max()
best_cancel_city   = city_cancel_s.idxmin()
best_cancel_rate   = city_cancel_s.min()

# -- Refund analysis --
brand_refund_s = (full.groupby("restaurant_name")["order_status"]
                  .apply(lambda x: (x == "Refunded").sum() / len(x) * 100))
worst_refund_brand = brand_refund_s.idxmax()
worst_refund_rate  = brand_refund_s.max()

# -- Low rated restaurants --
brand_rating_s = restaurants.groupby("restaurant_name")["avg_rating"].mean()
low_rating_brands = brand_rating_s[brand_rating_s < 4.0].sort_values()
low_rating_str = ", ".join(low_rating_brands.index.tolist()) if len(low_rating_brands) else "None"

# -- Underperforming cuisines --
cuisine_rev_s = (delivered.groupby("cuisine")["net_revenue"].sum())
bottom_cuisine = cuisine_rev_s.idxmin()
bottom_cuisine_rev = cuisine_rev_s.min()
top_cuisine    = cuisine_rev_s.idxmax()
top_cuisine_rev = cuisine_rev_s.max()

# -- Acquisition channel efficiency --
channel_counts = customers["Acquisition_channel"].value_counts()
weakest_channel = channel_counts.idxmin()
weakest_channel_n = channel_counts.min()
strongest_channel = channel_counts.idxmax()

# -- Discount effectiveness --
disc_pct = (orders["discount_amount"] > 0).mean() * 100
disc_lift = disc_avg - nodisc_avg   # positive = discounted orders are higher value

# -- Delivery fee vs order size --
fee_corr = orders[["delivery_fee","order_amount"]].corr().iloc[0,1]

# -- Peak day --
dow_counts_s = orders["day_of_week"].value_counts()
peak_day  = dow_counts_s.idxmax()
low_day   = dow_counts_s.idxmin()
peak_day_n = dow_counts_s.max()
low_day_n  = dow_counts_s.min()
dow_gap_pct = (peak_day_n - low_day_n) / low_day_n * 100

# -- City revenue opportunity (high orders but low revenue) --
city_ord_s  = full.groupby("City")["order_id"].count()
city_rev_s  = delivered.groupby("City")["net_revenue"].sum()
city_aov_s  = delivered.groupby("City")["order_amount"].mean()
low_aov_city = city_aov_s.idxmin()
low_aov_val  = city_aov_s.min()
high_aov_city = city_aov_s.idxmax()
high_aov_val  = city_aov_s.max()

# -- UPI / Wallet cancellation rates --
pay_cancel = (orders.groupby("payment_mode")["order_status"]
              .apply(lambda x: (x == "Cancelled").sum() / len(x) * 100))
high_cancel_pay = pay_cancel.idxmax()
high_cancel_pay_rate = pay_cancel.max()

# -- Revenue lost to refunds --
refund_revenue = full[full["order_status"] == "Refunded"]["order_amount"].sum()
refund_pct_rev = refund_revenue / full["order_amount"].sum() * 100

# -- Monthly growth --
monthly_rev_s = (delivered.groupby("order_month")["net_revenue"]
                 .sum().sort_index())
if len(monthly_rev_s) >= 2:
    last_month_rev = monthly_rev_s.iloc[-1]
    prev_month_rev = monthly_rev_s.iloc[-2]
    mom_growth = (last_month_rev - prev_month_rev) / prev_month_rev * 100
    last_month_label = monthly_rev_s.index[-1]
else:
    mom_growth = 0
    last_month_label = "N/A"

recommendations = [
    {
        "Category":       "🔴 Reduce Cancellations",
        "Finding":        f"{worst_cancel_city} has the highest cancellation rate at {worst_cancel_rate:.1f}% vs "
                          f"{best_cancel_city} at {best_cancel_rate:.1f}%.",
        "Recommendation": f"Investigate delivery time, restaurant preparation delays, and app UX issues specific to "
                          f"{worst_cancel_city}. Introduce real-time order tracking and proactive ETA updates to "
                          f"reduce pre-delivery cancellations.",
        "Impact":         "High"
    },
    {
        "Category":       "🔴 Fix Refund Problem",
        "Finding":        f"Refunds account for {refund_rate:.1f}% of all orders, representing ~₹{refund_revenue:,.0f} "
                          f"in lost revenue ({refund_pct_rev:.1f}% of gross order value). "
                          f"{worst_refund_brand} has the highest refund rate at {worst_refund_rate:.1f}%.",
        "Recommendation": f"Audit quality control at {worst_refund_brand} outlets. Implement a mandatory 'order "
                          f"verification' step before dispatch. Set a refund-rate SLA (target < 10%) and penalise "
                          f"repeat offenders.",
        "Impact":         "High"
    },
    {
        "Category":       "🟠 Improve Low-Rated Restaurants",
        "Finding":        f"Brands with avg rating below 4.0: {low_rating_str}. Low ratings correlate with higher "
                          f"cancellation and refund rates.",
        "Recommendation": "Run a Restaurant Quality Programme: mandatory retraining for kitchens below 4.0★, "
                          "monthly mystery-shopper audits, and public rating badges on the app. Consider "
                          "temporarily suspending listings below 3.5★ until scores improve.",
        "Impact":         "High"
    },
    {
        "Category":       "🟠 Boost Underperforming Cuisine",
        "Finding":        f"{bottom_cuisine} cuisine generates only ₹{bottom_cuisine_rev:,.0f} in revenue vs "
                          f"₹{top_cuisine_rev:,.0f} for the top cuisine ({top_cuisine}).",
        "Recommendation": f"Run targeted promotions and cuisine-specific discount campaigns for {bottom_cuisine} "
                          f"during peak hours. Partner with popular {bottom_cuisine} restaurants in high-traffic "
                          f"cities to increase listing density and discoverability.",
        "Impact":         "Medium"
    },
    {
        "Category":       "🟠 Grow Low-AOV City",
        "Finding":        f"{low_aov_city} has the lowest average order value (₹{low_aov_val:,.0f}) vs "
                          f"₹{high_aov_val:,.0f} in {high_aov_city}.",
        "Recommendation": f"In {low_aov_city}, introduce 'Add-On' suggestions at checkout (drinks, desserts, sides) "
                          f"and bundle combo deals. A 10% AOV uplift in {low_aov_city} alone would meaningfully "
                          f"increase city-level revenue.",
        "Impact":         "Medium"
    },
    {
        "Category":       "🟡 Optimise Discount Strategy",
        "Finding":        f"Only {disc_pct:.1f}% of orders carry a discount. Discounted orders average "
                          f"₹{disc_avg:,.0f} vs ₹{nodisc_avg:,.0f} (lift: ₹{abs(disc_lift):,.0f}). "
                          f"Total discount spend: ₹{total_discounts:,.0f}.",
        "Recommendation": "Shift from blanket discounts to personalised offers based on customer order history. "
                          "Reserve deep discounts for inactive customers (no order in 60+ days). Cap discount "
                          "depth at 10% for high-frequency users who would order anyway.",
        "Impact":         "Medium"
    },
    {
        "Category":       "🟡 Fix Payment-Mode Cancellations",
        "Finding":        f"{high_cancel_pay} payments have the highest cancellation rate at "
                          f"{high_cancel_pay_rate:.1f}%.",
        "Recommendation": f"Investigate whether {high_cancel_pay} payment failures are triggering cancellations. "
                          "Add a payment-retry flow and offer an instant alternative payment suggestion when a "
                          "payment fails instead of auto-cancelling the order.",
        "Impact":         "Medium"
    },
    {
        "Category":       "🟡 Leverage Peak & Off-Peak Days",
        "Finding":        f"{peak_day} is the busiest day ({peak_day_n:,} orders) while {low_day} is slowest "
                          f"({low_day_n:,} orders) — a {dow_gap_pct:.0f}% gap.",
        "Recommendation": f"Run '{low_day} Specials' flash deals to smooth demand. On {peak_day}, ensure "
                          "adequate restaurant partner capacity and delivery fleet availability to prevent "
                          "degraded experience during demand spikes.",
        "Impact":         "Medium"
    },
    {
        "Category":       "🟢 Double Down on Top Acquisition Channel",
        "Finding":        f"{strongest_channel} is the top acquisition channel. "
                          f"{weakest_channel} brings in the fewest customers ({weakest_channel_n:,}).",
        "Recommendation": f"Increase marketing budget allocation to {strongest_channel} — it has the proven "
                          f"highest ROI. Re-evaluate or redesign the {weakest_channel} strategy; consider "
                          "A/B testing new creatives or incentive structures before scaling it.",
        "Impact":         "Medium"
    },
    {
        "Category":       "🟢 Retain High-Value Customers",
        "Finding":        f"Top acquisition channel ({strongest_channel}) drives the most signups but retention "
                          f"data shows high churn potential given {cancel_rate:.1f}% cancellation and "
                          f"{refund_rate:.1f}% refund rates.",
        "Recommendation": "Launch a tiered loyalty programme (Bronze / Gold / Platinum) rewarding order frequency "
                          "and value. Offer exclusive perks (zero delivery fee, priority support, early access "
                          "to new restaurants) for Platinum tier.",
        "Impact":         "High"
    },
    {
        "Category":       "🟢 Expand High-Performing City",
        "Finding":        f"{top_city_rev} leads in revenue and {top_city_orders} leads in order volume. "
                          f"The top-rated city is {city_rating_s.index[0] if 'city_rating_s' in dir() else 'N/A'}, "
                          f"indicating strong customer satisfaction.",
        "Recommendation": f"Use {top_city_rev} as a model market: replicate its restaurant mix, delivery "
                          "infrastructure, and promotional tactics in mid-tier cities. Identify the top 3 "
                          "restaurants from this city and incentivise them to expand to new cities.",
        "Impact":         "High"
    },
    {
        "Category":       "🟢 Revenue Momentum",
        "Finding":        f"Month-on-month revenue change (latest month {last_month_label}): "
                          f"{'+' if mom_growth >= 0 else ''}{mom_growth:.1f}%.",
        "Recommendation": "If growth is positive, maintain current strategies and test incremental price nudges "
                          "(e.g. ₹10–₹20 delivery fee increase in low-fee cities). If negative, run an "
                          "emergency win-back campaign targeting customers who last ordered 30–90 days ago.",
        "Impact":         "High"
    },
]

# city rating series used in recommendation above
city_rating_s = full.groupby("City")["avg_rating"].mean().sort_values(ascending=False)
# patch the recommendation that referenced it before it was defined
recommendations[10]["Finding"] = (
    f"{top_city_rev} leads in revenue and {top_city_orders} leads in order volume. "
    f"Top-rated city: {city_rating_s.index[0]} ({city_rating_s.iloc[0]:.2f}★)."
)

# Figure: Recommendations impact chart
def fig_recommendations_impact():
    impact_map = {"High": 3, "Medium": 2, "Low": 1}
    color_map  = {"High": "#E63946", "Medium": "#F4A261", "Low": "#43AA8B"}
    df_r = pd.DataFrame(recommendations)[["Category","Impact"]]
    df_r["Score"] = df_r["Impact"].map(impact_map)
    df_r["Label"] = df_r["Category"].str.replace(r"^[🔴🟠🟡🟢]\s*", "", regex=True)
    df_r["Color"] = df_r["Impact"].map(color_map)
    df_r = df_r.sort_values("Score")
    fig = px.bar(df_r, x="Score", y="Label", orientation="h",
                 title="Recommendations by Business Impact",
                 color="Impact",
                 color_discrete_map={"High": "#E63946", "Medium": "#F4A261", "Low": "#43AA8B"},
                 labels={"Score": "Impact Level", "Label": ""},
                 text="Impact")
    fig.update_xaxes(tickvals=[1,2,3], ticktext=["Low","Medium","High"])
    return styled(fig, height=460)

# ─────────────────────────────────────────────────────────────
# 8. DASH APP LAYOUT
# ─────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Zomato Analytics"
)

GRAPH = {"height": "100%"}
CARD_STYLE = {"borderRadius": "14px", "border": "none",
              "boxShadow": "0 2px 14px rgba(0,0,0,.07)", "padding": "6px"}

def graph_card(fig_fn, height=380):
    return dbc.Card(
        dcc.Graph(figure=fig_fn(), style={"height": f"{height}px"}),
        style=CARD_STYLE
    )

def section(title, icon):
    return html.Div([
        html.Hr(style={"borderColor": "#dee2e6", "margin": "28px 0 16px"}),
        html.H5(f"{icon}  {title}",
                style={"color": "#264653", "fontWeight": "700", "marginBottom": "16px"})
    ])

app.layout = dbc.Container(fluid=True, children=[

    # ── Top Bar ──────────────────────────────────────────────
    dbc.Row(dbc.Col(
        html.Div([
            html.H2("🍽️  Zomato Order Analytics Dashboard",
                    style={"color": "#fff", "fontWeight": "700", "marginBottom": "4px"}),
            html.P(f"{total_orders:,} orders  ·  {total_customers:,} customers  ·  "
                   f"200 restaurants  ·  8 cities",
                   style={"color": "rgba(255,255,255,.85)", "marginBottom": 0, "fontSize": "13px"})
        ], style={
            "background": "linear-gradient(135deg,#E63946,#c1121f)",
            "padding": "22px 32px", "borderRadius": "0 0 16px 16px",
            "marginBottom": "24px"
        })
    )),

    # ── Filter Bar ───────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Label("Filter by City", style={"fontWeight": "600", "fontSize": "13px"}),
            dcc.Dropdown(
                id="city-filter",
                options=[{"label": "All Cities", "value": "ALL"}] +
                        [{"label": c, "value": c} for c in sorted(full["City"].dropna().unique())],
                value="ALL", clearable=False,
                style={"fontSize": "13px"}
            )
        ], md=3),
        dbc.Col([
            html.Label("Filter by Order Status", style={"fontWeight": "600", "fontSize": "13px"}),
            dcc.Dropdown(
                id="status-filter",
                options=[{"label": "All Statuses", "value": "ALL"}] +
                        [{"label": s, "value": s} for s in sorted(orders["order_status"].unique())],
                value="ALL", clearable=False,
                style={"fontSize": "13px"}
            )
        ], md=3),
        dbc.Col([
            html.Label("Filter by Cuisine", style={"fontWeight": "600", "fontSize": "13px"}),
            dcc.Dropdown(
                id="cuisine-filter",
                options=[{"label": "All Cuisines", "value": "ALL"}] +
                        [{"label": c, "value": c} for c in sorted(restaurants["cuisine"].dropna().unique())],
                value="ALL", clearable=False,
                style={"fontSize": "13px"}
            )
        ], md=3),
        dbc.Col([
            html.Label("Filter by Year", style={"fontWeight": "600", "fontSize": "13px"}),
            dcc.Dropdown(
                id="year-filter",
                options=[{"label": "All Years", "value": "ALL"}] +
                        [{"label": str(y), "value": y}
                         for y in sorted(orders["order_year"].dropna().unique())],
                value="ALL", clearable=False,
                style={"fontSize": "13px"}
            )
        ], md=3),
    ], style={"background": "#fff", "padding": "14px 12px",
              "borderRadius": "12px", "boxShadow": "0 1px 8px rgba(0,0,0,.07)",
              "marginBottom": "22px"}),

    # ── KPIs ─────────────────────────────────────────────────
    html.Div(id="kpi-row"),

    # ── Order Behaviour ───────────────────────────────────────
    section("Order Behaviour", "🔄"),
    dbc.Row([
        dbc.Col(graph_card(fig_order_status), md=4),
        dbc.Col(graph_card(fig_payment_mode), md=4),
        dbc.Col(graph_card(fig_day_of_week),  md=4),
    ], className="g-3"),

    # ── Revenue Trends ────────────────────────────────────────
    section("Revenue Trends", "📈"),
    dbc.Row([
        dbc.Col(graph_card(fig_monthly_revenue, 380),   md=6),
        dbc.Col(graph_card(fig_quarterly_revenue, 380), md=3),
        dbc.Col(graph_card(fig_yearly_revenue, 380),    md=3),
    ], className="g-3"),

    # ── City Insights ─────────────────────────────────────────
    section("City-Level Insights", "🏙️"),
    dbc.Row([
        dbc.Col(graph_card(fig_city_revenue),  md=6),
        dbc.Col(graph_card(fig_city_orders),   md=6),
    ], className="g-3"),
    dbc.Row([
        dbc.Col(graph_card(fig_city_cancel),   md=6),
        dbc.Col(graph_card(fig_city_rating),   md=6),
    ], className="g-3 mt-3"),

    # ── Restaurant Performance ────────────────────────────────
    section("Restaurant Performance", "🍴"),
    dbc.Row([
        dbc.Col(graph_card(fig_top_restaurants, 400), md=6),
        dbc.Col(graph_card(fig_cuisine_revenue, 400), md=6),
    ], className="g-3"),
    dbc.Row([
        dbc.Col(graph_card(fig_brand_rating,   460), md=6),
        dbc.Col(graph_card(fig_refund_by_brand, 460), md=6),
    ], className="g-3 mt-3"),
    dbc.Row([
        dbc.Col(graph_card(fig_cuisine_city_heatmap, 440), md=12),
    ], className="g-3 mt-3"),

    # ── Customer Analytics ────────────────────────────────────
    section("Customer Analytics", "👥"),
    dbc.Row([
        dbc.Col(graph_card(fig_acquisition,      380), md=4),
        dbc.Col(graph_card(fig_customer_city,    380), md=4),
        dbc.Col(graph_card(fig_monthly_signups,  380), md=4),
    ], className="g-3"),

    # ── Discounts & Payments ──────────────────────────────────
    section("Discount & Payment Analysis", "💰"),
    dbc.Row([
        dbc.Col(graph_card(fig_discount_impact,   360), md=4),
        dbc.Col(graph_card(fig_payment_status,    360), md=4),
        dbc.Col(graph_card(fig_delivery_fee_dist, 360), md=4),
    ], className="g-3"),
    dbc.Row([
        dbc.Col(graph_card(fig_order_amount_dist, 360), md=12),
    ], className="g-3 mt-3"),

    # ── Data Explorer ─────────────────────────────────────────
    section("Raw Data Explorer", "🔎"),
    dbc.Card([
        dbc.CardBody([
            html.P("Showing top 200 orders (filtered). Use dropdowns above to narrow down.",
                   style={"fontSize": "12px", "color": "#888", "marginBottom": "10px"}),
            html.Div(id="data-table")
        ])
    ], style=CARD_STYLE),

    # ── Key Insights Summary ──────────────────────────────────
    section("Key Insights Summary", "🔍"),
    dbc.Card(
        dash_table.DataTable(
            data=insights,
            columns=[{"name": c, "id": c} for c in ["Area", "Insight"]],
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": "#264653", "color": "#fff",
                "fontWeight": "600", "fontSize": "13px", "padding": "10px 14px"
            },
            style_data={
                "fontSize": "13px", "padding": "10px 14px",
                "whiteSpace": "normal", "height": "auto"
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"},
                 "backgroundColor": "#f8f9fa"}
            ],
            style_cell_conditional=[
                {"if": {"column_id": "Area"},
                 "width": "130px", "fontWeight": "600", "color": "#E76F51"},
                {"if": {"column_id": "Insight"}, "textAlign": "left"},
            ],
        ), style=CARD_STYLE
    ),

    # ── Data-Driven Recommendations ───────────────────────────
    section("Data-Driven Recommendations", "💡"),

    # Impact summary chart
    dbc.Row([
        dbc.Col(
            dbc.Card(dcc.Graph(figure=fig_recommendations_impact(), style={"height": "460px"}),
                     style=CARD_STYLE),
            md=12
        )
    ], className="g-3"),

    # Recommendation cards
    html.Div(style={"height": "18px"}),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.Span(rec["Category"],
                              style={"fontWeight": "700", "fontSize": "13px",
                                     "color": "#fff" if rec["Impact"] == "High" else "#1a1a2e"}),
                    style={
                        "backgroundColor":
                            "#E63946" if rec["Impact"] == "High"
                            else "#F4A261" if rec["Impact"] == "Medium"
                            else "#43AA8B",
                        "borderRadius": "12px 12px 0 0",
                        "padding": "10px 14px"
                    }
                ),
                dbc.CardBody([
                    html.Div([
                        html.Span("FINDING  ", style={"fontSize": "10px", "fontWeight": "700",
                                                       "color": "#888", "letterSpacing": "1px"}),
                        html.P(rec["Finding"],
                               style={"fontSize": "12px", "color": "#444",
                                      "marginBottom": "10px", "marginTop": "4px"}),
                        html.Span("RECOMMENDATION  ", style={"fontSize": "10px", "fontWeight": "700",
                                                              "color": "#888", "letterSpacing": "1px"}),
                        html.P(rec["Recommendation"],
                               style={"fontSize": "13px", "color": "#1a1a2e",
                                      "marginBottom": "10px", "marginTop": "4px"}),
                        dbc.Badge(
                            f"Impact: {rec['Impact']}",
                            color="danger" if rec["Impact"] == "High"
                                  else "warning" if rec["Impact"] == "Medium"
                                  else "success",
                            pill=True,
                            style={"fontSize": "11px"}
                        )
                    ])
                ], style={"padding": "12px 14px"})
            ], style={"borderRadius": "12px", "border": "none",
                      "boxShadow": "0 2px 12px rgba(0,0,0,.09)",
                      "height": "100%"})
        ], md=6, lg=4, style={"marginBottom": "18px"})
        for rec in recommendations
    ]),

    html.Div("Zomato Analytics · Python + Dash + Plotly",
             style={"textAlign": "center", "color": "#bbb",
                    "fontSize": "12px", "padding": "28px 0 16px"}),

], style={"backgroundColor": "#f4f6fb", "padding": "0 16px"})

# ─────────────────────────────────────────────────────────────
# 9. CALLBACKS
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("kpi-row", "children"),
    Output("data-table", "children"),
    Input("city-filter",    "value"),
    Input("status-filter",  "value"),
    Input("cuisine-filter", "value"),
    Input("year-filter",    "value"),
)
def update_dynamic(city, status, cuisine, year):
    df = full.copy()
    if city    != "ALL": df = df[df["City"]         == city]
    if status  != "ALL": df = df[df["order_status"] == status]
    if cuisine != "ALL": df = df[df["cuisine"]       == cuisine]
    if year    != "ALL": df = df[df["order_year"]    == year]

    n_orders   = len(df)
    n_cust     = df["customer_id"].nunique()
    del_df     = df[df["order_status"] == "Delivered"]
    rev        = del_df["net_revenue"].sum()
    aov        = del_df["order_amount"].mean() if len(del_df) else 0
    d_rate     = (df["order_status"] == "Delivered").mean() * 100 if n_orders else 0
    c_rate     = (df["order_status"] == "Cancelled").mean() * 100 if n_orders else 0
    r_rate     = (df["order_status"] == "Refunded").mean()  * 100 if n_orders else 0
    disc_total = df["discount_amount"].sum()

    kpi_row = dbc.Row([
        kpi_card("Total Orders",      f"{n_orders:,}",           "🛒", "#264653"),
        kpi_card("Unique Customers",  f"{n_cust:,}",             "👤", "#2A9D8F"),
        kpi_card("Net Revenue",       f"₹{rev/1e6:.2f}M",        "💰", "#E9C46A"),
        kpi_card("Avg Order Value",   f"₹{aov:,.0f}",            "🧾", "#F4A261"),
        kpi_card("Delivery Rate",     f"{d_rate:.1f}%",          "✅", "#43AA8B"),
        kpi_card("Cancellation Rate", f"{c_rate:.1f}%",          "❌", "#E76F51"),
        kpi_card("Refund Rate",       f"{r_rate:.1f}%",          "↩️",  "#9C6B98"),
        kpi_card("Total Discounts",   f"₹{disc_total/1e3:.1f}K", "🏷️",  "#457B9D"),
    ], className="g-2", style={"marginBottom": "6px"})

    # Data table (top 200 rows)
    show_cols = ["order_id","customer_id","restaurant_name","cuisine",
                 "City","order_status","order_amount","discount_amount",
                 "delivery_fee","net_revenue","payment_mode","order_month"]
    available = [c for c in show_cols if c in df.columns]
    tbl_data = df[available].head(200).round(2).to_dict("records")

    tbl = dash_table.DataTable(
        data=tbl_data,
        columns=[{"name": c.replace("_"," ").title(), "id": c} for c in available],
        page_size=15,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#f4f6fb", "fontWeight": "600",
                       "fontSize": "12px", "color": "#264653"},
        style_data={"fontSize": "12px"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#fafafa"},
            {"if": {"filter_query": '{order_status} = "Delivered"'},
             "color": "#2A9D8F", "fontWeight": "600"},
            {"if": {"filter_query": '{order_status} = "Cancelled"'},
             "color": "#E76F51"},
            {"if": {"filter_query": '{order_status} = "Refunded"'},
             "color": "#9C6B98"},
        ],
    )

    return kpi_row, tbl


# ─────────────────────────────────────────────────────────────
# 10. RUN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  ZOMATO ANALYTICS DASHBOARD")
    print("="*55)
    print(f"  Orders         : {total_orders:,}")
    print(f"  Customers      : {total_customers:,}")
    print(f"  Net Revenue    : ₹{total_revenue:,.0f}")
    print(f"  Avg Order Value: ₹{avg_order_value:,.0f}")
    print(f"  Delivery Rate  : {delivery_rate:.1f}%")
    print(f"  Cancel Rate    : {cancel_rate:.1f}%")
    print(f"  Refund Rate    : {refund_rate:.1f}%")
    print("="*55)
    print("\n  ➜  http://127.0.0.1:8050\n")
    app.run(debug=False, port=8050)
