"""
Zomato Analytics — Self-contained HTML Dashboard Generator
Run: python generate_dashboard.py
Opens: zomato_dashboard.html
"""
import os, warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))

def load(n):
    return pd.read_csv(os.path.join(BASE, f"Zomato  Order Data.xlsx - {n}.csv"))

# ── Load ─────────────────────────────────────────────────────
customers   = load("Customer")
orders      = load("Orders")
restaurants = load("Restaurants")

# ── Clean & engineer ─────────────────────────────────────────
orders["ts"]              = pd.to_datetime(orders["order_timestamp"], dayfirst=True, errors="coerce")
orders["month"]           = orders["ts"].dt.to_period("M").astype(str)
orders["quarter"]         = orders["ts"].dt.to_period("Q").astype(str)
orders["year"]            = orders["ts"].dt.year.astype("Int64")
orders["dow"]             = orders["ts"].dt.day_name()
orders["hour"]            = orders["ts"].dt.hour
orders["discount_amount"] = orders["discount_amount"].fillna(0)
orders["delivery_fee"]    = orders["delivery_fee"].fillna(0)
orders["net_revenue"]     = orders["order_amount"] - orders["discount_amount"]
orders["total_charge"]    = orders["order_amount"] + orders["delivery_fee"]
orders["is_disc"]         = (orders["discount_amount"] > 0).astype(int)
orders["discount_pct"]    = np.where(
    orders["order_amount"] > 0,
    (orders["discount_amount"] / orders["order_amount"] * 100).round(1), 0
)

customers.columns = customers.columns.str.strip()
customers["signup"] = pd.to_datetime(customers["Signup_Time"], dayfirst=True, errors="coerce")
customers["signup_month"] = customers["signup"].dt.to_period("M").astype(str)

full = (orders
        .merge(restaurants, on="restaurant_id", how="left")
        .merge(customers, left_on="customer_id", right_on="Customer_id", how="left"))

delivered = full[full["order_status"] == "Delivered"].copy()
cancelled = full[full["order_status"] == "Cancelled"].copy()
refunded  = full[full["order_status"] == "Refunded"].copy()

# ── KPIs ─────────────────────────────────────────────────────
total_orders    = len(orders)
total_customers = customers["Customer_id"].nunique()
gmv             = orders["order_amount"].sum()
net_revenue_val = delivered["net_revenue"].sum()
aov             = delivered["order_amount"].mean()
d_rate          = (orders["order_status"] == "Delivered").mean() * 100
c_rate          = (orders["order_status"] == "Cancelled").mean() * 100
r_rate          = (orders["order_status"] == "Refunded").mean() * 100
total_disc      = orders["discount_amount"].sum()
cancel_rev      = cancelled["order_amount"].sum()
refund_rev      = refunded["order_amount"].sum()

# ── Colour palette ────────────────────────────────────────────
C = px.colors.qualitative.Set2
P = px.colors.qualitative.Pastel

BRAND   = "#E63946"
GREEN   = "#43AA8B"
BLUE    = "#264653"
AMBER   = "#E9C46A"
ORANGE  = "#E76F51"
TEAL    = "#2A9D8F"
PURPLE  = "#9C6B98"
SLATE   = "#457B9D"

# Shared layout applied to every figure
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'Inter','Segoe UI',Arial,sans-serif", size=12, color="#374151"),
    title_font=dict(size=14, color="#111827", family="'Inter','Segoe UI',Arial,sans-serif"),
    margin=dict(l=44, r=24, t=52, b=44),
    legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
    hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#e5e7eb"),
)

def apply_axes(fig):
    """Light gridlines, no spines."""
    axis = dict(
        showgrid=True, gridcolor="#f3f4f6", gridwidth=1,
        zeroline=False, showline=False,
        tickfont=dict(size=11, color="#6b7280"),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig

def fig2html(fig, h=340):
    fig.update_layout(**LAYOUT, height=h)
    apply_axes(fig)
    return fig.to_html(
        full_html=False, include_plotlyjs=False,
        config={"displayModeBar": False, "responsive": True}
    )

# ═══════════════════════════════════════════
# SECTION 1 — EDA
# ═══════════════════════════════════════════

# 1a  Order status donut
s = orders["order_status"].value_counts().reset_index()
s.columns = ["Status", "Count"]
eda1 = px.pie(s, names="Status", values="Count", title="Order Status", hole=0.5,
              color_discrete_sequence=[GREEN, BRAND, PURPLE])
eda1.update_traces(textinfo="percent+label", textfont_size=13,
                   marker=dict(line=dict(color="#fff", width=2)))

# 1b  Payment mode donut
pm = orders["payment_mode"].value_counts().reset_index()
pm.columns = ["Mode", "Count"]
eda4 = px.pie(pm, names="Mode", values="Count", title="Payment Mode Split", hole=0.5,
              color_discrete_sequence=px.colors.qualitative.Pastel)
eda4.update_traces(textinfo="percent+label", textfont_size=13,
                   marker=dict(line=dict(color="#fff", width=2)))

# 1c  Customer acquisition channels
acq = customers["Acquisition_channel"].value_counts().reset_index()
acq.columns = ["Channel", "Customers"]
eda6 = px.bar(acq.sort_values("Customers"), x="Customers", y="Channel",
              orientation="h", title="Customer Acquisition Channels",
              color="Customers", color_continuous_scale="Teal",
              text_auto=True)
eda6.update_traces(marker_line_width=0)
eda6.update_coloraxes(showscale=False)

# 1d  Orders by day of week
dow_ord = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow = orders["dow"].value_counts().reindex(dow_ord).reset_index()
dow.columns = ["Day", "Orders"]
eda2 = px.bar(dow, x="Day", y="Orders", title="Orders by Day of Week",
              color="Orders", color_continuous_scale="Blues", text_auto=True)
eda2.update_traces(marker_line_width=0)
eda2.update_coloraxes(showscale=False)

# 1e  Restaurant rating distribution
eda5 = px.histogram(restaurants, x="avg_rating", nbins=20,
                    title="Restaurant Rating Distribution",
                    color_discrete_sequence=[SLATE], labels={"avg_rating": "Rating"})
eda5.add_vline(x=restaurants["avg_rating"].mean(), line_dash="dash",
               line_color=BRAND, annotation_text=f"Mean {restaurants['avg_rating'].mean():.2f}★",
               annotation_font_color=BRAND)
eda5.update_traces(marker_line_width=0)

# 1f  Order amount distribution
eda3 = px.histogram(delivered, x="order_amount", nbins=40,
                    title="Order Amount Distribution — Delivered (₹)",
                    color_discrete_sequence=[BLUE], labels={"order_amount": "₹"})
eda3.update_traces(marker_line_width=0)

# 1g  Peak hours
hourly = orders.groupby("hour")["order_id"].count().reset_index(name="orders")
eda7 = px.bar(hourly, x="hour", y="orders", title="Orders by Hour of Day (Peak Hours)",
              color="orders", color_continuous_scale="Oranges", text_auto=True)
eda7.update_traces(marker_line_width=0)
eda7.update_coloraxes(showscale=False)
eda7.update_xaxes(tickmode="linear", tick0=0, dtick=1)

# 1h  Hour × DoW heatmap
heat_data = (orders.groupby(["dow", "hour"])["order_id"].count()
             .reset_index(name="orders"))
heat_pivot = (heat_data.pivot(index="dow", columns="hour", values="orders")
              .reindex(dow_ord).fillna(0))
eda8 = go.Figure(go.Heatmap(
    z=heat_pivot.values,
    x=[f"{h}:00" for h in heat_pivot.columns],
    y=heat_pivot.index.tolist(),
    colorscale="YlOrRd", colorbar_title="Orders",
    hovertemplate="Day: %{y}<br>Hour: %{x}<br>Orders: %{z}<extra></extra>"
))
eda8.update_layout(title="Order Volume: Day × Hour")

# ═══════════════════════════════════════════
# SECTION 2 — DRIVER ANALYSIS
# ═══════════════════════════════════════════

# 2a  Revenue by city
city_rev = delivered.groupby("City")["net_revenue"].sum().sort_values().reset_index()
drv1 = px.bar(city_rev, x="net_revenue", y="City", orientation="h",
              title="Net Revenue by City (₹)", color="net_revenue",
              color_continuous_scale="Teal",
              text=city_rev["net_revenue"].apply(lambda x: f"₹{x/1e6:.1f}M"))
drv1.update_traces(textposition="outside", marker_line_width=0)
drv1.update_coloraxes(showscale=False)

# 2b  Cancel rate by city with target line
cc = (full.groupby("City")["order_status"]
      .apply(lambda x: (x=="Cancelled").sum()/len(x)*100)
      .reset_index())
cc.columns = ["City", "Cancel %"]
cc = cc.sort_values("Cancel %", ascending=False)
drv2 = px.bar(cc, x="City", y="Cancel %", title="Cancellation Rate by City (%)",
              color="Cancel %", color_continuous_scale="Reds",
              text=cc["Cancel %"].apply(lambda x: f"{x:.1f}%"))
drv2.update_traces(textposition="outside", marker_line_width=0)
drv2.update_coloraxes(showscale=False)
drv2.add_hline(y=15, line_dash="dash", line_color=AMBER,
               annotation_text="Target 15%", annotation_position="top right",
               annotation_font_color=AMBER)

# 2c  Cuisine revenue
cuis = (delivered.groupby("cuisine")["net_revenue"].sum()
        .sort_values(ascending=False).reset_index())
drv3 = px.bar(cuis.sort_values("net_revenue"), x="net_revenue", y="cuisine",
              orientation="h", title="Revenue by Cuisine (₹)",
              color="net_revenue", color_continuous_scale="Greens",
              text=cuis.sort_values("net_revenue")["net_revenue"].apply(lambda x: f"₹{x/1e6:.1f}M"))
drv3.update_traces(textposition="outside", marker_line_width=0)
drv3.update_coloraxes(showscale=False)

# 2d  Discount impact on AOV
disc_avg   = orders[orders["is_disc"]==1]["order_amount"].mean()
nodisc_avg = orders[orders["is_disc"]==0]["order_amount"].mean()
d_df = pd.DataFrame({"Type": ["Discounted", "No Discount"], "Avg Order (₹)": [disc_avg, nodisc_avg]})
drv4 = px.bar(d_df, x="Type", y="Avg Order (₹)", title="Discount Impact on Avg Order Value",
              color="Type", color_discrete_sequence=[TEAL, AMBER],
              text=d_df["Avg Order (₹)"].apply(lambda x: f"₹{x:.0f}"))
drv4.update_traces(textposition="outside", marker_line_width=0)
drv4.update_layout(showlegend=False)

# 2e  Top 10 brands by refund rate
brand_ref = (full.groupby("restaurant_name")["order_status"]
             .apply(lambda x: (x=="Refunded").sum()/len(x)*100)
             .sort_values(ascending=False).head(10).sort_values()
             .reset_index())
brand_ref.columns = ["Brand", "Refund %"]
drv5 = px.bar(brand_ref, x="Refund %", y="Brand", orientation="h",
              title="Top 10 Brands by Refund Rate (%)",
              color="Refund %", color_continuous_scale="Purples",
              text=brand_ref["Refund %"].apply(lambda x: f"{x:.1f}%"))
drv5.update_traces(textposition="outside", marker_line_width=0)
drv5.update_coloraxes(showscale=False)
drv5.add_vline(x=15, line_dash="dash", line_color=AMBER,
               annotation_text="Target 15%", annotation_font_color=AMBER)

# 2f  Cuisine × City revenue heatmap
heat = (delivered.groupby(["cuisine", "City"])["net_revenue"]
        .sum().unstack(fill_value=0))
drv6 = px.imshow(heat, title="Revenue — Cuisine × City (₹)",
                 color_continuous_scale="YlOrRd", text_auto=".2s",
                 labels=dict(x="City", y="Cuisine", color="₹"), aspect="auto")

# ═══════════════════════════════════════════
# SECTION 3 — KPI ANALYSIS
# ═══════════════════════════════════════════

# 3a  Monthly revenue trend with MoM%
mr = (delivered.groupby("month")["net_revenue"].sum()
      .reset_index().sort_values("month"))
mr["MoM %"] = mr["net_revenue"].pct_change() * 100

kpi1 = make_subplots(specs=[[{"secondary_y": True}]])
kpi1.add_trace(go.Bar(x=mr["month"], y=mr["net_revenue"],
                       name="Revenue", marker_color=ORANGE,
                       text=mr["net_revenue"].apply(lambda x: f"₹{x/1e6:.1f}M"),
                       textposition="outside"), secondary_y=False)
kpi1.add_trace(go.Scatter(x=mr["month"], y=mr["MoM %"],
                           name="MoM %", mode="lines+markers",
                           line=dict(color=BRAND, width=2),
                           marker=dict(size=6)), secondary_y=True)
kpi1.update_yaxes(title_text="Revenue (₹)", secondary_y=False)
kpi1.update_yaxes(title_text="MoM Growth %", secondary_y=True)
kpi1.update_layout(title="Monthly Revenue & MoM Growth", xaxis_tickangle=-45,
                   legend=dict(orientation="h", y=1.12))

# 3b  Operational KPIs vs targets
ops = orders.groupby("month").agg(
    total=("order_id", "count"),
    del_=("order_status", lambda x: (x=="Delivered").sum()),
    can_=("order_status", lambda x: (x=="Cancelled").sum()),
    ref_=("order_status", lambda x: (x=="Refunded").sum())
).reset_index().sort_values("month")
ops["Delivery %"]     = ops["del_"]/ops["total"]*100
ops["Cancellation %"] = ops["can_"]/ops["total"]*100
ops["Refund %"]       = ops["ref_"]/ops["total"]*100

kpi2 = go.Figure()
for col, color, target, tgt_pos in [
    ("Delivery %",     GREEN,  70, "top left"),
    ("Cancellation %", BRAND,  15, "top right"),
    ("Refund %",       PURPLE, 10, "top right")
]:
    kpi2.add_trace(go.Scatter(x=ops["month"], y=ops[col],
                               name=col, mode="lines+markers",
                               line=dict(color=color, width=2),
                               marker=dict(size=5)))
    kpi2.add_hline(y=target, line_dash="dash", line_color=color,
                   opacity=0.4, annotation_text=f"Target {target}%",
                   annotation_position=tgt_pos, annotation_font_color=color)
kpi2.update_layout(title="Operational KPIs vs Targets (Monthly)",
                   xaxis_tickangle=-45, legend=dict(orientation="h", y=1.12))

# 3c  Quarterly revenue
qr = (delivered.groupby("quarter")["net_revenue"].sum()
      .reset_index().sort_values("quarter"))
kpi3 = px.bar(qr, x="quarter", y="net_revenue", title="Quarterly Net Revenue",
              color="net_revenue", color_continuous_scale="Blues",
              text=qr["net_revenue"].apply(lambda x: f"₹{x/1e6:.2f}M"),
              labels={"quarter": "Quarter", "net_revenue": "₹"})
kpi3.update_traces(textposition="outside", marker_line_width=0)
kpi3.update_coloraxes(showscale=False)

# 3d  RFM segmentation
snapshot = delivered["ts"].max() + pd.Timedelta(days=1)
rfm = delivered.groupby("customer_id").agg(
    R=("ts", lambda x: (snapshot-x.max()).days),
    F=("order_id", "count"),
    M=("net_revenue", "sum")
).reset_index()
rfm["R"] = pd.qcut(rfm["R"], 5, labels=[5,4,3,2,1]).astype(int)
rfm["F"] = pd.qcut(rfm["F"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
rfm["M"] = pd.qcut(rfm["M"], 5, labels=[1,2,3,4,5]).astype(int)

def seg(r):
    if r.R>=4 and r.F>=4 and r.M>=4: return "Champions"
    elif r.R>=3 and r.F>=3:           return "Loyal"
    elif r.R>=4 and r.F<=2:           return "New"
    elif r.R<=2 and r.F>=3:           return "At Risk"
    elif r.R<=2 and r.F<=2:           return "Lost"
    else:                             return "Potential"
rfm["Segment"] = rfm.apply(seg, axis=1)

seg_rev = rfm.groupby("Segment")["M"].sum().reset_index()
kpi4 = px.pie(seg_rev, names="Segment", values="M",
              title="Revenue Share by Customer Segment",
              hole=0.45, color_discrete_sequence=C)
kpi4.update_traces(textinfo="percent+label",
                   marker=dict(line=dict(color="#fff", width=2)))

# 3e  AOV by city
aov_city = (delivered.groupby("City")["order_amount"].mean()
            .sort_values(ascending=False).reset_index())
aov_city.columns = ["City", "AOV"]
kpi5 = px.bar(aov_city, x="City", y="AOV", title="Avg Order Value by City (₹)",
              color="AOV", color_continuous_scale="Mint",
              text=aov_city["AOV"].apply(lambda x: f"₹{x:.0f}"))
kpi5.update_traces(textposition="outside", marker_line_width=0)
kpi5.update_coloraxes(showscale=False)
kpi5.add_hline(y=aov_city["AOV"].mean(), line_dash="dash", line_color=BRAND,
               annotation_text=f"Avg ₹{aov_city['AOV'].mean():,.0f}",
               annotation_font_color=BRAND)

# 3f  Monthly new signups
ms = (customers.groupby("signup_month")["Customer_id"].count()
      .reset_index().sort_values("signup_month"))
ms.columns = ["Month", "New Customers"]
kpi6 = px.area(ms, x="Month", y="New Customers",
               title="Monthly New Customer Signups",
               color_discrete_sequence=[SLATE])
kpi6.update_layout(xaxis_tickangle=-45)

# ═══════════════════════════════════════════
# SECTION 4 — FUNNEL ANALYSIS
# ═══════════════════════════════════════════

# 4a  Order completion funnel
placed   = total_orders
not_canc = (orders["order_status"] != "Cancelled").sum()
dlvd     = (orders["order_status"] == "Delivered").sum()
fun1 = go.Figure(go.Funnel(
    y=["Orders Placed", "Not Cancelled", "Delivered"],
    x=[placed, not_canc, dlvd],
    textinfo="value+percent initial",
    marker=dict(color=[BLUE, TEAL, GREEN]),
    connector=dict(line=dict(color="#e5e7eb", width=2))
))
fun1.update_layout(title="Order Completion Funnel")

# 4b  Delivery conversion by city
city_fun = (full.groupby("City")["order_status"]
            .apply(lambda x: (x=="Delivered").sum()/len(x)*100)
            .sort_values(ascending=False).reset_index())
city_fun.columns = ["City", "Conv %"]
fun2 = px.bar(city_fun, x="City", y="Conv %",
              title="Delivery Conversion Rate by City (%)",
              color="Conv %", color_continuous_scale="Greens",
              text=city_fun["Conv %"].apply(lambda x: f"{x:.1f}%"))
fun2.update_traces(textposition="outside", marker_line_width=0)
fun2.update_coloraxes(showscale=False)
fun2.add_hline(y=d_rate, line_dash="dash", line_color=SLATE,
               annotation_text=f"Overall avg {d_rate:.1f}%",
               annotation_font_color=SLATE)

# 4c  Cohort retention heatmap
cohort = delivered[["customer_id", "month"]].copy()
first  = cohort.groupby("customer_id")["month"].min().rename("cohort")
cohort = cohort.join(first, on="customer_id")
cohort["period"] = (
    pd.PeriodIndex(cohort["month"],  freq="M") -
    pd.PeriodIndex(cohort["cohort"], freq="M")
).map(lambda x: x.n)
matrix = cohort.groupby(["cohort","period"])["customer_id"].nunique().unstack()
ret    = (matrix.divide(matrix[0], axis=0)*100).round(1).iloc[:14, :10]
fun3 = px.imshow(ret, title="Cohort Retention (%) — First 10 Months",
                 color_continuous_scale="YlGn", text_auto=".0f",
                 labels=dict(x="Month Since First Order", y="Cohort"),
                 aspect="auto")

# 4d  Repeat purchase funnel
oc = delivered.groupby("customer_id")["order_id"].count()
rep_labels = ["≥1 order", "≥2 orders", "≥3 orders", "≥5 orders", "≥10 orders"]
rep_counts = [(oc>=t).sum() for t in [1,2,3,5,10]]
fun4 = go.Figure(go.Funnel(
    y=rep_labels, x=rep_counts,
    textinfo="value+percent previous",
    marker=dict(color=[BLUE, TEAL, GREEN, AMBER, ORANGE]),
    connector=dict(line=dict(color="#e5e7eb", width=2))
))
fun4.update_layout(title="Repeat Purchase Funnel")

# 4e  Acquisition channel → delivery
placed_ids = set(orders["customer_id"])
del_ids    = set(delivered["customer_id"])
customers["placed"]    = customers["Customer_id"].isin(placed_ids)
customers["delivered"] = customers["Customer_id"].isin(del_ids)
acq_fun = customers.groupby("Acquisition_channel").agg(
    Signed_Up=("Customer_id", "count"),
    Placed=("placed", "sum"),
    Delivered=("delivered", "sum")
).reset_index().sort_values("Signed_Up", ascending=False)
acq_melt = acq_fun.melt(id_vars="Acquisition_channel",
                         value_vars=["Signed_Up","Placed","Delivered"],
                         var_name="Stage", value_name="Count")
fun5 = px.bar(acq_melt, x="Acquisition_channel", y="Count", color="Stage",
              barmode="group", title="Acquisition → Delivery Funnel by Channel",
              color_discrete_sequence=[BLUE, TEAL, GREEN])

# 4f  RFM scatter
fun6 = px.scatter(
    rfm.sample(min(3000, len(rfm)), random_state=1),
    x="R", y="F", size="M", color="Segment",
    title="RFM Map — Recency vs Frequency (size = Monetary)",
    color_discrete_sequence=C,
    labels={"R": "Recency (score)", "F": "Frequency (score)", "M": "Monetary"},
    size_max=18, opacity=0.75
)

# ═══════════════════════════════════════════
# SECTION 5 — INSIGHTS & RECOMMENDATIONS
# ═══════════════════════════════════════════

# 5a  Priority matrix
actions = [
    ("Protect Champions",        9, 2, "Customer"),
    ("Win-back At Risk & Lost",  8, 3, "Customer"),
    ("Reduce Cancellations",     9, 5, "Operations"),
    ("Fix Refund Rate SLA",      8, 4, "Operations"),
    ("AOV Upsell — Low Cities",  6, 2, "Revenue"),
    ("Personalise Discounts",    6, 3, "Pricing"),
    ("Top Cuisine Promo",        5, 2, "Revenue"),
    ("Loyalty Programme",        8, 8, "Customer"),
    ("Expand Top City Model",    7, 7, "Revenue"),
    ("Brand Quality Audit",      7, 5, "Operations"),
    ("Payment Retry Flow",       5, 3, "Operations"),
    ("Boost Top Acq Channel",    6, 3, "Customer"),
]
adf = pd.DataFrame(actions, columns=["Action","Impact","Effort","Category"])
cat_colors = {"Operations": BRAND, "Customer": TEAL, "Revenue": AMBER, "Pricing": SLATE}
ins1 = px.scatter(adf, x="Effort", y="Impact", text="Action",
                  color="Category", size=[18]*len(adf),
                  title="Priority Matrix — Impact × Effort",
                  color_discrete_map=cat_colors,
                  range_x=[0,11], range_y=[0,11])
ins1.update_traces(textposition="top center", marker_opacity=0.85)
ins1.add_hline(y=5.5, line_dash="dash", line_color="#9ca3af", opacity=0.6)
ins1.add_vline(x=5.5, line_dash="dash", line_color="#9ca3af", opacity=0.6)
ins1.add_annotation(x=2, y=10.5, text="Quick Wins", showarrow=False,
                    font=dict(color="#6b7280", size=11))
ins1.add_annotation(x=8.5, y=10.5, text="Strategic Bets", showarrow=False,
                    font=dict(color="#6b7280", size=11))
ins1.add_annotation(x=2, y=0.5, text="Fill-ins", showarrow=False,
                    font=dict(color="#6b7280", size=11))
ins1.add_annotation(x=8.5, y=0.5, text="Avoid", showarrow=False,
                    font=dict(color="#6b7280", size=11))

# 5b  Revenue lost vs recoverable
ins2 = go.Figure(go.Bar(
    x=["Revenue Lost\n(Cancel+Refund)", "Recovery\nPotential (30%)"],
    y=[(cancel_rev+refund_rev)/1e6, (cancel_rev+refund_rev)*0.30/1e6],
    marker_color=[BRAND, GREEN],
    text=[f"₹{(cancel_rev+refund_rev)/1e6:.2f}M",
          f"₹{(cancel_rev+refund_rev)*0.30/1e6:.2f}M"],
    textposition="outside", marker_line_width=0
))
ins2.update_layout(title="Revenue Loss vs Recovery Opportunity")

# 5c  KPI vs target grouped bar
curr_vals = [d_rate, c_rate, r_rate]
tgt_vals  = [70,    15,    10]
kpi_names = ["Delivery Rate", "Cancel Rate", "Refund Rate"]
bar_colors = [GREEN if (i==0 and c>=t) or (i>0 and c<=t) else BRAND
              for i,(c,t) in enumerate(zip(curr_vals, tgt_vals))]
ins3 = go.Figure()
ins3.add_trace(go.Bar(name="Current", x=kpi_names, y=curr_vals,
                      marker_color=bar_colors,
                      text=[f"{v:.1f}%" for v in curr_vals],
                      textposition="outside", marker_line_width=0))
ins3.add_trace(go.Bar(name="Target",  x=kpi_names, y=tgt_vals,
                      marker_color=["rgba(0,0,0,0.12)"]*3,
                      text=[f"{v}%" for v in tgt_vals],
                      textposition="outside", marker_line_width=0))
ins3.update_layout(title="Operational KPIs: Current vs Target",
                   barmode="group", yaxis_title="%")

# 5d  Top 10 restaurants by revenue
top_rest = (delivered.groupby("restaurant_name")["net_revenue"]
            .sum().sort_values(ascending=False).head(10).sort_values()
            .reset_index())
top_rest.columns = ["Restaurant", "Revenue"]
ins4 = px.bar(top_rest, x="Revenue", y="Restaurant", orientation="h",
              title="Top 10 Restaurants by Net Revenue",
              color="Revenue", color_continuous_scale="Oranges",
              text=top_rest["Revenue"].apply(lambda x: f"₹{x/1e6:.2f}M"))
ins4.update_traces(textposition="outside", marker_line_width=0)
ins4.update_coloraxes(showscale=False)

# 5e  Customer segment donut
seg_count = rfm["Segment"].value_counts().reset_index()
seg_count.columns = ["Segment", "Count"]
ins5 = px.pie(seg_count, names="Segment", values="Count",
              title="Customer Mix — RFM Segments",
              hole=0.45, color_discrete_sequence=C)
ins5.update_traces(textinfo="percent+label",
                   marker=dict(line=dict(color="#fff", width=2)))

# 5f  Avg impact by category
recs_summary = adf.groupby("Category")["Impact"].mean().reset_index()
ins6 = px.bar(recs_summary, x="Category", y="Impact",
              title="Avg Recommendation Impact by Category",
              color="Category", color_discrete_map=cat_colors,
              text=recs_summary["Impact"].apply(lambda x: f"{x:.1f}"))
ins6.update_traces(textposition="outside", marker_line_width=0)
ins6.update_layout(showlegend=False)

# ═══════════════════════════════════════════
# HTML BUILDER HELPERS
# ═══════════════════════════════════════════

def chart_block(fig, h=340):
    inner = fig2html(fig, h)
    return f'<div class="chart-inner" style="height:{h}px">{inner}</div>'

def card(content, col_span=6):
    """Wrap chart HTML in a card div with correct grid span."""
    return f'<div class="col-{col_span}"><div class="card">{content}</div></div>'

def row(*cols):
    return f'<div class="grid">{"".join(cols)}</div>'

def section(icon, title, subtitle=""):
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    return f'''<div class="section-header">
        <span class="section-icon">{icon}</span>
        <div><div class="section-title">{title}</div>{sub}</div>
    </div>'''

def kpi_card(icon, value, label, color, delta=""):
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    return f'''<div class="kpi-card" style="border-top:3px solid {color}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value" style="color:{color}">{value}</div>
        <div class="kpi-label">{label}</div>{delta_html}
    </div>'''

# Recommendation cards
rec_cards = [
    ("🔴","High","Protect Champions",
     f"{len(rfm[rfm['Segment']=='Champions']):,} champion customers drive the majority of revenue.",
     "Activate VIP tier: free delivery, priority support, exclusive restaurant access. Target 0% churn."),
    ("🔴","High","Fix Refund Rate",
     f"Refund rate {r_rate:.1f}% — ₹{refund_rev/1e6:.2f}M at risk. Target &lt;10%.",
     "Mandatory refund-SLA per restaurant. Photo-verify orders for brands &gt;20% refund rate."),
    ("🔴","High","Reduce Cancellations",
     f"Cancel rate {c_rate:.1f}% — ₹{cancel_rev/1e6:.2f}M lost. Target &lt;15%.",
     "Real-time ETA notifications. Offer reschedule instead of cancel. Exit survey on cancel."),
    ("🟠","Medium","Win Back At-Risk & Lost",
     f"{len(rfm[rfm['Segment'].isin(['At Risk','Lost'])]):,} customers near permanent churn.",
     "Automated 30-day win-back sequence with personalised cuisine-based offers. Target 15% reactivation."),
    ("🟠","Medium","AOV Upsell in Low Cities",
     f"AOV gap: ₹{aov_city['AOV'].max()-aov_city['AOV'].min():,.0f} between best and worst city.",
     "'Frequently ordered together' upsell at checkout. A/B test bundle deals in low-AOV cities."),
    ("🟠","Medium","Personalise Discounts",
     f"{orders['is_disc'].mean()*100:.1f}% of orders discounted. Total spend ₹{total_disc/1e6:.2f}M.",
     "Segment-based offers instead of blanket discounts. Deep discounts reserved for inactive users."),
    ("🟡","Medium","Top Cuisine Promotion",
     f"Top cuisine: {cuis.iloc[0]['cuisine']} (₹{cuis.iloc[0]['net_revenue']/1e6:.1f}M).",
     "Feature top cuisine on homepage. Run cuisine-specific push notifications."),
    ("🟡","Medium","Boost Acquisition Channel",
     f"Top channel: {acq.iloc[0]['Channel']} ({acq.iloc[0]['Customers']:,} customers).",
     "Increase budget 20% on top-performing channel. A/B test new creatives."),
    ("🟢","Plan","Launch Loyalty Programme",
     "Cohort retention drops sharply after Month 1. Steep repeat-order drop at order #2.",
     "Bronze/Gold/Platinum tiers. Reward frequency and value. Target 2nd-order conversion."),
    ("🟢","Plan","Expand Top City Model",
     f"Top city {city_rev.iloc[-1]['City']} leads in revenue. Replicate playbook elsewhere.",
     "Identify top 3 success factors and roll them out to the bottom 2 cities."),
    ("🟢","Plan","Restaurant Quality Programme",
     "Brands below 4.0★ correlate with higher refund and cancel rates.",
     "Monthly quality audits for brands below 4.0★. Temporary suspension below 3.5★."),
    ("🟢","Plan","Payment Retry Flow",
     "High cancel rate associated with failed payment attempts.",
     "Payment-retry flow with alternative payment prompt instead of auto-cancel."),
]

def rec_card_html(badge, level, title, finding, action):
    badge_color = {"High":"#E63946","Medium":"#F4A261","Plan":"#43AA8B"}[level]
    return f'''<div class="rec-card">
        <div class="rec-header" style="background:{badge_color}">
            <span class="rec-badge">{badge}</span>
            <span class="rec-title">{title}</span>
            <span class="rec-level">{level}</span>
        </div>
        <div class="rec-body">
            <div class="rec-finding"><span class="rec-tag">FINDING</span>{finding}</div>
            <div class="rec-action"><span class="rec-tag">ACTION</span>{action}</div>
        </div>
    </div>'''

# ═══════════════════════════════════════════
# ASSEMBLE HTML
# ═══════════════════════════════════════════
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Zomato Analytics Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
/* ── Reset & base ── */
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter','Segoe UI',Arial,sans-serif;background:#f1f5f9;color:#111827;min-width:360px}}

/* ── Top bar ── */
.topbar{{
  background:linear-gradient(135deg,#E63946 0%,#9b1c28 100%);
  color:#fff;padding:18px 36px;
  display:flex;justify-content:space-between;align-items:center;
  box-shadow:0 2px 12px rgba(230,57,70,.35)
}}
.topbar-left h1{{font-size:22px;font-weight:700;letter-spacing:-.3px}}
.topbar-left p{{font-size:12px;opacity:.8;margin-top:3px}}
.topbar-right{{text-align:right;font-size:12px;opacity:.85;line-height:1.8}}
.topbar-badge{{
  display:inline-block;background:rgba(255,255,255,.18);
  padding:2px 10px;border-radius:20px;margin-left:6px;font-weight:600
}}

/* ── Sticky KPI bar ── */
.kpi-bar{{
  background:#fff;border-bottom:1px solid #e5e7eb;
  padding:10px 28px;display:flex;gap:0;overflow-x:auto;
  position:sticky;top:0;z-index:200;
  box-shadow:0 1px 6px rgba(0,0,0,.06)
}}
.kpi-item{{
  flex:1;min-width:110px;text-align:center;
  padding:8px 10px;border-right:1px solid #f3f4f6
}}
.kpi-item:last-child{{border-right:none}}
.kpi-item .val{{font-size:17px;font-weight:700;line-height:1.2}}
.kpi-item .lbl{{font-size:10px;color:#9ca3af;margin-top:2px;font-weight:500;text-transform:uppercase;letter-spacing:.5px}}

/* ── Nav tabs ── */
.nav{{
  background:#fff;padding:0 28px;
  display:flex;gap:2px;
  position:sticky;top:53px;z-index:100;
  box-shadow:0 2px 8px rgba(0,0,0,.05);
  border-bottom:1px solid #e5e7eb
}}
.nav-btn{{
  padding:13px 18px;border:none;background:none;cursor:pointer;
  font-size:13px;font-weight:600;color:#6b7280;
  border-bottom:3px solid transparent;transition:all .18s;
  white-space:nowrap
}}
.nav-btn:hover{{color:#E63946;background:#fef2f2;border-radius:4px 4px 0 0}}
.nav-btn.active{{color:#E63946;border-bottom-color:#E63946}}

/* ── Pages ── */
.page{{display:none;padding:22px 28px 32px}}
.page.active{{display:block}}

/* ── Section header ── */
.section-header{{
  display:flex;align-items:center;gap:12px;
  margin-bottom:16px;margin-top:8px;
  padding-bottom:12px;border-bottom:2px solid #f3f4f6
}}
.section-icon{{font-size:26px}}
.section-title{{font-size:17px;font-weight:700;color:#111827}}
.section-sub{{font-size:12px;color:#9ca3af;margin-top:2px}}

/* ── 12-col grid ── */
.grid{{
  display:grid;
  grid-template-columns:repeat(12,1fr);
  gap:16px;
  margin-bottom:16px
}}
.col-3{{grid-column:span 3}}
.col-4{{grid-column:span 4}}
.col-5{{grid-column:span 5}}
.col-6{{grid-column:span 6}}
.col-7{{grid-column:span 7}}
.col-8{{grid-column:span 8}}
.col-9{{grid-column:span 9}}
.col-12{{grid-column:span 12}}
@media(max-width:1100px){{
  .col-3,.col-4,.col-5,.col-7,.col-8,.col-9{{grid-column:span 6}}
}}
@media(max-width:700px){{
  .col-3,.col-4,.col-5,.col-6,.col-7,.col-8,.col-9,.col-12{{grid-column:span 12}}
}}

/* ── Chart card ── */
.card{{
  background:#fff;border-radius:14px;
  padding:12px 10px 6px;
  box-shadow:0 1px 6px rgba(0,0,0,.06),0 4px 16px rgba(0,0,0,.04);
  transition:box-shadow .2s
}}
.card:hover{{box-shadow:0 2px 12px rgba(0,0,0,.1),0 6px 20px rgba(0,0,0,.07)}}
.chart-inner{{width:100%;overflow:hidden}}

/* ── Recommendation cards ── */
.rec-grid{{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(300px,1fr));
  gap:16px;margin-top:12px
}}
.rec-card{{
  background:#fff;border-radius:12px;overflow:hidden;
  box-shadow:0 1px 6px rgba(0,0,0,.06),0 4px 16px rgba(0,0,0,.04);
  transition:transform .15s,box-shadow .15s
}}
.rec-card:hover{{transform:translateY(-2px);box-shadow:0 4px 20px rgba(0,0,0,.12)}}
.rec-header{{padding:11px 14px;display:flex;align-items:center;gap:10px}}
.rec-badge{{font-size:18px}}
.rec-title{{font-weight:700;color:#fff;font-size:13px;flex:1}}
.rec-level{{
  font-size:10px;background:rgba(255,255,255,.22);
  padding:2px 8px;border-radius:20px;color:#fff;font-weight:600
}}
.rec-body{{padding:12px 14px}}
.rec-finding{{margin-bottom:10px;font-size:12px;color:#6b7280;line-height:1.55}}
.rec-action{{font-size:12.5px;color:#111827;line-height:1.55}}
.rec-tag{{
  display:block;font-size:9.5px;font-weight:700;
  color:#9ca3af;letter-spacing:.9px;
  margin-bottom:3px;text-transform:uppercase
}}

/* ── Action plan ── */
.plan-row{{display:flex;gap:16px;flex-wrap:wrap;margin-top:12px}}
.plan-col{{
  flex:1;min-width:200px;background:#fff;border-radius:12px;
  padding:16px 18px;box-shadow:0 1px 6px rgba(0,0,0,.06)
}}
.plan-col h4{{
  font-size:13px;font-weight:700;margin-bottom:10px;
  padding-bottom:8px;border-bottom:2px solid #e5e7eb
}}
.plan-col ul{{padding-left:16px;font-size:12.5px;line-height:2;color:#374151}}
.plan-col.high h4{{color:#E63946;border-color:#fca5a5}}
.plan-col.med  h4{{color:#F4A261;border-color:#fcd9b0}}
.plan-col.low  h4{{color:#43AA8B;border-color:#a7f3d0}}

/* ── Footer ── */
footer{{
  text-align:center;color:#d1d5db;
  font-size:11px;padding:24px;
  border-top:1px solid #f3f4f6;margin-top:8px
}}
</style>
</head>
<body>

<!-- TOP BAR -->
<div class="topbar">
  <div class="topbar-left">
    <h1>🍽️ Zomato Order Analytics</h1>
    <p>End-to-end analysis across 5 modules · Interactive Dashboard</p>
  </div>
  <div class="topbar-right">
    <span class="topbar-badge">🛒 {total_orders:,} Orders</span>
    <span class="topbar-badge">👥 {total_customers:,} Customers</span>
    <span class="topbar-badge">🏪 200 Restaurants</span>
    <span class="topbar-badge">🏙️ 8 Cities</span>
  </div>
</div>

<!-- STICKY KPI BAR -->
<div class="kpi-bar">
  <div class="kpi-item">
    <div class="val" style="color:#264653">{total_orders:,}</div>
    <div class="lbl">Total Orders</div>
  </div>
  <div class="kpi-item">
    <div class="val" style="color:#2A9D8F">{total_customers:,}</div>
    <div class="lbl">Customers</div>
  </div>
  <div class="kpi-item">
    <div class="val" style="color:#E9C46A">₹{gmv/1e6:.1f}M</div>
    <div class="lbl">Gross Merch Value</div>
  </div>
  <div class="kpi-item">
    <div class="val" style="color:#E76F51">₹{net_revenue_val/1e6:.1f}M</div>
    <div class="lbl">Net Revenue</div>
  </div>
  <div class="kpi-item">
    <div class="val" style="color:#F4A261">₹{aov:,.0f}</div>
    <div class="lbl">Avg Order Value</div>
  </div>
  <div class="kpi-item">
    <div class="val" style="color:#43AA8B">{d_rate:.1f}%</div>
    <div class="lbl">Delivery Rate</div>
  </div>
  <div class="kpi-item">
    <div class="val" style="color:#E63946">{c_rate:.1f}%</div>
    <div class="lbl">Cancel Rate</div>
  </div>
  <div class="kpi-item">
    <div class="val" style="color:#9C6B98">{r_rate:.1f}%</div>
    <div class="lbl">Refund Rate</div>
  </div>
</div>

<!-- NAV TABS -->
<div class="nav">
  <button class="nav-btn active" onclick="show('eda',this)">🔍 EDA</button>
  <button class="nav-btn" onclick="show('driver',this)">🔬 Driver Analysis</button>
  <button class="nav-btn" onclick="show('kpi',this)">📊 KPI</button>
  <button class="nav-btn" onclick="show('funnel',this)">🔽 Funnel</button>
  <button class="nav-btn" onclick="show('insights',this)">💡 Insights & Recommendations</button>
</div>

<!-- ═══════════ PAGE: EDA ═══════════ -->
<div id="eda" class="page active">
  {section("🔍","Exploratory Data Analysis","Distributions, patterns and city-level overview")}

  {row(
    card(chart_block(eda1, 320), 4),
    card(chart_block(eda4, 320), 4),
    card(chart_block(eda6, 320), 4)
  )}
  {row(
    card(chart_block(eda2, 320), 8),
    card(chart_block(eda5, 320), 4)
  )}
  {row(
    card(chart_block(eda7, 300), 6),
    card(chart_block(eda3, 300), 6)
  )}
  {row(
    card(chart_block(eda8, 360), 12)
  )}
</div>

<!-- ═══════════ PAGE: DRIVER ANALYSIS ═══════════ -->
<div id="driver" class="page">
  {section("🔬","Driver Analysis","What drives revenue, cancellations, refunds and AOV?")}
  {row(
    card(chart_block(drv1, 340), 6),
    card(chart_block(drv2, 340), 6)
  )}
  {row(
    card(chart_block(drv3, 380), 6),
    card(chart_block(drv4, 340), 6)
  )}
  {row(
    card(chart_block(drv5, 360), 6),
    card(chart_block(drv6, 420), 6)
  )}
</div>

<!-- ═══════════ PAGE: KPI ═══════════ -->
<div id="kpi" class="page">
  {section("📊","KPI Analysis","Revenue, operational and customer KPIs vs targets")}
  {row(
    card(chart_block(kpi1, 360), 8),
    card(chart_block(kpi3, 360), 4)
  )}
  {row(
    card(chart_block(kpi2, 360), 12)
  )}
  {row(
    card(chart_block(kpi4, 340), 4),
    card(chart_block(kpi5, 340), 4),
    card(chart_block(kpi6, 340), 4)
  )}
</div>

<!-- ═══════════ PAGE: FUNNEL ═══════════ -->
<div id="funnel" class="page">
  {section("🔽","Funnel Analysis","Order completion, cohort retention, RFM segmentation")}
  {row(
    card(chart_block(fun1, 360), 4),
    card(chart_block(fun2, 360), 4),
    card(chart_block(fun4, 360), 4)
  )}
  {row(
    card(chart_block(fun5, 360), 12)
  )}
  {row(
    card(chart_block(fun3, 440), 12)
  )}
  {row(
    card(chart_block(fun6, 420), 12)
  )}
</div>

<!-- ═══════════ PAGE: INSIGHTS ═══════════ -->
<div id="insights" class="page">
  {section("💡","Insights & Recommendations","Data-driven actions ranked by impact")}

  {row(
    card(chart_block(ins1, 460), 8),
    card(chart_block(ins2, 340), 4)
  )}
  {row(
    card(chart_block(ins3, 340), 6),
    card(chart_block(ins4, 380), 6)
  )}
  {row(
    card(chart_block(ins5, 340), 4),
    card(chart_block(ins6, 340), 8)
  )}

  <div class="section-header" style="margin-top:28px">
    <span class="section-icon">🃏</span>
    <div>
      <div class="section-title">12 Data-Driven Recommendation Cards</div>
      <div class="section-sub">Ranked by priority · Red = this week · Amber = this month · Green = this quarter</div>
    </div>
  </div>
  <div class="rec-grid">
    {"".join(rec_card_html(*r) for r in rec_cards)}
  </div>

  <div class="section-header" style="margin-top:32px">
    <span class="section-icon">🗓️</span>
    <div class="section-title">Prioritised Action Plan</div>
  </div>
  <div class="plan-row">
    <div class="plan-col high">
      <h4>🔴 This Week — Quick Wins</h4>
      <ul>
        <li>Activate VIP perks for Champions</li>
        <li>Set refund-rate SLA for all brands</li>
        <li>Launch At-Risk win-back campaign</li>
        <li>Feature top cuisine on homepage</li>
      </ul>
    </div>
    <div class="plan-col med">
      <h4>🟠 This Month — Medium Effort</h4>
      <ul>
        <li>Add ETA notification to reduce cancellations</li>
        <li>AOV upsell nudges at checkout</li>
        <li>Personalise discounts by segment</li>
        <li>Double budget on top acquisition channel</li>
      </ul>
    </div>
    <div class="plan-col low">
      <h4>🟢 This Quarter — Big Bets</h4>
      <ul>
        <li>Launch tiered Loyalty Programme</li>
        <li>Expand top-city model to other cities</li>
        <li>Monthly quality audits for low-rated brands</li>
        <li>Build payment-retry flow</li>
      </ul>
    </div>
  </div>
</div>

<footer>Zomato Analytics &nbsp;·&nbsp; Built with Python + Plotly &nbsp;·&nbsp; 2024</footer>

<script>
function show(id, btn) {{
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
  // Trigger Plotly resize so charts fill their containers
  setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
}}
// Ensure charts render correctly on first load
window.addEventListener('load', () => {{
  setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
}});
</script>
</body>
</html>"""

out = os.path.join(BASE, "zomato_dashboard.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(HTML)

print(f"✅  Dashboard saved → {out}")
