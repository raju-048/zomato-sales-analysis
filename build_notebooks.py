"""
Build all 5 Zomato analysis notebooks.
Run: python build_notebooks.py
"""
import nbformat as nbf, os

BASE = os.path.dirname(os.path.abspath(__file__))

def nb():
    n = nbf.v4.new_notebook()
    n.metadata = {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
                  "language_info": {"name":"python","version":"3.12.0"}}
    return n

md   = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

def save(notebook, fname):
    with open(os.path.join(BASE, fname), "w", encoding="utf-8") as f:
        nbf.write(notebook, f)
    print(f"  ✅  {fname}")

# ── Shared setup injected into every notebook ──────────────────────────────
SETUP = (
'import os, warnings\nimport pandas as pd\nimport numpy as np\n'
'import matplotlib.pyplot as plt\nimport matplotlib.ticker as mticker\nimport seaborn as sns\n'
'from scipy import stats\nwarnings.filterwarnings("ignore")\n'
'sns.set_theme(style="whitegrid", palette="Set2")\n'
'plt.rcParams.update({"figure.dpi":110,"figure.figsize":(13,5),"axes.titlesize":13})\n\n'
'BASE = r"' + BASE + '"\n'
'def load(n): return pd.read_csv(os.path.join(BASE, f"Zomato  Order Data.xlsx - {n}.csv"))\n'
'customers, orders, restaurants = load("Customer"), load("Orders"), load("Restaurants")\n\n'
'orders["order_timestamp"] = pd.to_datetime(orders["order_timestamp"], format="%m/%d/%Y", errors="coerce")\n'
'orders["order_month"]   = orders["order_timestamp"].dt.to_period("M")\n'
'orders["order_quarter"] = orders["order_timestamp"].dt.to_period("Q")\n'
'orders["order_year"]    = orders["order_timestamp"].dt.year.astype("Int64")\n'
'orders["day_of_week"]   = orders["order_timestamp"].dt.day_name()\n'
'orders["discount_amount"] = orders["discount_amount"].fillna(0)\n'
'orders["delivery_fee"]    = orders["delivery_fee"].fillna(0)\n'
'orders["net_revenue"]     = orders["order_amount"] - orders["discount_amount"]\n'
'orders["is_discounted"]   = (orders["discount_amount"] > 0).astype(int)\n'
'customers["Signup_Time"]  = pd.to_datetime(customers["Signup_Time"], format="%d/%m/%Y", errors="coerce")\n'
'customers["signup_month"] = customers["Signup_Time"].dt.to_period("M")\n'
'full = (orders.merge(restaurants, on="restaurant_id", how="left")\n'
'              .merge(customers, left_on="customer_id", right_on="Customer_id", how="left"))\n'
'delivered = full[full["order_status"]=="Delivered"].copy()\n'
'cancelled = full[full["order_status"]=="Cancelled"].copy()\n'
'refunded  = full[full["order_status"]=="Refunded"].copy()\n'
'print(f"Orders:{len(orders):,} | Customers:{customers[\'Customer_id\'].nunique():,} | Restaurants:{len(restaurants)}")\n'
'print(f"Date range: {orders[\'order_timestamp\'].min().date()} to {orders[\'order_timestamp\'].max().date()}")\n'
)

# ════════════════════════════════════════════════════════════
# NB1 — EXPLORATORY DATA ANALYSIS
# ════════════════════════════════════════════════════════════
n1 = nb()
n1.cells = [
md("# 🔍 EDA — Exploratory Data Analysis\nCovers: data quality · univariate · bivariate · multivariate · temporal · city-level."),
code(SETUP),

md("## 1. Data Quality"),
code('''\
print(orders[["order_amount","discount_amount","delivery_fee","net_revenue"]].describe().T.round(2))
print("\\nMissing values:\\n", orders.isnull().sum()[orders.isnull().sum()>0].to_string())
print("\\nDuplicate order_ids:", orders["order_id"].duplicated().sum())
print("Unique statuses:", orders["order_status"].unique())
print("Unique payment modes:", orders["payment_mode"].unique())
'''),

md("## 2. Univariate Distributions"),
code('''\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(orders["order_amount"],  bins=40, color="#264653", edgecolor="white"); axes[0].set_title("Order Amount")
axes[1].hist(orders["delivery_fee"],  bins=30, color="#2A9D8F", edgecolor="white"); axes[1].set_title("Delivery Fee")
axes[2].hist(restaurants["avg_rating"], bins=20, color="#457B9D", edgecolor="white"); axes[2].set_title("Restaurant Rating")
plt.suptitle("Financial & Rating Distributions", fontweight="bold"); plt.tight_layout(); plt.show()
'''),
code('''\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, title in zip(axes,
    ["order_status","payment_mode","day_of_week"],
    ["Order Status","Payment Mode","Day of Week"]):
    vc = orders[col].value_counts()
    ax.bar(vc.index, vc.values, color=sns.color_palette("Set2", len(vc)))
    ax.set_title(title); ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(vc.values): ax.text(i, v+30, f"{v:,}", ha="center", fontsize=9)
plt.suptitle("Order Behaviour", fontweight="bold"); plt.tight_layout(); plt.show()
'''),

md("## 3. Bivariate Analysis"),
code('''\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
city_rev = delivered.groupby("City")["net_revenue"].sum().sort_values()
axes[0].barh(city_rev.index, city_rev/1e6, color=sns.color_palette("teal", len(city_rev)))
axes[0].set_title("Net Revenue by City (₹M)"); axes[0].set_xlabel("₹M")
for i, v in enumerate(city_rev/1e6): axes[0].text(v+.02, i, f"₹{v:.2f}M", va="center", fontsize=9)

cancel_city = full.groupby("City")["order_status"].apply(lambda x: (x=="Cancelled").sum()/len(x)*100)
axes[1].bar(cancel_city.index, cancel_city.values, color=sns.color_palette("Reds_r", len(cancel_city)))
axes[1].axhline(15, color="orange", linestyle="--", label="Target 15%")
axes[1].set_title("Cancellation Rate by City (%)"); axes[1].legend()
for i, v in enumerate(cancel_city): axes[1].text(i, v+.2, f"{v:.1f}%", ha="center", fontsize=9)
plt.suptitle("Revenue & Cancellation by City", fontweight="bold"); plt.tight_layout(); plt.show()
'''),
code('''\
rest = full.groupby("restaurant_name").agg(revenue=("net_revenue","sum"), orders=("order_id","count"), rating=("avg_rating","mean")).reset_index()
fig, ax = plt.subplots(figsize=(9, 5))
sc = ax.scatter(rest["rating"], rest["revenue"]/1e6, s=rest["orders"]/8, alpha=0.6,
                c=rest["revenue"], cmap="YlOrRd", edgecolors="grey", linewidths=0.4)
plt.colorbar(sc, ax=ax, label="Revenue (₹M)")
ax.set_xlabel("Avg Rating"); ax.set_ylabel("Net Revenue (₹M)")
ax.set_title("Rating vs Revenue (bubble = order count)", fontweight="bold")
print("Correlation:\\n", rest[["rating","revenue","orders"]].corr().round(3))
plt.tight_layout(); plt.show()
'''),

md("## 4. Multivariate Analysis"),
code('''\
pivot = delivered.groupby(["cuisine","City"])["net_revenue"].sum().unstack(fill_value=0)/1e6
fig, ax = plt.subplots(figsize=(13, 6))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.4, ax=ax, cbar_kws={"label":"₹M"})
ax.set_title("Revenue Heatmap — Cuisine × City (₹M)", fontweight="bold"); plt.tight_layout(); plt.show()
'''),
code('''\
pivot2 = orders.groupby(["payment_mode","order_status"])["order_id"].count().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(pivot2, annot=True, fmt=",", cmap="Blues", linewidths=0.4, ax=ax)
ax.set_title("Order Count — Payment Mode × Status", fontweight="bold"); plt.tight_layout(); plt.show()
'''),

md("## 5. Temporal Analysis"),
code('''\
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
mo = orders.groupby("order_month")["order_id"].count()
axes[0].plot(mo.index.astype(str), mo.values, marker="o", color="#264653", linewidth=2)
axes[0].set_title("Monthly Order Volume"); axes[0].tick_params(axis="x", rotation=60)

mr = delivered.groupby("order_month")["net_revenue"].sum()
mr_df = mr.reset_index(); mr_df["mom"] = mr_df["net_revenue"].pct_change()*100
mr_df["order_month"] = mr_df["order_month"].astype(str)
colors = ["#2A9D8F" if v>=0 else "#E63946" for v in mr_df["mom"].fillna(0)]
axes[1].bar(mr_df["order_month"], mr_df["mom"].fillna(0), color=colors)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("MoM Revenue Growth (%)"); axes[1].tick_params(axis="x", rotation=60)
plt.suptitle("Temporal Trends", fontweight="bold"); plt.tight_layout(); plt.show()
'''),

md("## 6. City-Level Summary"),
code('''\
city_sum = full.groupby("City").agg(
    orders=("order_id","count"),
    revenue=("net_revenue","sum"),
    aov=("order_amount","mean"),
    delivery_rate=("order_status", lambda x: (x=="Delivered").sum()/len(x)*100),
    cancel_rate=("order_status",   lambda x: (x=="Cancelled").sum()/len(x)*100),
    avg_rating=("avg_rating","mean")
).round(2)
display(city_sum.sort_values("revenue", ascending=False))
'''),

md("## ✅ Key EDA Findings\n- 50K orders, 4,999 customers, 200 restaurants, 8 cities\n- Delivery ~60%, cancel ~20%, refund ~20% — non-delivery is too high\n- Revenue is top-heavy: few cities & cuisines dominate\n- Order amounts uniformly distributed ₹300–₹1,500 with no strong outliers"),
]
save(n1, "01_EDA.ipynb")

# ════════════════════════════════════════════════════════════
# NB2 — DRIVER ANALYSIS
# ════════════════════════════════════════════════════════════
n2 = nb()
n2.cells = [
md("# 🔬 Driver Analysis\nCovers: revenue drivers · cancellation drivers · refund drivers · AOV · CLV · feature importance."),
code(SETUP),

md("## 1. Revenue Correlation"),
code('''\
from scipy import stats
num_cols = ["order_amount","discount_amount","delivery_fee","avg_rating","is_discounted"]
corr = delivered[num_cols+["net_revenue"]].corr()["net_revenue"].drop("net_revenue").sort_values()
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(corr.index, corr.values, color=["#E63946" if v<0 else "#2A9D8F" for v in corr.values])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Feature Correlation with Net Revenue", fontweight="bold")
for i, v in enumerate(corr.values): ax.text(v+(0.003 if v>=0 else -0.003), i, f"{v:.3f}", va="center", fontsize=9)
plt.tight_layout(); plt.show()
'''),

md("## 2. Discount Impact"),
code('''\
disc = orders[orders["is_discounted"]==1]["order_amount"]
nodisc = orders[orders["is_discounted"]==0]["order_amount"]
t, p = stats.ttest_ind(disc, nodisc)
print(f"Discounted mean: ₹{disc.mean():,.0f} | Non-disc mean: ₹{nodisc.mean():,.0f}")
print(f"Lift: ₹{disc.mean()-nodisc.mean():,.0f} | t-test p={p:.4f} ({'Significant' if p<0.05 else 'Not significant'})")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].boxplot([disc, nodisc], labels=["Discounted","No Discount"], patch_artist=True,
                boxprops=dict(facecolor="#2A9D8F"), medianprops=dict(color="red"))
axes[0].set_title("Order Amount Distribution"); axes[0].set_ylabel("₹")

depth = orders[orders["discount_amount"]>0]["discount_amount"]/orders[orders["discount_amount"]>0]["order_amount"]*100
axes[1].hist(depth, bins=20, color="#E9C46A", edgecolor="white")
axes[1].axvline(depth.mean(), color="red", linestyle="--", label=f"Mean {depth.mean():.1f}%")
axes[1].set_title("Discount Depth Distribution (%)"); axes[1].legend()
plt.suptitle("Discount Effectiveness", fontweight="bold"); plt.tight_layout(); plt.show()
'''),

md("## 3. Cancellation Drivers"),
code('''\
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, dim in zip(axes, ["City","cuisine","payment_mode"]):
    df = (full.groupby(dim)["order_status"]
          .apply(lambda x: (x=="Cancelled").sum()/len(x)*100)
          .sort_values(ascending=True).reset_index())
    df.columns = [dim, "Rate"]
    ax.barh(df[dim].astype(str), df["Rate"], color=sns.color_palette("Reds_r", len(df)))
    ax.set_title(f"Cancel Rate by {dim}", fontweight="bold")
    for i, v in enumerate(df["Rate"]): ax.text(v+.1, i, f"{v:.1f}%", va="center", fontsize=8)
plt.suptitle("Cancellation Drivers", fontweight="bold"); plt.tight_layout(); plt.show()
'''),
code('''\
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

df = full[["order_amount","discount_amount","delivery_fee","is_discounted","avg_rating",
           "payment_mode","cuisine","City","day_of_week","order_status"]].dropna().copy()
df["cancelled"] = (df["order_status"]=="Cancelled").astype(int)
for c in ["payment_mode","cuisine","City","day_of_week"]:
    df[c] = LabelEncoder().fit_transform(df[c].astype(str))
feats = ["order_amount","discount_amount","delivery_fee","is_discounted","avg_rating",
         "payment_mode","cuisine","City","day_of_week"]
X_tr, X_te, y_tr, y_te = train_test_split(df[feats], df["cancelled"], test_size=0.2, random_state=42)
lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_tr, y_tr)
coef = pd.Series(lr.coef_[0], index=feats).sort_values()
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(coef.index, coef.values, color=["#E63946" if v>0 else "#2A9D8F" for v in coef.values])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title(f"Cancellation Predictors (ROC-AUC={roc_auc_score(y_te,lr.predict_proba(X_te)[:,1]):.3f})", fontweight="bold")
plt.tight_layout(); plt.show()
'''),

md("## 4. Refund Drivers"),
code('''\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, dim in zip(axes, ["restaurant_name","cuisine"]):
    df = (full.groupby(dim)["order_status"]
          .apply(lambda x: (x=="Refunded").sum()/len(x)*100)
          .sort_values(ascending=False).head(10).sort_values().reset_index())
    df.columns = [dim, "Rate"]
    ax.barh(df[dim].astype(str), df["Rate"], color=sns.color_palette("Purples_r", len(df)))
    ax.set_title(f"Refund Rate by {dim} (top 10)", fontweight="bold")
    for i, v in enumerate(df["Rate"]): ax.text(v+.1, i, f"{v:.1f}%", va="center", fontsize=9)
refund_rev = full[full["order_status"]=="Refunded"]["order_amount"].sum()
print(f"Revenue lost to refunds: ₹{refund_rev:,.0f} ({refund_rev/full['order_amount'].sum()*100:.2f}% of GMV)")
plt.suptitle("Refund Drivers", fontweight="bold"); plt.tight_layout(); plt.show()
'''),

md("## 5. AOV & CLV Drivers"),
code('''\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
aov_city = delivered.groupby("City")["order_amount"].mean().sort_values()
axes[0].barh(aov_city.index, aov_city.values, color=sns.color_palette("teal", len(aov_city)))
axes[0].set_title("AOV by City"); axes[0].set_xlabel("₹")
for i, v in enumerate(aov_city): axes[0].text(v+2, i, f"₹{v:,.0f}", va="center", fontsize=9)

clv = delivered.groupby("customer_id")["net_revenue"].sum()
axes[1].hist(clv, bins=40, color="#457B9D", edgecolor="white")
axes[1].axvline(clv.mean(), color="red", linestyle="--", label=f"Mean ₹{clv.mean():,.0f}")
axes[1].set_title("Customer Lifetime Value Distribution"); axes[1].legend()
top10pct = clv.nlargest(int(len(clv)*.1)).sum()/clv.sum()*100
print(f"Top 10% customers drive {top10pct:.1f}% of revenue")
plt.suptitle("AOV & CLV", fontweight="bold"); plt.tight_layout(); plt.show()
'''),

md("## 6. Feature Importance (Random Forest)"),
code('''\
from sklearn.ensemble import RandomForestRegressor
df_rf = delivered[["order_amount","discount_amount","delivery_fee","is_discounted","avg_rating",
                    "payment_mode","cuisine","City","day_of_week","net_revenue"]].dropna().copy()
for c in ["payment_mode","cuisine","City","day_of_week"]:
    df_rf[c] = LabelEncoder().fit_transform(df_rf[c].astype(str))
feats = ["order_amount","discount_amount","delivery_fee","is_discounted","avg_rating",
         "payment_mode","cuisine","City","day_of_week"]
X_tr, X_te, y_tr, y_te = train_test_split(df_rf[feats], df_rf["net_revenue"], test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
imp = pd.Series(rf.feature_importances_, index=feats).sort_values()
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(imp.index, imp.values, color=sns.color_palette("YlOrRd_r", len(imp)))
ax.set_title(f"RF Feature Importance — Revenue (R²={rf.score(X_te,y_te):.3f})", fontweight="bold")
plt.tight_layout(); plt.show()
'''),

md("## ✅ Driver Findings\n- `order_amount` is the dominant revenue driver (RF importance + correlation)\n- Discount lift is statistically significant but small in practice\n- City and cuisine are the strongest categorical drivers for cancellations\n- Top 10% customers drive disproportionate revenue — retention is critical"),
]
save(n2, "02_Driver_Analysis.ipynb")

# ════════════════════════════════════════════════════════════
# NB3 — KPI ANALYSIS
# ════════════════════════════════════════════════════════════
n3 = nb()
n3.cells = [
md("# 📊 KPI Analysis\nCovers: scorecard · revenue KPIs · operational KPIs · customer KPIs · restaurant KPIs · city×month heatmap."),
code(SETUP),

md("## 1. KPI Scorecard"),
code('''\
total_orders    = len(orders)
total_customers = customers["Customer_id"].nunique()
gmv             = orders["order_amount"].sum()
net_revenue     = delivered["net_revenue"].sum()
aov             = delivered["order_amount"].mean()
delivery_rate   = (orders["order_status"]=="Delivered").mean()*100
cancel_rate     = (orders["order_status"]=="Cancelled").mean()*100
refund_rate     = (orders["order_status"]=="Refunded").mean()*100
total_discounts = orders["discount_amount"].sum()
repeat_rate     = (delivered.groupby("customer_id")["order_id"].count()>1).sum()/total_customers*100

kpis = [
    ("Total Orders",          f"{total_orders:,}",         "🛒"),
    ("Unique Customers",      f"{total_customers:,}",      "👥"),
    ("GMV",                   f"₹{gmv/1e6:.2f}M",         "💳"),
    ("Net Revenue",           f"₹{net_revenue/1e6:.2f}M",  "💰"),
    ("Avg Order Value",       f"₹{aov:,.0f}",              "🧾"),
    ("Delivery Rate",         f"{delivery_rate:.1f}%",     "✅"),
    ("Cancellation Rate",     f"{cancel_rate:.1f}%",       "❌"),
    ("Refund Rate",           f"{refund_rate:.1f}%",       "↩️"),
    ("Repeat Customer Rate",  f"{repeat_rate:.1f}%",       "🔄"),
    ("Total Discounts",       f"₹{total_discounts/1e6:.2f}M","🏷️"),
]
print(f"{'KPI':<26} {'VALUE':>12}  ICON")
print("="*45)
for k, v, i in kpis: print(f"  {k:<24} {v:>12}  {i}")
'''),

md("## 2. Revenue KPIs — Trends"),
code('''\
m = delivered.groupby("order_month").agg(
    net_rev=("net_revenue","sum"), orders=("order_id","count"),
    aov=("order_amount","mean"), customers=("customer_id","nunique")).reset_index()
m["order_month"] = m["order_month"].astype(str)
m["mom_growth"]  = m["net_rev"].pct_change()*100

fig, axes = plt.subplots(2, 2, figsize=(15, 8))
for ax, col, label, color in zip(axes.flat,
    ["net_rev","aov","customers","mom_growth"],
    ["Net Revenue (₹)","Avg Order Value (₹)","Active Customers","MoM Growth (%)"],
    ["#264653","#2A9D8F","#E9C46A","#E76F51"]):
    ax.plot(range(len(m)), m[col].fillna(0), marker="o", color=color, linewidth=2)
    ax.set_xticks(range(len(m))); ax.set_xticklabels(m["order_month"], rotation=60, fontsize=7)
    ax.set_title(label, fontweight="bold")
    if col == "mom_growth": ax.axhline(0, color="black", linewidth=0.8)
plt.suptitle("Monthly Revenue KPI Trends", fontweight="bold"); plt.tight_layout(); plt.show()
'''),

md("## 3. Operational KPIs vs Targets"),
code('''\
ops = orders.groupby("order_month").agg(
    total=("order_id","count"),
    delivered=("order_status", lambda x: (x=="Delivered").sum()),
    cancelled=("order_status", lambda x: (x=="Cancelled").sum()),
    refunded=("order_status",  lambda x: (x=="Refunded").sum())).reset_index()
ops["order_month"]   = ops["order_month"].astype(str)
ops["delivery_rate"] = ops["delivered"]/ops["total"]*100
ops["cancel_rate"]   = ops["cancelled"]/ops["total"]*100
ops["refund_rate"]   = ops["refunded"]/ops["total"]*100

fig, ax = plt.subplots(figsize=(14, 5))
x = range(len(ops))
ax.plot(x, ops["delivery_rate"], marker="o", label="Delivery %",  color="#43AA8B", linewidth=2)
ax.plot(x, ops["cancel_rate"],   marker="s", label="Cancel %",    color="#E63946", linewidth=2)
ax.plot(x, ops["refund_rate"],   marker="^", label="Refund %",    color="#9C6B98", linewidth=2)
ax.axhline(70, color="#43AA8B", linestyle="--", alpha=0.4, label="Delivery target 70%")
ax.axhline(15, color="#E63946", linestyle="--", alpha=0.4, label="Cancel target 15%")
ax.axhline(10, color="#9C6B98", linestyle="--", alpha=0.4, label="Refund target 10%")
ax.set_xticks(x); ax.set_xticklabels(ops["order_month"], rotation=60, fontsize=8)
ax.set_title("Operational KPI Trends vs Targets", fontweight="bold"); ax.legend(ncol=3, fontsize=9)
plt.tight_layout(); plt.show()
'''),

md("## 4. Customer KPIs — RFM Segments"),
code('''\
snapshot = delivered["order_timestamp"].max() + pd.Timedelta(days=1)
rfm = delivered.groupby("customer_id").agg(
    recency  =("order_timestamp", lambda x: (snapshot-x.max()).days),
    frequency=("order_id","count"), monetary=("net_revenue","sum")).reset_index()
rfm["R"] = pd.qcut(rfm["recency"],  5, labels=[5,4,3,2,1]).astype(int)
rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
rfm["M"] = pd.qcut(rfm["monetary"], 5, labels=[1,2,3,4,5]).astype(int)
def seg(r):
    if r.R>=4 and r.F>=4 and r.M>=4: return "Champions"
    elif r.R>=3 and r.F>=3:           return "Loyal"
    elif r.R>=4 and r.F<=2:           return "New"
    elif r.R<=2 and r.F>=3:           return "At Risk"
    elif r.R<=2 and r.F<=2:           return "Lost"
    else:                             return "Potential"
rfm["Segment"] = rfm.apply(seg, axis=1)
seg_stats = rfm.groupby("Segment").agg(customers=("customer_id","count"), avg_rev=("monetary","mean"), total_rev=("monetary","sum")).sort_values("total_rev", ascending=False)
display(seg_stats.round(2))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].pie(seg_stats["customers"], labels=seg_stats.index, autopct="%1.1f%%", colors=sns.color_palette("Set2",len(seg_stats)))
axes[0].set_title("Customer Mix")
axes[1].bar(seg_stats.index, seg_stats["total_rev"]/1e6, color=sns.color_palette("Set2",len(seg_stats)))
axes[1].set_title("Revenue by Segment (₹M)"); axes[1].tick_params(axis="x", rotation=20)
plt.suptitle("RFM Customer Segments", fontweight="bold"); plt.tight_layout(); plt.show()
'''),

md("## 5. Restaurant KPIs"),
code('''\
rest_kpi = full.groupby("restaurant_name").agg(
    orders=("order_id","count"),
    revenue=("net_revenue","sum"),
    rating=("avg_rating","mean"),
    delivery_rate=("order_status", lambda x: (x=="Delivered").sum()/len(x)*100),
    refund_rate=("order_status",   lambda x: (x=="Refunded").sum()/len(x)*100)
).round(2).sort_values("revenue", ascending=False)
display(rest_kpi.head(10))
'''),

md("## 6. City × Month KPI Heatmap"),
code('''\
for metric, label in [
    (lambda g: (g["order_status"]=="Delivered").sum()/len(g)*100, "Delivery Rate (%)"),
    (lambda g: (g["order_status"]=="Cancelled").sum()/len(g)*100, "Cancel Rate (%)")]:
    pivot = full.groupby(["City","order_month"]).apply(metric).unstack()
    pivot.columns = pivot.columns.astype(str)
    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd" if "Cancel" in label else "YlGn",
                linewidths=0.3, ax=ax, cbar_kws={"label":label})
    ax.set_title(f"{label} — City × Month", fontweight="bold"); plt.xticks(rotation=60, fontsize=7)
    plt.tight_layout(); plt.show()
'''),

md("## ✅ KPI Summary\n- Delivery rate 59.7% vs 70% target — significant gap\n- Cancel + refund combined = 40.3% loss rate\n- Champions are <15% of customers but drive majority of revenue\n- AOV stable ~₹900; revenue growth varies month-on-month"),
]
save(n3, "03_KPI_Analysis.ipynb")

# ════════════════════════════════════════════════════════════
# NB4 — FUNNEL ANALYSIS
# ════════════════════════════════════════════════════════════
n4 = nb()
n4.cells = [
md("# 🔽 Funnel Analysis\nCovers: order completion funnel · segment funnel · acquisition funnel · cohort retention · RFM · repeat purchase."),
code(SETUP),

md("## 1. Order Completion Funnel"),
code('''\
stages = ["Placed","Not Cancelled","Delivered"]
counts = [len(orders), (orders["order_status"]!="Cancelled").sum(), (orders["order_status"]=="Delivered").sum()]
drops  = [0]+[counts[i-1]-counts[i] for i in range(1,len(counts))]

print(f"{'Stage':<22} {'Count':>8}  {'% of Total':>10}  Drop-off")
print("="*55)
for s, c, d in zip(stages, counts, drops):
    pct = c/counts[0]*100
    print(f"  {s:<20} {c:>8,}  {pct:>9.1f}%  {'-'+str(d)+' ('+f'{d/counts[0]*100:.1f}%'+')' if d else '—'}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(stages[::-1], counts[::-1], color=["#43AA8B","#2A9D8F","#264653"])
for bar, c in zip(ax.patches, counts[::-1]):
    ax.text(bar.get_width()+200, bar.get_y()+bar.get_height()/2, f"{c:,} ({c/counts[0]*100:.1f}%)", va="center", fontsize=10)
ax.set_title("Order Completion Funnel", fontweight="bold"); ax.set_xlim(0, counts[0]*1.3)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
plt.tight_layout(); plt.show()
'''),

md("## 2. Funnel by City, Cuisine & Payment"),
code('''\
for col in ["City","cuisine","payment_mode"]:
    df = full.groupby(col)["order_status"].apply(
        lambda x: pd.Series({"Placed":len(x), "Delivered":(x=="Delivered").sum(),
                              "Cancel%":(x=="Cancelled").sum()/len(x)*100,
                              "Refund%":(x=="Refunded").sum()/len(x)*100})).unstack().round(2)
    df["Conversion%"] = df["Delivered"]/df["Placed"]*100
    display(df.sort_values("Conversion%", ascending=False).round(2))
'''),

md("## 3. Acquisition → Delivery Funnel"),
code('''\
placed_ids   = set(orders["customer_id"].unique())
delivered_ids = set(delivered["customer_id"].unique())
customers["placed"]    = customers["Customer_id"].isin(placed_ids)
customers["delivered"] = customers["Customer_id"].isin(delivered_ids)

acq = customers.groupby("Acquisition_channel").agg(
    signed_up=("Customer_id","count"), placed=("placed","sum"), delivered=("delivered","sum")).reset_index()
acq["conv_%"] = acq["placed"]/acq["signed_up"]*100

fig, ax = plt.subplots(figsize=(10, 4))
x = range(len(acq)); w = 0.28
ax.bar([i-w for i in x], acq["signed_up"],  w, label="Signed Up",   color="#264653")
ax.bar(x,                 acq["placed"],     w, label="Placed Order",color="#2A9D8F")
ax.bar([i+w for i in x],  acq["delivered"],  w, label="Delivered",   color="#43AA8B")
ax.set_xticks(x); ax.set_xticklabels(acq["Acquisition_channel"], rotation=15)
ax.set_title("Acquisition → Delivery Funnel by Channel", fontweight="bold"); ax.legend()
for i, r in acq.iterrows():
    ax.text(i+w, r["delivered"]+5, f"{r['conv_%']:.0f}%", ha="center", fontsize=9)
plt.tight_layout(); plt.show()
'''),

md("## 4. Cohort Retention"),
code('''\
cohort = delivered[["customer_id","order_month"]].copy()
first  = cohort.groupby("customer_id")["order_month"].min().rename("cohort_month")
cohort = cohort.join(first, on="customer_id")
cohort["period"] = (cohort["order_month"] - cohort["cohort_month"]).apply(lambda x: x.n)
matrix = cohort.groupby(["cohort_month","period"])["customer_id"].nunique().unstack()
retention = (matrix.divide(matrix[0], axis=0)*100).round(1).iloc[:15, :12]

fig, ax = plt.subplots(figsize=(15, 7))
sns.heatmap(retention, annot=True, fmt=".0f", cmap="YlGn", linewidths=0.3, ax=ax,
            vmin=0, vmax=100, cbar_kws={"label":"Retention %"})
ax.set_title("Cohort Retention (%) — Month 0 to 11", fontweight="bold")
ax.set_xlabel("Months Since First Order"); ax.set_ylabel("Cohort Month")
plt.tight_layout(); plt.show()
avg = retention.mean().dropna()
print("Avg retention by period:", {f"M{p}":f"{r:.1f}%" for p,r in avg.items()})
'''),

md("## 5. RFM Segmentation"),
code('''\
snapshot = delivered["order_timestamp"].max() + pd.Timedelta(days=1)
rfm = delivered.groupby("customer_id").agg(
    R=("order_timestamp", lambda x: (snapshot-x.max()).days),
    F=("order_id","count"), M=("net_revenue","sum")).reset_index()
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
seg_rev = rfm.groupby("Segment")["M"].sum().sort_values(ascending=False)
print("Revenue by segment:\\n", seg_rev.apply(lambda x: f"₹{x:,.0f}").to_string())

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(rfm["R"], rfm["F"], s=rfm["M"]/500, alpha=0.5, c=rfm["M"], cmap="YlOrRd",
           edgecolors="grey", linewidths=0.3)
ax.set_xlabel("Recency (days)"); ax.set_ylabel("Frequency")
ax.set_title("RFM Scatter — Recency vs Frequency (size=Monetary)", fontweight="bold")
plt.tight_layout(); plt.show()
'''),

md("## 6. Repeat Purchase Funnel"),
code('''\
oc = delivered.groupby("customer_id")["order_id"].count()
thresholds = [1,2,3,5,10]
labels = [f"≥{t}" for t in thresholds]
counts = [(oc>=t).sum() for t in thresholds]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(labels, counts, color=sns.color_palette("Blues_d", len(labels)))
for i, (c, base) in enumerate(zip(counts, [counts[0]]+counts[:-1])):
    pct = c/base*100 if i>0 else 100
    ax.text(i, c+10, f"{c:,}\n({pct:.0f}%)", ha="center", fontsize=9, fontweight="bold")
ax.set_title("Repeat Purchase Funnel", fontweight="bold"); ax.set_ylabel("Customers")
plt.tight_layout(); plt.show()
'''),

md("## ✅ Funnel Findings\n- 40.3% of orders never deliver — critical operational gap\n- Cohort retention drops sharply after Month 1 (~30–40%)\n- Champions + Loyal = small but revenue-dominant segments\n- Biggest drop in repeat funnel is 1st → 2nd order"),
]
save(n4, "04_Funnel_Analysis.ipynb")

# ════════════════════════════════════════════════════════════
# NB5 — INSIGHTS & RECOMMENDATIONS
# ════════════════════════════════════════════════════════════
n5 = nb()
n5.cells = [
md("# 💡 Insights & Recommendations\nCovers: executive summary · revenue · operations · customers · restaurants · discounts · priority matrix · KPI targets."),
code(SETUP),

md("## 1. Executive Summary"),
code('''\
total_orders  = len(orders); gmv = orders["order_amount"].sum()
net_rev = delivered["net_revenue"].sum(); aov = delivered["order_amount"].mean()
d_rate = (orders["order_status"]=="Delivered").mean()*100
c_rate = (orders["order_status"]=="Cancelled").mean()*100
r_rate = (orders["order_status"]=="Refunded").mean()*100
cancel_rev = cancelled["order_amount"].sum(); refund_rev = refunded["order_amount"].sum()

print("="*55)
print(f"  Total Orders    : {total_orders:,}")
print(f"  GMV             : ₹{gmv/1e6:.2f}M")
print(f"  Net Revenue     : ₹{net_rev/1e6:.2f}M")
print(f"  Avg Order Value : ₹{aov:,.0f}")
print(f"  Delivery Rate   : {d_rate:.1f}%  (Target 70%)")
print(f"  Cancel Rate     : {c_rate:.1f}%  (Target <15%)")
print(f"  Refund Rate     : {r_rate:.1f}%  (Target <10%)")
print(f"  Lost Revenue    : ₹{(cancel_rev+refund_rev)/1e6:.2f}M (cancel+refund)")
print("="*55)
'''),

md("## 2. Revenue Insights"),
code('''\
city_rev = delivered.groupby("City")["net_revenue"].sum().sort_values(ascending=False)
city_aov = delivered.groupby("City")["order_amount"].mean()
cuisine_rev = delivered.groupby("cuisine")["net_revenue"].sum().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
city_rev.sort_values().plot.barh(ax=axes[0], color=sns.color_palette("teal", len(city_rev)))
axes[0].set_title("Revenue by City (₹)"); axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1e6:.1f}M"))

city_aov.sort_values().plot.barh(ax=axes[1], color=sns.color_palette("Blues_d", len(city_aov)))
axes[1].axvline(city_aov.mean(), color="red", linestyle="--", label=f"Avg ₹{city_aov.mean():,.0f}")
axes[1].set_title("AOV by City (₹)"); axes[1].legend()
plt.suptitle("Revenue Insights", fontweight="bold"); plt.tight_layout(); plt.show()

print(f"Top city: {city_rev.index[0]} (₹{city_rev.iloc[0]/1e6:.2f}M) | Top cuisine: {cuisine_rev.index[0]} (₹{cuisine_rev.iloc[0]/1e6:.2f}M)")
print(f"AOV gap: ₹{city_aov.max()-city_aov.min():,.0f} between {city_aov.idxmax()} and {city_aov.idxmin()}")
'''),

md("## 3. Operational Insights"),
code('''\
city_cancel = full.groupby("City")["order_status"].apply(lambda x:(x=="Cancelled").sum()/len(x)*100)
city_refund = full.groupby("City")["order_status"].apply(lambda x:(x=="Refunded").sum()/len(x)*100)
brand_refund= full.groupby("restaurant_name")["order_status"].apply(lambda x:(x=="Refunded").sum()/len(x)*100)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, series, title, color, target in [
    (axes[0], city_cancel.sort_values(ascending=True), "Cancel Rate by City", "Reds", 15),
    (axes[1], city_refund.sort_values(ascending=True), "Refund Rate by City", "Purples", 10),
    (axes[2], brand_refund.sort_values(ascending=False).head(8).sort_values(), "Refund Rate — Top 8 Brands", "Oranges", 15)]:
    ax.barh(series.index, series.values, color=sns.color_palette(f"{color}_r", len(series)))
    ax.axvline(target, color="orange", linestyle="--", label=f"Target {target}%")
    ax.set_title(title, fontweight="bold"); ax.legend(fontsize=8)
    for i, v in enumerate(series): ax.text(v+.1, i, f"{v:.1f}%", va="center", fontsize=8)

rescue = (cancel_rev+refund_rev)*0.30/1e6
print(f"Revenue recovery (30% fix): ₹{rescue:.2f}M")
plt.suptitle("Operational Problem Areas", fontweight="bold"); plt.tight_layout(); plt.show()
'''),

md("## 4. Customer Insights"),
code('''\
snapshot = delivered["order_timestamp"].max() + pd.Timedelta(days=1)
rfm = delivered.groupby("customer_id").agg(
    R=("order_timestamp",lambda x:(snapshot-x.max()).days),
    F=("order_id","count"), M=("net_revenue","sum")).reset_index()
rfm["R"] = pd.qcut(rfm["R"],5,labels=[5,4,3,2,1]).astype(int)
rfm["F"] = pd.qcut(rfm["F"].rank(method="first"),5,labels=[1,2,3,4,5]).astype(int)
rfm["M"] = pd.qcut(rfm["M"],5,labels=[1,2,3,4,5]).astype(int)
def seg(r):
    if r.R>=4 and r.F>=4 and r.M>=4: return "Champions"
    elif r.R>=3 and r.F>=3: return "Loyal"
    elif r.R>=4 and r.F<=2: return "New"
    elif r.R<=2 and r.F>=3: return "At Risk"
    elif r.R<=2 and r.F<=2: return "Lost"
    else: return "Potential"
rfm["Segment"] = rfm.apply(seg, axis=1)
segs = rfm.groupby("Segment").agg(n=("customer_id","count"), rev=("M","sum")).sort_values("rev", ascending=False)
display(segs.assign(rev=segs["rev"].map("₹{:,.0f}".format)))

acq = customers["Acquisition_channel"].value_counts()
print(f"\nTop channel: {acq.index[0]} ({acq.iloc[0]:,}) | Weakest: {acq.index[-1]} ({acq.iloc[-1]:,})")
print(f"Champions: {len(rfm[rfm['Segment']=='Champions']):,} | At Risk: {len(rfm[rfm['Segment']=='At Risk']):,} | Lost: {len(rfm[rfm['Segment']=='Lost']):,}")
'''),

md("## 5. Priority Matrix — Impact × Effort"),
code('''\
from matplotlib.patches import Patch
actions = [
    ("Protect Champions",        9, 2, "Customer"),
    ("Win-back At Risk & Lost",  8, 3, "Customer"),
    ("Reduce Cancellations",     9, 5, "Operations"),
    ("Fix Refund Rate SLA",      8, 4, "Operations"),
    ("AOV Upsell in Low Cities", 6, 2, "Revenue"),
    ("Personalise Discounts",    6, 3, "Pricing"),
    ("Top Cuisine Promo",        5, 2, "Revenue"),
    ("Loyalty Programme",        8, 8, "Customer"),
    ("Expand Top City Model",    7, 7, "Revenue"),
    ("Low-Rating Brand Audit",   7, 5, "Operations"),
    ("Payment Retry Flow",       5, 3, "Operations"),
    ("Boost Top Acq Channel",    6, 3, "Customer"),
]
cat_col = {"Operations":"#E63946","Customer":"#2A9D8F","Revenue":"#E9C46A","Pricing":"#457B9D"}
fig, ax = plt.subplots(figsize=(10, 8))
for label, imp, eff, cat in actions:
    ax.scatter(eff, imp, s=280, c=cat_col[cat], alpha=0.85, edgecolors="white", linewidths=1.5, zorder=3)
    ax.annotate(label, (eff, imp), xytext=(5, 4), textcoords="offset points", fontsize=8.5)
ax.axhline(5.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
ax.axvline(5.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
ax.text(0.5, 9.5, "QUICK WINS", fontsize=10, color="#43AA8B", fontweight="bold", alpha=0.7)
ax.text(7,   9.5, "BIG BETS",   fontsize=10, color="#E9C46A", fontweight="bold", alpha=0.7)
ax.text(0.5, 0.8, "FILL-INS",   fontsize=10, color="#888",    fontweight="bold", alpha=0.7)
ax.text(7,   0.8, "RECONSIDER", fontsize=10, color="#E63946", fontweight="bold", alpha=0.7)
ax.legend(handles=[Patch(facecolor=v, label=k) for k,v in cat_col.items()], loc="lower right")
ax.set_xlabel("Effort (1=Low, 10=High)"); ax.set_ylabel("Impact (1=Low, 10=High)")
ax.set_title("Action Priority Matrix", fontsize=13, fontweight="bold")
ax.set_xlim(0,11); ax.set_ylim(0,11); ax.grid(alpha=0.2)
plt.tight_layout(); plt.show()
'''),

md("## 6. KPI Targets & Action Plan"),
code('''\
total_orders  = len(orders)
d_rate = (orders["order_status"]=="Delivered").mean()*100
c_rate = (orders["order_status"]=="Cancelled").mean()*100
r_rate = (orders["order_status"]=="Refunded").mean()*100
aov    = delivered["order_amount"].mean()
mr_s   = delivered.groupby("order_month")["net_revenue"].sum()
mom    = (mr_s.iloc[-1]-mr_s.iloc[-2])/mr_s.iloc[-2]*100 if len(mr_s)>=2 else 0

targets = [
    ("Delivery Rate",     f"{d_rate:.1f}%", "70%",  "🔴"),
    ("Cancel Rate",       f"{c_rate:.1f}%", "<15%", "🔴"),
    ("Refund Rate",       f"{r_rate:.1f}%", "<10%", "🔴"),
    ("Avg Order Value",   f"₹{aov:,.0f}",  "₹950", "🟡"),
    ("MoM Revenue Growth",f"{mom:+.1f}%",  "+5%",  "🟡"),
]
print(f"{'KPI':<22} {'CURRENT':>10}  {'TARGET':>8}  STATUS")
print("="*52)
for k,c,t,s in targets: print(f"  {k:<20} {c:>10}  {t:>8}  {s}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (label, curr, tgt, _) in zip(axes, targets[:3]):
    curr_val = float(curr.replace("%",""))
    tgt_val  = float(tgt.replace("<","").replace("%",""))
    ok = (label=="Delivery Rate" and curr_val>=tgt_val) or (label!="Delivery Rate" and curr_val<=tgt_val)
    ax.bar(["Current","Target"], [curr_val, tgt_val], color=["#43AA8B" if ok else "#E63946","#2A9D8F"], width=0.4)
    ax.set_title(label, fontweight="bold"); ax.set_ylabel("%")
    for i, v in enumerate([curr_val, tgt_val]): ax.text(i, v+0.5, f"{v:.1f}%", ha="center", fontweight="bold")
plt.suptitle("KPI: Current vs Target", fontweight="bold"); plt.tight_layout(); plt.show()

print("""
🎯 Action Plan
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THIS WEEK  (Quick Wins)
  1. Activate VIP perks for Champions segment
  2. Set refund-rate SLA for all restaurant partners
  3. Launch At-Risk win-back email + push campaign
  4. Feature top cuisine on homepage banner

THIS MONTH  (Medium effort)
  5. Build ETA notification → reduce cancellations
  6. Add AOV upsell prompts at checkout (low-AOV cities)
  7. Replace blanket discounts with segment-based offers

THIS QUARTER  (Big Bets)
  8. Launch tiered Loyalty Programme
  9. Replicate top-city operational model in bottom cities
  10. Monthly quality audits for brands below 4.0★
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")
'''),
]
save(n5, "05_Insights_Recommendations.ipynb")

print("\n✅  All 5 notebooks rebuilt (lean version)")
