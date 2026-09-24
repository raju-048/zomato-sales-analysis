# Zomato Order Analytics

End-to-end analysis of 50,000 food delivery orders across 8 Indian cities.

## Data
- `Orders.csv` — 50,000 orders
- `Customer.csv` — 4,999 customers
- `Restaurants.csv` — 200 restaurants

## Notebooks
| File | Description |
|---|---|
| `01_EDA.ipynb` | Distributions, patterns, city overview |
| `02_Driver_Analysis.ipynb` | Revenue, cancel & refund drivers |
| `03_KPI_Analysis.ipynb` | KPI scorecard vs targets |
| `04_Funnel_Analysis.ipynb` | Cohort retention & RFM |
| `05_Insights_Recommendations.ipynb` | Action plan |

## Dashboard
```bash
python generate_dashboard.py
```
Then open `zomato_dashboard.html` in a browser.

## Setup
```bash
pip install pandas numpy plotly matplotlib seaborn scipy scikit-learn nbformat jupyter
```

## Key Metrics
- Delivery Rate: 59.7% (target 70%)
- Cancel Rate: 20.2% (target 15%)
- Refund Rate: 20.1% (target 10%)
- Net Revenue: ₹25.77M
