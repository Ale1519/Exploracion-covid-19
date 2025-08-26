import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.proportion import proportions_ztest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px


st.set_page_config(page_title="COVID-19 – Dashboard integral", layout="wide")


# ------------------------------
# Data loaders (cached)
# ------------------------------
GITHUB_BASE = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data"
DAILY_BASE = f"{GITHUB_BASE}/csse_covid_19_daily_reports"
TS_BASE = f"{GITHUB_BASE}/csse_covid_19_time_series"


@st.cache_data(show_spinner=False)
def load_daily_report(yyyy_mm_dd: str):
yyyy, mm, dd = yyyy_mm_dd.split("-")
url = f"{DAILY_BASE}/{mm}-{dd}-{yyyy}.csv"
df = pd.read_csv(url)
# normalizar nombres que cambian entre fechas
lower = {c.lower(): c for c in df.columns}
cols = {
"country": lower.get("country_region", "Country_Region"),
"province": lower.get("province_state", "Province_State"),
"confirmed": lower.get("confirmed", "Confirmed"),
"deaths": lower.get("deaths", "Deaths"),
"recovered": lower.get("recovered", "Recovered") if "recovered" in lower else None,
"active": lower.get("active", "Active") if "active" in lower else None,
"lat": lower.get("lat", "Lat"),
"lon": lower.get("long_", "Long_"),
}
# tipado básico
for k in ["confirmed","deaths","recovered","active"]:
c = cols.get(k)
if c and c in df.columns:
df[c] = pd.to_numeric(df[c], errors="coerce")
return df, url, cols


@st.cache_data(show_spinner=False)
def load_time_series(kind: str = "deaths"): # kind: deaths | confirmed | recovered
url = f"{TS_BASE}/time_series_covid19_{kind}_global.csv"
ts = pd.read_csv(url)
return ts, url


# ------------------------------
# Utils
# ------------------------------


def fig_to_png_bytes(fig) -> bytes:
buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight")
buf.seek(0)
return buf.getvalue()




def series_from_ts(ts_df: pd.DataFrame, country: str) -> pd.Series:
st.success("Listo. Usa la barra lateral para cargar poblaciones opcionalmente y explora las pestañas.")
