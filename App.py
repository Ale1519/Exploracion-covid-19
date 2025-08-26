# App.py
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# statsmodels para tests y modelos
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------------------------------------------
# Configuración de página (solo 1 vez)
# ---------------------------------------------------
st.set_page_config(page_title="COVID-19 Viz – Pregunta 2", layout="wide")

# ---------------------------------------------------
# Carga del reporte diario JHU
# ---------------------------------------------------
GITHUB_BASE = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports"

@st.cache_data(show_spinner=False)
def load_daily_report(yyyy_mm_dd: str):
    yyyy, mm, dd = yyyy_mm_dd.split("-")
    url = f"{GITHUB_BASE}/{mm}-{dd}-{yyyy}.csv"
    df = pd.read_csv(url)
    # normalizar nombres por si varían
    lower = {c.lower(): c for c in df.columns}
    cols = {
        "country": lower.get("country_region", "Country_Region"),
        "province": lower.get("province_state", "Province_State"),
        "confirmed": lower.get("confirmed", "Confirmed"),
        "deaths": lower.get("deaths", "Deaths"),
        "recovered": lower.get("recovered", "Recovered") if "recovered" in lower else None,
        "active": lower.get("active", "Active") if "active" in lower else None,
    }
    return df, url, cols

st.sidebar.title("Opciones")
fecha = st.sidebar.date_input("Fecha del reporte (JHU CSSE)", value=pd.to_datetime("2022-09-09"))
fecha_str = pd.to_datetime(fecha).strftime("%Y-%m-%d")
df, source_url, cols = load_daily_report(fecha_str)
st.sidebar.caption(f"Fuente: {source_url}")

# Atajos de columnas según los nombres detectados
country_col = cols["country"]
prov_col = cols["province"]
C, D = cols["confirmed"], cols["deaths"]
R, A = cols["recovered"], cols["active"]

# ===================================================
# Pregunta 2 (exploración y visualización)
# ===================================================
st.title("Exploración COVID-19 – Versión Streamlit (Preg2)")
st.caption("Mostrar/ocultar filas/columnas y gráficos (líneas, barras, sectores, histograma y boxplot).")

# a) Mostrar filas
st.header("a) Mostrar filas")
mostrar_todas = st.checkbox("Mostrar todas las filas", value=False)
if mostrar_todas:
    st.dataframe(df, use_container_width=True)
else:
    st.dataframe(df.head(25), use_container_width=True)

# b) Mostrar columnas
st.header("b) Mostrar columnas")
with st.expander("Vista de columnas"):
    st.write(list(df.columns))
st.caption("Usa el scroll horizontal de la tabla para ver todas las columnas en pantalla.")

# c) Gráfica de líneas por país (muertes > 2500)
st.header("c) Gráfica de líneas por país (muertes > 2500)")
metrics = [m for m in [C, D, R, A] if m and m in df.columns]
base = df[[country_col] + metrics].copy()
base = base.rename(columns={country_col: "Country_Region"})

filtrado = base.loc[base[D] > 2500]
agr = filtrado.groupby("Country_Region").sum(numeric_only=True)
orden = agr.sort_values(D)
if not orden.empty:
    st.line_chart(orden[[c for c in [C, R, A] if c in orden.columns]])

# d) Barras: fallecidos por estado de EE.UU.
st.header("d) Barras: fallecidos por estado de EE.UU.")
dfu = df[df[country_col] == "US"]
if len(dfu) == 0:
    st.info("Para esta fecha no hay registros con Country_Region='US'.")
else:
    agg_us = dfu.groupby(prov_col)[D].sum(numeric_only=True).sort_values(ascending=False)
    top_n = st.slider("Top estados por fallecidos", 5, 50, 20)
    st.bar_chart(agg_us.head(top_n))

# e) Gráfica de sectores (simulada con barra)
st.header("e) Gráfica de sectores (simulada)")
lista_paises = ["Colombia", "Chile", "Peru", "Argentina", "Mexico"]
sel = st.multiselect("Países", sorted(df[country_col].unique().tolist()), default=lista_paises)
agg_latam = df[df[country_col].isin(sel)].groupby(country_col)[D].sum(numeric_only=True)
if agg_latam.sum() > 0:
    st.write("Participación de fallecidos")
    st.dataframe(agg_latam)
    normalized = agg_latam / agg_latam.sum()
    st.bar_chart(normalized)
else:
    st.warning("Sin datos para los países seleccionados")

# f) Histograma (simulado con bar chart) de fallecidos por país
st.header("f) Histograma de fallecidos por país")
muertes_pais = df.groupby(country_col)[D].sum(numeric_only=True).sort_values(ascending=False)
st.bar_chart(muertes_pais)

# g) Boxplot (simulado) con resumen estadístico
st.header("g) Boxplot (simulado)")
cols_box = [c for c in [C, D, R, A] if c and c in df.columns]
subset = df[cols_box].fillna(0)
subset_plot = subset.head(25)
st.write("Resumen estadístico (simulación de boxplot):")
st.dataframe(subset_plot.describe().T)

# ===================================================
# Pregunta 3 (estadística descriptiva y avanzada)
# ===================================================
st.header("Exploración COVID-19 – Versión Streamlit (Preg3)")
st.title("2. Estadística descriptiva y avanzada")

# 2.1 Métricas clave por país
st.subheader("2.1. Métricas clave por país")
poblaciones = {
    "Colombia": 51500000,
    "Chile": 19000000,
    "Peru": 33500000,
    "Argentina": 46000000,
    "Mexico": 126000000,
    "US": 331000000,
    "Brazil": 214000000
}
agr2 = df.groupby(country_col)[[C, D]].sum(numeric_only=True)
agr2["CFR"] = (agr2[D] / agr2[C]).replace([np.inf, np.nan], 0)
agr2["Rate_per_100k"] = [
    (agr2.loc[c, C] / poblaciones.get(c, 1e6)) * 100000 for c in agr2.index
]
st.dataframe(agr2.sort_values("CFR", ascending=False).head(20))

# 2.2 Intervalos de confianza para CFR (95%)
st.subheader("2.2. Intervalos de confianza para CFR (95%)")
sel_country = st.selectbox("Selecciona país para IC del CFR", agr2.index)
n_tot = agr2.loc[sel_country, C]
d_tot = agr2.loc[sel_country, D]
if n_tot > 0:
    p_hat = d_tot / n_tot
    se = np.sqrt(p_hat * (1 - p_hat) / n_tot)
    ci_low, ci_high = p_hat - 1.96 * se, p_hat + 1.96 * se
    st.write(f"CFR {sel_country}: {p_hat:.3%}")
    st.write(f"IC95%: ({ci_low:.3%}, {ci_high:.3%})")
else:
    st.warning("No hay suficientes datos para calcular IC.")

# 2.3 Test de hipótesis de proporciones (CFR)
st.subheader("2.3. Comparación de CFR entre dos países")
p1, p2 = st.select_slider("Selecciona dos países", options=agr2.index.tolist(), value=("Colombia","Mexico"))
n1, d1 = agr2.loc[p1, C], agr2.loc[p1, D]
n2, d2 = agr2.loc[p2, C], agr2.loc[p2, D]
if n1 > 0 and n2 > 0:
    stat_z, pval = proportions_ztest(count=[d1, d2], nobs=[n1, n2])
    st.write(f"CFR {p1}: {d1/n1:.3%}, CFR {p2}: {d2/n2:.3%}")
    st.write(f"Z = {stat_z:.2f}, p-valor = {pval:.4f}")
    if pval < 0.05:
        st.success("Se rechaza H0: Hay diferencia significativa en CFR.")
    else:
        st.info("No se rechaza H0: No hay diferencia significativa en CFR.")
else:
    st.warning("Datos insuficientes para el test.")

# 2.4 Outliers (Z-score) en fallecidos por país
st.header("2.4 Outliers en fallecidos (Z-score)")
muertes_pais2 = df.groupby(country_col)[D].sum(numeric_only=True)
if muertes_pais2.std() > 0:
    z_scores = (muertes_pais2 - muertes_pais2.mean()) / muertes_pais2.std()
    outliers = muertes_pais2[z_scores.abs() > 3]
else:
    outliers = pd.Series(dtype=float)
st.write("Outliers detectados (|Z| > 3):")
st.dataframe(outliers)

# 2.5 Gráfico de control (3σ) de muertes diarias
st.subheader("2.5. Gráfico de control de muertes diarias")
country_daily = st.selectbox("País para gráfico de control", agr2.index, index=agr2.index.tolist().index("Colombia") if "Colombia" in agr2.index else 0)

# Serie de muertes diarias a partir de JHU (time series global)
url_time = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
df_deaths = pd.read_csv(url_time)
ts_country = df_deaths[df_deaths["Country/Region"] == country_daily].drop(columns=["Province/State","Lat","Long","Country/Region"]).sum()
daily_series = ts_country.diff().fillna(0).clip(lower=0)
mean_v, std_v = daily_series.mean(), daily_series.std()
ucl, lcl = mean_v + 3*std_v, max(mean_v - 3*std_v, 0)

fig_ctrl, ax_ctrl = plt.subplots(figsize=(10,5))
ax_ctrl.plot(daily_series.index, daily_series.values, marker="o", label="Muertes diarias")
ax_ctrl.axhline(mean_v, color="green", linestyle="--", label="Media")
ax_ctrl.axhline(ucl, color="red", linestyle="--", label="+3σ")
ax_ctrl.axhline(lcl, color="red", linestyle="--", label="-3σ")
ax_ctrl.set_title(f"Gráfico de control (3σ) – {country_daily}")
ax_ctrl.legend()
st.pyplot(fig_ctrl)

# ===================================================
# 3. Modelado y proyecciones
# ===================================================
st.header("3. Modelado y proyecciones")

@st.cache_data(show_spinner=False)
def load_time_series(kind: str):
    """
    kind: 'confirmed' o 'deaths'
    Devuelve DataFrame (index=fecha, columnas=país) con valores ACUMULADOS.
    """
    base = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series"
    if kind == "confirmed":
        url = f"{base}/time_series_covid19_confirmed_global.csv"
    else:
        url = f"{base}/time_series_covid19_deaths_global.csv"
    tdf = pd.read_csv(url)
    id_cols = ["Country/Region", "Province/State", "Lat", "Long"]
    date_cols = [c for c in tdf.columns if c not in id_cols]
    by_country = tdf.groupby("Country/Region")[date_cols].sum()
    by_country = by_country.T
    by_country.index = pd.to_datetime(by_country.index, errors="coerce")
    by_country = by_country.sort_index()
    return by_country

@st.cache_data(show_spinner=False)
def mk_daily(series_accum: pd.Series) -> pd.Series:
    """De acumulados a diarios (no negativos)."""
    s = series_accum.diff().fillna(0)
    return s.clip(lower=0)

# Selección de variable/país/horizonte/modelo
st.subheader("Selección de país y variable")
ts_kind = st.radio("Variable a modelar", ["Muertes diarias", "Casos diarios"], index=0, horizontal=True)
horizon = st.slider("Horizonte de pronóstico (días)", 7, 28, 14, step=7)
model_type = st.selectbox("Modelo", ["SARIMA (semanal)", "ETS (Holt-Winters)"])

acc_deaths = load_time_series("deaths")
acc_conf   = load_time_series("confirmed")
available_countries = sorted(list(set(acc_deaths.columns) & set(acc_conf.columns)))
country_sel = st.selectbox("País", available_countries, index=available_countries.index("Peru") if "Peru" in available_countries else 0)

accum_series = acc_deaths[country_sel] if ts_kind == "Muertes diarias" else acc_conf[country_sel]
daily = mk_daily(accum_series).asfreq("D").fillna(0)
smooth7 = daily.rolling(7, min_periods=1).mean()

st.write("Vista rápida (últimos 30 días):")
st.dataframe(pd.DataFrame({"valor_diario": daily.tail(30).round(2), "suavizado_7d": smooth7.tail(30).round(2)}))

# 3.3 Backtesting (ventana expansiva)
def backtest_expanding(y: pd.Series, model_type: str, seasonal_periods=7, order=(1,1,1), sorder=(1,0,1,7),
                       horizon=14, origins=4, min_train=120):
    y = y.astype(float)
    n = len(y)
    rows, cut_points = [], []

    last_block = min(n - min_train, origins * horizon)
    if last_block <= 0:
        return None, {"MAE": np.nan, "MAPE": np.nan}
    start = n - last_block - horizon
    for k in range(origins):
        cut = start + k * horizon
        if cut >= min_train:
            cut_points.append(cut)

    for cut in cut_points:
        train = y.iloc[:cut]
        test  = y.iloc[cut:cut+horizon]

        try:
            if model_type.startswith("SARIMA"):
                mod = SARIMAX(train, order=order, seasonal_order=sorder, enforce_stationarity=False, enforce_invertibility=False)
                res = mod.fit(disp=False)
                pred = res.get_forecast(steps=len(test)).predicted_mean
            else:
                mod = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=seasonal_periods, initialization_method="estimated")
                res = mod.fit(optimized=True)
                pred = res.forecast(len(test))
        except Exception:
            pred = pd.Series([np.nan]*len(test), index=test.index)

        fold = pd.DataFrame({"y_true": test, "y_pred": pred})
        fold["origin_end"] = train.index[-1]
        rows.append(fold)

    if not rows:
        return None, {"MAE": np.nan, "MAPE": np.nan}
    bt = pd.concat(rows)
    mae = (bt["y_true"] - bt["y_pred"]).abs().mean()
    mape = ((bt["y_true"] - bt["y_pred"]).abs() / bt["y_true"].replace(0, np.nan)).mean() * 100
    return bt, {"MAE": float(mae), "MAPE": float(mape)}

with st.spinner("Ejecutando backtesting..."):
    bt, metrics = backtest_expanding(
        daily,
        model_type=model_type,
        seasonal_periods=7,
        order=(1,1,1),
        sorder=(1,0,1,7),
        horizon=horizon,
        origins=4,
        min_train=120
    )

st.subheader("3.3 Backtesting (expanding window)")
if bt is None:
    st.info("Serie demasiado corta para backtesting. Prueba con otro país o reduce el horizonte.")
else:
    st.write(f"**MAE:** {metrics['MAE']:.2f} | **MAPE:** {metrics['MAPE']:.2f}%")
    last_origin = bt["origin_end"].max()
    last_fold = bt[bt["origin_end"] == last_origin]
    fig_bt, ax_bt = plt.subplots(figsize=(10,4))
    ax_bt.plot(last_fold.index, last_fold["y_true"], marker="o", label="Real")
    ax_bt.plot(last_fold.index, last_fold["y_pred"], marker="o", label="Pronóstico (fold más reciente)")
    ax_bt.set_title(f"Backtesting – {country_sel} – {ts_kind}")
    ax_bt.legend()
    st.pyplot(fig_bt)

# 3.2 Pronóstico final + 3.4 Bandas de confianza
st.subheader("3.2 Pronóstico a futuro y 3.4 Bandas de confianza")
mean_fc, lower, upper = None, None, None
try:
    if model_type.startswith("SARIMA"):
        mod = SARIMAX(daily, order=(1,1,1), seasonal_order=(1,0,1,7),
                      enforce_stationarity=False, enforce_invertibility=False)
        res = mod.fit(disp=False)
        fc = res.get_forecast(steps=horizon)
        mean_fc = fc.predicted_mean
        conf = fc.conf_int(alpha=0.05)
        lower = conf.iloc[:, 0]
        upper = conf.iloc[:, 1]
    else:
        mod = ExponentialSmoothing(daily, trend="add", seasonal="add", seasonal_periods=7, initialization_method="estimated")
        res = mod.fit(optimized=True)
        mean_fc = res.forecast(horizon)
        resid = daily - res.fittedvalues.reindex(daily.index).fillna(method="bfill").fillna(method="ffill")
        sigma = resid.std()
        lower = mean_fc - 1.96 * sigma
        upper = mean_fc + 1.96 * sigma
except Exception as e:
    st.error(f"No se pudo ajustar el modelo: {e}")

if mean_fc is not None:
    hist_win = 120
    plot_hist = daily.tail(hist_win)
    plot_sma = smooth7.tail(hist_win)

    fig_f, ax_f = plt.subplots(figsize=(12,5))
    ax_f.plot(plot_hist.index, plot_hist.values, label="Diario", alpha=0.5)
    ax_f.plot(plot_sma.index,  plot_sma.values,  label="Suavizado 7d", linewidth=2)
    ax_f.plot(mean_fc.index, mean_fc.values, label=f"Pronóstico {horizon}d ({model_type})", marker="o")
    ax_f.fill_between(mean_fc.index, lower.values, upper.values, alpha=0.2, label="IC 95%")
    ax_f.set_title(f"{ts_kind} – {country_sel}")
    ax_f.set_ylabel("conteo")
    ax_f.legend()
    st.pyplot(fig_f)

    st.caption("Nota: Para ETS las bandas de confianza se aproximan usando la desviación estándar de los residuos.")
