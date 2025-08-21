import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="COVID-19 Viz – Pregunta 2", layout="wide")

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

st.title("Exploración COVID-19 – Versión Streamlit (Preg2)")
st.caption("Adaptación fiel del script original: mostrar/ocultar filas/columnas y varios gráficos (líneas, barras, sectores, histograma y boxplot).")

# ———————————————————————————————————————————————
# a) Mostrar todas las filas del dataset, luego volver al estado inicial
# ———————————————————————————————————————————————
st.header("a) Mostrar filas")
mostrar_todas = st.checkbox("Mostrar todas las filas", value=False)
if mostrar_todas:
    st.dataframe(df, use_container_width=True)
else:
    st.dataframe(df.head(25), use_container_width=True)

# ———————————————————————————————————————————————
# b) Mostrar todas las columnas del dataset
# ———————————————————————————————————————————————
st.header("b) Mostrar columnas")
with st.expander("Vista de columnas"):
    st.write(list(df.columns))

st.caption("Usa el scroll horizontal de la tabla para ver todas las columnas en pantalla.")

# ———————————————————————————————————————————————
# c) Línea del total de fallecidos (>2500) vs Confirmed/Recovered/Active por país
# ———————————————————————————————————————————————
st.header("c) Gráfica de líneas por país (muertes > 2500)")
C, D = cols["confirmed"], cols["deaths"]
R, A = cols["recovered"], cols["active"]

metrics = [m for m in [C, D, R, A] if m and m in df.columns]
base = df[[cols["country"]] + metrics].copy()
base = base.rename(columns={cols["country"]: "Country_Region"})

filtrado = base.loc[base[D] > 2500]
agr = filtrado.groupby("Country_Region").sum(numeric_only=True)
orden = agr.sort_values(D)

if not orden.empty:
    st.line_chart(orden[[c for c in [C, R, A] if c in orden.columns]])

# ———————————————————————————————————————————————
# d) Barras de fallecidos de estados de Estados Unidos
# ———————————————————————————————————————————————
st.header("d) Barras: fallecidos por estado de EE.UU.")
country_col = cols["country"]
prov_col = cols["province"]

dfu = df[df[country_col] == "US"]
if len(dfu) == 0:
    st.info("Para esta fecha no hay registros con Country_Region='US'.")
else:
    agg_us = dfu.groupby(prov_col)[D].sum(numeric_only=True).sort_values(ascending=False)
    top_n = st.slider("Top estados por fallecidos", 5, 50, 20)
    st.bar_chart(agg_us.head(top_n))

# ———————————————————————————————————————————————
# e) Gráfica de sectores (simulada con barra si no hay pie nativo)
# ———————————————————————————————————————————————
st.header("e) Gráfica de sectores (simulada)")
lista_paises = ["Colombia", "Chile", "Peru", "Argentina", "Mexico"]
sel = st.multiselect("Países", sorted(df[country_col].unique().tolist()), default=lista_paises)
agg_latam = df[df[country_col].isin(sel)].groupby(country_col)[D].sum(numeric_only=True)
if agg_latam.sum() > 0:
    st.write("Participación de fallecidos")
    st.dataframe(agg_latam)
    # Como Streamlit no tiene pie nativo, mostramos distribución normalizada como barra
    normalized = agg_latam / agg_latam.sum()
    st.bar_chart(normalized)
else:
    st.warning("Sin datos para los países seleccionados")

# ———————————————————————————————————————————————
# f) Histograma del total de fallecidos por país (simulado con bar_chart)
# ———————————————————————————————————————————————
st.header("f) Histograma de fallecidos por país")
muertes_pais = df.groupby(country_col)[D].sum(numeric_only=True)
st.bar_chart(muertes_pais)

# ———————————————————————————————————————————————
# g) Boxplot de Confirmed, Deaths, Recovered, Active (simulado con box_chart)
# ———————————————————————————————————————————————
st.header("g) Boxplot (simulado)")
cols_box = [c for c in [C, D, R, A] if c and c in df.columns]
subset = df[cols_box].fillna(0)
subset_plot = subset.head(25)
# Streamlit no tiene boxplot nativo, así que mostramos estadísticas resumen en tabla
st.write("Resumen estadístico (simulación de boxplot):")
st.dataframe(subset_plot.describe().T)

st.header("Exploración COVID-19 – Versión Streamlit (Preg3)")

st.set_page_config(page_title="Análisis COVID-19", layout="wide")
st.title("Análisis COVID-19 - 18 de Abril 2022")

st.title("2. Estadística descriptiva y avanzada")

C, D = cols["confirmed"], cols["deaths"]

# 2.1. Calcular métricas clave por país
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
agr = df.groupby(cols["country"])[[C, D]].sum(numeric_only=True)
agr["CFR"] = (agr[D] / agr[C]).replace([np.inf, np.nan], 0)
agr["Rate_per_100k"] = [
    (agr.loc[c, C] / poblaciones.get(c, 1e6)) * 100000 for c in agr.index
]
st.dataframe(agr.sort_values("CFR", ascending=False).head(20))

# 2.2. Intervalos de confianza para CFR
st.subheader("2.2. Intervalos de confianza para CFR (95%)")
sel_country = st.selectbox("Selecciona país para IC del CFR", agr.index)
n = agr.loc[sel_country, C]
d = agr.loc[sel_country, D]
if n > 0:
    p = d / n
    se = np.sqrt(p * (1 - p) / n)
    ci_low, ci_high = p - 1.96 * se, p + 1.96 * se
    st.write(f"CFR {sel_country}: {p:.3%}")
    st.write(f"IC95%: ({ci_low:.3%}, {ci_high:.3%})")
else:
    st.warning("No hay suficientes datos para calcular IC.")

# 2.3. Test de hipótesis de proporciones
st.subheader("2.3. Comparación de CFR entre dos países")
c1, c2 = st.select_slider("Selecciona dos países", options=agr.index, value=("Colombia","Mexico"))
n1, d1 = agr.loc[c1, C], agr.loc[c1, D]
n2, d2 = agr.loc[c2, C], agr.loc[c2, D]
if n1 > 0 and n2 > 0:
    p1, p2 = d1/n1, d2/n2
    p_pool = (d1+d2)/(n1+n2)
    se = np.sqrt(p_pool*(1-p_pool)*(1/n1+1/n2))
    z = (p1-p2)/se if se>0 else 0
    pval = 2*(1-stats.norm.cdf(abs(z)))
    st.write(f"CFR {c1}: {p1:.3%}, CFR {c2}: {p2:.3%}")
    st.write(f"Z = {z:.2f}, p-valor = {pval:.4f}")
    if pval<0.05:
        st.success("Diferencia estadísticamente significativa (p<0.05).")
    else:
        st.info("No se encontró diferencia significativa.")
else:
    st.warning("Datos insuficientes para test.")

# 2.4. Outliers (Z-score)
st.subheader("2.4. Detección de outliers (Z-score en CFR)")
agr["zscore_cfr"] = stats.zscore(agr["CFR"].fillna(0))
outliers = agr[agr["zscore_cfr"].abs()>3]
st.dataframe(outliers)

# 2.5. Gráfico de control (3σ) de muertes diarias
st.subheader("2.5. Gráfico de control de muertes diarias")
country_daily = st.selectbox("Selecciona país para gráfico de control", agr.index, index=agr.index.tolist().index("Colombia") if "Colombia" in agr.index else 0)

# Datos diarios de muertes para país seleccionado
url_time = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
df_deaths = pd.read_csv(url_time)
df_country = df_deaths[df_deaths["Country/Region"]==country_daily].drop(columns=["Province/State","Lat","Long","Country/Region"]).sum()
daily = df_country.diff().fillna(0)

mean, std = daily.mean(), daily.std()
ucl, lcl = mean+3*std, max(mean-3*std,0)

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(daily.index, daily.values, marker="o", label="Muertes diarias")
ax.axhline(mean, color="green", linestyle="--", label="Media")
ax.axhline(ucl, color="red", linestyle="--", label="+3σ")
ax.axhline(lcl, color="red", linestyle="--", label="-3σ")
ax.set_title(f"Gráfico de control (3σ) – {country_daily}")
ax.legend()
st.pyplot(fig)

