# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from funciones import (
    estadisticas_basicas,
    pearson_r,
    linear_regression,
    classification_by_r,
    test_mean_z,
    test_mean_t,
    p_value_from_stat
)
from scipy import stats

st.set_page_config(layout="wide", page_title="Calculadora: Tests, Correlación y Regresión")

# --- Helper UI functions ---------------------------------------------------
def complementary(op: str) -> str:
    mapping = {
        "=": "!=",
        "!=": "=",
        ">=": "<",
        "<=": ">",
        ">": "<=",
        "<": ">="
    }
    return mapping.get(op, "!=")

def operator_to_test_type(op_h1: str) -> str:
    """Convierte operador de H1 a tipo de test para cálculo de p-value."""
    if op_h1 in ["!="]:
        return "two-sided"
    if op_h1 in [">", ">="]:
        return "greater"
    if op_h1 in ["<", "<="]:
        return "less"
    return "two-sided"

# Sidebar: menu izquierda con pestañas
st.sidebar.title("Menú")
menu = st.sidebar.selectbox("Selecciona sección", ["Datos", "Pruebas de hipótesis", "Regresión lineal"])

# --- Datos: carga CSV ------------------------------------------------------
if menu == "Datos":
    st.header("Carga de datos")
    uploaded = st.file_uploader("Sube un archivo CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.subheader("Vista previa (head)")
        st.dataframe(df.head())
        st.markdown(f"**Filas:** {df.shape[0]} — **Columnas:** {df.shape[1]}")
        st.markdown("**Columnas detectadas:**")
        st.write(df.dtypes)
        st.markdown("---")
        st.info("Nota: recuerda que las pestañas 'Pruebas de hipótesis' y 'Regresión lineal' usan el mismo CSV cargado aquí. Si vas a esas pestañas sin cargar, te pedirá el CSV de nuevo.")
    else:
        st.info("Sube un CSV para comenzar. Debe contener columnas numéricas para los análisis.")

# --- Pestaña: Pruebas de hipótesis ------------------------------------------
if menu == "Pruebas de hipótesis":
    st.header("Pruebas de hipótesis (media)")
    uploaded = st.file_uploader("Sube un archivo CSV (misma que en Datos)", type=["csv"], key="h_csv")
    if uploaded is None:
        st.warning("Sube un CSV para usar esta pestaña.")
        st.stop()
    df = pd.read_csv(uploaded)
    st.subheader("Head de los datos")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.error("No hay columnas numéricas en el CSV.")
        st.stop()

    col = st.selectbox("Selecciona la variable numérica para testear (columna Y)", numeric_cols)

    # Slider para tamaño de muestra
    total = df.shape[0]
    min_slider = 10 if total >= 10 else 1
    max_slider = total
    sample_size = st.slider("Tamaño de la muestra", min_value=min_slider, max_value=max_slider, value=min_slider, step=1)

    # Si sample_size < total => sample aleatorio; si igual => tomar todos en orden ascendente
    if sample_size < total:
        sample_df = df.sample(n=sample_size, random_state=42).sort_index()
    else:
        # orden ascendente por primera columna numérica, si existe; si no, por index
        numeric_cols_all = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols_all:
            sample_df = df.sort_values(by=numeric_cols_all[0]).head(sample_size)
        else:
            sample_df = df.head(sample_size)

    st.subheader("Muestra seleccionada")
    st.write(f"Registros mostrados: {sample_df.shape[0]}")
    st.dataframe(sample_df)

    # Descarga del dataframe de la muestra
    csv_bytes = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button("Descargar muestra (CSV)", data=csv_bytes, file_name="muestra.csv", mime="text/csv")

    # Estadísticas
    stats_res = estadisticas_basicas(sample_df[col])
    st.write("Estadísticas básicas:")
    st.metric("n", stats_res["n"])
    st.metric("Media (x̄)", f"{stats_res['mean']:.4f}")
    st.metric("Desv. estándar (muestral)", f"{stats_res['std_m']:.4f}")

    # Inputs H0 y H1 con operadores
    st.subheader("Formulación de hipótesis")
    st.write("Define H0 y H1. Si eliges un operador en H0, H1 se autocompleta con el op. complementario (puedes editar).")
    col1, col2 = st.columns(2)
    with col1:
        h0_value = st.number_input("Valor H0 (media µ0)", value=float(stats_res["mean"]), format="%.6f")
        op_h0 = st.selectbox("Operador H0", ["=", "!=", ">=", "<=", ">", "<"], index=0)
    with col2:
        # sugerir operador complementario
        suggested_h1 = complementary(op_h0)
        op_h1 = st.selectbox("Operador H1 (sugerido)", ["=", "!=", ">", "<", ">=", "<="], index=["=", "!=", ">", "<", ">=", "<="].index(suggested_h1))
        st.write("H1 será interpretada automáticamente para calcular p-valor.")

    # Selección: usar Z (varianza poblacional conocida) o T (varianza muestral)
    st.subheader("Tipo de estadístico")
    use_z = st.radio("¿Usar Z o T?", options=["Z (varianza poblacional conocida)", "T (varianza muestral)"])
    if use_z.startswith("Z"):
        # pedir sigma poblacional (opcional)
        sigma_pop = st.number_input("Ingrese desviación estándar poblacional σ (si la conoce)", value=float(stats_res["std_p"]) if stats_res["std_p"]>0 else 1.0, format="%.6f")
    else:
        sigma_pop = None

    alpha = st.number_input("Nivel de significancia α", min_value=0.0001, max_value=0.5, value=0.05, step=0.01, format="%.4f")

    # Cálculo del estadístico
    xbar = stats_res["mean"]
    n = stats_res["n"]
    if use_z.startswith("Z"):
        test = test_mean_z(xbar=xbar, mu0=h0_value, sigma=sigma_pop, n=n)
        stat = test["stat"]
        se = test["se"]
        use_t_flag = False
        dfree = None
    else:
        test = test_mean_t(xbar=xbar, mu0=h0_value, s=stats_res["std_m"], n=n)
        stat = test["stat"]
        se = test["se"]
        use_t_flag = True
        dfree = test["df"]

    # p-valor según op H1
    test_type = operator_to_test_type(op_h1)
    pval = p_value_from_stat(stat, df=dfree, test_type=test_type, use_t=use_t_flag)

    st.subheader("Resultados del test")
    st.write(f"Estadístico calculado: {'t' if use_t_flag else 'z'} = {stat:.4f}")
    st.write(f"Error estándar (SE) = {se:.4f}")
    if dfree:
        st.write(f"Grados de libertad = {dfree}")
    st.write(f"P-valor (según alternativa \"H1 {op_h1} {h0_value}\") = {pval:.6f}")
    reject = pval < alpha
    conclusion = "Rechazamos H0" if reject else "No rechazamos H0"
    st.markdown(f"### Conclusión: **{conclusion}** (α = {alpha})")

    # Gráfica de la normal o t con estadístico
    st.subheader("Gráfica de la distribución (ubicación del estadístico)")
    fig, ax = plt.subplots(figsize=(8,4))
    xs = np.linspace(-4, 4, 1000)
    if use_t_flag:
        # t distribution with df
        df_t = dfree
        ys = stats.t.pdf(xs, df_t)
        ax.plot(xs, ys)
        # vertical line at stat
        ax.axvline(stat, color="red", linestyle="--", label=f"t = {stat:.3f}")
    else:
        ys = stats.norm.pdf(xs)
        ax.plot(xs, ys)
        ax.axvline(stat, color="red", linestyle="--", label=f"z = {stat:.3f}")
    ax.legend()
    ax.set_title("Distribución nula y estadístico")
    st.pyplot(fig)

# --- Pestaña: Regresión lineal ---------------------------------------------
if menu == "Regresión lineal":
    st.header("Regresión lineal simple y correlación")
    uploaded = st.file_uploader("Sube un archivo CSV (misma que en Datos)", type=["csv"], key="r_csv")
    if uploaded is None:
        st.warning("Sube un CSV para usar esta pestaña.")
        st.stop()
    df = pd.read_csv(uploaded)
    st.subheader("Head de los datos")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Necesitas al menos 2 columnas numéricas para regresión.")
        st.stop()

    col_x = st.selectbox("Selecciona la variable independiente X", numeric_cols, index=0)
    col_y = st.selectbox("Selecciona la variable dependiente Y", [c for c in numeric_cols if c!=col_x], index=0)

    # Slider para tamaño de muestra
    total = df.shape[0]
    min_slider = 10 if total >= 10 else 1
    max_slider = total
    sample_size = st.slider("Tamaño de la muestra para regresión", min_value=min_slider, max_value=max_slider, value=min_slider, step=1, key="reg_slider")

    if sample_size < total:
        sample_df = df.sample(n=sample_size, random_state=123)
    else:
        # ordenar asc por X (como pediste)
        sample_df = df.sort_values(by=col_x).head(sample_size)

    st.write(f"Usando {sample_df.shape[0]} observaciones para la regresión")
    st.dataframe(sample_df[[col_x, col_y]].head(50))

    # Descarga de la muestra
    csv_bytes = sample_df[[col_x, col_y]].to_csv(index=False).encode('utf-8')
    st.download_button("Descargar muestra (CSV) - regresión", data=csv_bytes, file_name="muestra_regresion.csv", mime="text/csv")

    # Ajuste regresión
    reg = linear_regression(sample_df[col_x], sample_df[col_y])
    if not reg:
        st.error("No se pudo ajustar la regresión (pocos datos).")
        st.stop()

    st.subheader("Resultados de la regresión")
    st.write(f"Intercepto (a) = {reg['intercept']:.6f}")
    st.write(f"Pendiente (b) = {reg['slope']:.6f}")
    st.write(f"r (Pearson) = {reg['r']:.6f}")
    st.write(f"R² = {reg['r2']:.6f}")
    st.write(f"P-valor (pendiente) = {reg['pvalue']:.6g}")
    st.write("Resumen del modelo (texto):")
    st.code(reg["summary"])

    # Clasificación de la correlación en texto grande
    clase = classification_by_r(reg['r'])
    st.markdown(f"## Clasificación de correlación: **{clase}**")

    # Gráfica de dispersión + línea de regresión
    st.subheader("Gráfica: dispersión y recta de regresión")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(x=sample_df[col_x], y=sample_df[col_y], ax=ax)
    # linea predicha
    xs = np.linspace(sample_df[col_x].min(), sample_df[col_x].max(), 100)
    ys = reg['intercept'] + reg['slope'] * xs
    ax.plot(xs, ys, color='red')
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    st.pyplot(fig)

    # Test de significancia de la correlación (r) o pendiente: usar t con n-2
    st.subheader("Test: ¿es la pendiente significativa?")
    alpha = st.number_input("Nivel de significancia α (regresión)", min_value=0.0001, max_value=0.5, value=0.05, step=0.01, format="%.4f", key="alpha_reg")
    n = reg["n"]
    dfree = n - 2
    # t-statistic para la pendiente: estandar (b / stderr)
    t_stat = reg['slope'] / reg['stderr'] if reg['stderr'] != 0 else float('nan')
    pval = 2 * stats.t.sf(abs(t_stat), dfree)
    st.write(f"t = {t_stat:.4f}  (df = {dfree})")
    st.write(f"P-valor (two-sided) = {pval:.6f}")
    conclusion = "Rechazamos H0 (pendiente = 0)" if pval < alpha else "No rechazamos H0"
    st.markdown(f"### Conclusión: **{conclusion}** (α = {alpha})")

    # Distribución t (normal aproximada) con estadístico t
    st.subheader("Distribución t (n-2) y estadístico")
    fig, ax = plt.subplots(figsize=(8,4))
    xs = np.linspace(-4, 4, 1000)
    ys = stats.t.pdf(xs, dfree)
    ax.plot(xs, ys)
    ax.axvline(t_stat, color='red', linestyle='--', label=f"t = {t_stat:.3f}")
    ax.legend()
    st.pyplot(fig)
