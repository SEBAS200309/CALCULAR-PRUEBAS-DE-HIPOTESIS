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

# --- Helpers ---------------------------------------------------------------
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
    if op_h1 in ["!="]:
        return "two-sided"
    if op_h1 in [">", ">="]:
        return "greater"
    if op_h1 in ["<", "<="]:
        return "less"
    return "two-sided"

# Sidebar menu
st.sidebar.title("Menú")
menu = st.sidebar.selectbox("Selecciona sección", ["Datos", "Pruebas de hipótesis", "Regresión lineal"])

# --- Datos tab -------------------------------------------------------------
if menu == "Datos":
    st.header("Carga de datos")
    uploaded = st.file_uploader("Sube un archivo CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.subheader("Vista previa (head)")
        st.dataframe(df.head())
        st.markdown(f"**Filas:** {df.shape[0]} — **Columnas:** {df.shape[1]}")
        st.markdown("**Columnas detectadas (y tipos):**")
        st.write(df.dtypes)
    else:
        st.info("Sube un CSV para comenzar. Las pestañas siguientes también pedirán el CSV si no lo subiste aquí.")

# --- Pruebas de hipótesis tab ---------------------------------------------
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

    # Slider muestra
    total = df.shape[0]
    min_slider = 10 if total >= 10 else 1
    max_slider = total
    sample_size = st.slider("Tamaño de la muestra", min_value=min_slider, max_value=max_slider, value=min_slider, step=1)

    if sample_size < total:
        sample_df = df.sample(n=sample_size, random_state=42).sort_index().reset_index(drop=True)
    else:
        numeric_cols_all = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols_all:
            sample_df = (
                df.sort_values(by=numeric_cols_all[0])
                    .head(sample_size)
                    .reset_index(drop=True)
            )
        else:
            sample_df = df.head(sample_size).reset_index(drop=True)

    # Mostrar
    st.subheader("Muestra seleccionada")
    st.write(f"Registros mostrados: {sample_df.shape[0]}")
    st.dataframe(sample_df)

    csv_bytes = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button("Descargar muestra (CSV)", data=csv_bytes, file_name="muestra.csv", mime="text/csv")

    stats_res = estadisticas_basicas(sample_df[col])
    st.write("Estadísticas básicas:")
    st.metric("n", stats_res["n"])
    st.metric("Media (x̄)", f"{stats_res['mean']:.4f}")
    st.metric("Desv. estándar (muestral)", f"{stats_res['std_m']:.4f}")

    st.subheader("Formulación de hipótesis")
    col1, col2 = st.columns(2)
    with col1:
        h0_value = st.number_input("Valor H0 (media µ0)", value=float(stats_res["mean"]), format="%.6f")
        op_h0 = st.selectbox("Operador H0", ["=", "!=", ">=", "<=", ">", "<"], index=0)
    with col2:
        suggested_h1 = complementary(op_h0)
        ops = ["=", "!=", ">", "<", ">=", "<="]
        op_h1 = st.selectbox("Operador H1 (sugerido)", ops, index=ops.index(suggested_h1) if suggested_h1 in ops else 1)
        st.write("H1 será interpretada automáticamente para calcular p-valor.")

    st.subheader("Tipo de estadístico")
    use_z = st.radio("¿Usar Z o T?", options=["Z (varianza poblacional conocida)", "T (varianza muestral)"])
    if use_z.startswith("Z"):
        sigma_pop = st.number_input("Ingrese desviación estándar poblacional σ (si la conoce)", value=float(stats_res["std_p"]) if stats_res["std_p"]>0 else 1.0, format="%.6f")
    else:
        sigma_pop = None

    alpha = st.number_input("Nivel de significancia α", min_value=0.0001, max_value=0.5, value=0.05, step=0.01, format="%.4f")

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

    st.subheader("Gráfica de la distribución (ubicación del estadístico)")
    fig, ax = plt.subplots(figsize=(8,4))
    xs = np.linspace(-4, 4, 1000)
    if use_t_flag:
        df_t = dfree
        ys = stats.t.pdf(xs, df_t)
        ax.plot(xs, ys)
        ax.axvline(stat, color="red", linestyle="--", label=f"t = {stat:.3f}")
    else:
        ys = stats.norm.pdf(xs)
        ax.plot(xs, ys)
        ax.axvline(stat, color="red", linestyle="--", label=f"z = {stat:.3f}")
    ax.legend()
    ax.set_title("Distribución nula y estadístico")
    st.pyplot(fig)

# --- Regresión tab ---------------------------------------------------------
if menu == "Regresión lineal":
    st.header("Regresión lineal simple y correlación")
    uploaded = st.file_uploader("Sube un archivo CSV (misma que en Datos)", type=["csv"], key="r_csv")
    if uploaded is None:
        st.warning("Sube un CSV para usar esta pestaña.")
        st.stop()
    df = pd.read_csv(uploaded)
    st.subheader("Head de los datos")
    st.dataframe(df.head())
    st.subheader("Fórmulas utilizadas (Mínimos Cuadrados)")
    st.latex(r"b = r \cdot \frac{S_y}{S_x}")
    st.latex(r"a = \bar{Y} - b\bar{X}")
    st.latex(r"\hat{Y} = a + bX")


    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Necesitas al menos 2 columnas numéricas para regresión.")
        st.stop()

    col_x = st.selectbox("Selecciona la variable independiente X", numeric_cols, index=0)
    col_y = st.selectbox("Selecciona la variable dependiente Y", [c for c in numeric_cols if c!=col_x], index=0)

    total = df.shape[0]
    min_slider = 10 if total >= 10 else 1
    max_slider = total
    sample_size = st.slider("Tamaño de la muestra para regresión", min_value=min_slider, max_value=max_slider, value=min_slider, step=1, key="reg_slider")

    # seleccionar muestra y resetear índices para mostrar limpia
    if sample_size < total:
        sample_df = df.sample(n=sample_size, random_state=123).reset_index(drop=True)
    else:
        sample_df = df.sort_values(by=col_x).head(sample_size).reset_index(drop=True)

    st.write(f"Usando {sample_df.shape[0]} observaciones para la regresión")
    st.dataframe(sample_df[[col_x, col_y]].head(50))

    csv_bytes = sample_df[[col_x, col_y]].to_csv(index=False).encode('utf-8')
    st.download_button("Descargar muestra (CSV) - regresión", data=csv_bytes, file_name="muestra_regresion.csv", mime="text/csv")

    # === usar la función linear_regression para todos los valores del modelo ===
    reg = linear_regression(sample_df[col_x], sample_df[col_y])
    if not reg:
        st.error("No se pudo ajustar la regresión (pocos datos).")
        st.stop()

    intercept = reg['intercept']
    slope = reg['slope']
    r_val = reg['r']
    stderr = reg['stderr']
    n = reg['n']

    st.subheader("Resultados de la regresión")
    st.write(f"Intercepto (a) = {intercept:.6f}")
    st.write(f"Pendiente (b) = {slope:.6f}")
    st.write(f"r (Pearson) = {r_val:.6f}")
    st.write(f"P-valor (pendiente) = {reg['pvalue']:.6g}")
    st.write("Resumen del modelo (texto):")
    st.code(reg["summary"])

    clase = classification_by_r(r_val)
    st.markdown(f"## Clasificación de correlación: **{clase}**")

    # Gráfica: dispersión y recta calculada por la función
    st.subheader("Gráfica: dispersión y recta de regresión (usando intercept y slope de la función)")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(x=sample_df[col_x], y=sample_df[col_y], ax=ax)
    xs_line = np.linspace(sample_df[col_x].min(), sample_df[col_x].max(), 200)
    ys_line = intercept + slope * xs_line
    ax.plot(xs_line, ys_line, color='red', label='Recta (función)')
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.legend()
    st.pyplot(fig)

    # Predicción interactiva usando intercept y slope devueltos por la función
    st.subheader("Predicción interactiva (usar resultados de linear_regression)")
    x_input = st.number_input(f"Ingrese un valor de {col_x} para predecir {col_y}:", value=float(sample_df[col_x].median()))
    y_pred = intercept + slope * x_input
    st.markdown("### Ecuación usada:")
    st.latex(fr"\hat{{Y}} = {intercept:.4f} + {slope:.4f} X")
    st.write(f"**Predicción Ŷ para {col_x} = {x_input} : {y_pred:.4f}**")

    # Test de significancia de la pendiente y gráfico t con colas
    st.subheader("Test: ¿es la pendiente significativa?")
    alpha = st.number_input("Nivel de significancia α (regresión)", min_value=0.0001, max_value=0.5, value=0.05, step=0.01, format="%.4f", key="alpha_reg")

    dfree = n - 2
    t_stat = slope / stderr if stderr != 0 else float('nan')
    pval = 2 * stats.t.sf(abs(t_stat), dfree)

    st.write(f"t = {t_stat:.4f}  (df = {dfree})")
    st.write(f"P-valor (two-sided) = {pval:.6f}")

    conclusion = "Rechazamos H0 (pendiente = 0)" if pval < alpha else "No rechazamos H0"
    st.markdown(f"### Conclusión: **{conclusion}** (α = {alpha})")

    # -------- GRÁFICA ---------
    st.subheader("Distribución t (n-2) y zonas de rechazo")
    fig, ax = plt.subplots(figsize=(8,4))

    xs_t = np.linspace(-4, 4, 1500)
    ys_t = stats.t.pdf(xs_t, dfree)
    ax.plot(xs_t, ys_t, label="Distribución t")

    # Valor crítico bilateral
    t_crit = stats.t.ppf(1 - alpha/2, dfree)
    x_left = xs_t[xs_t < -t_crit]
    x_right = xs_t[xs_t > t_crit]
    ax.fill_between(x_left, 0, stats.t.pdf(x_left, dfree), color="orange", alpha=0.5)
    ax.fill_between(x_right, 0, stats.t.pdf(x_right, dfree), color="orange", alpha=0.5)

    # Líneas críticas y estadístico
    ax.axvline(-t_crit, color='orange', linestyle='--', label=f"± t crítico = {t_crit:.3f}")
    ax.axvline(t_crit, color='orange', linestyle='--')
    ax.axvline(t_stat, color='red', linestyle='--', linewidth=2, label=f"t = {t_stat:.3f}")

    ax.set_xlabel("t")
    ax.set_ylabel("densidad")
    ax.legend()
    st.pyplot(fig)