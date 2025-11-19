# funciones.py
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from typing import Dict

def estadisticas_basicas(series: pd.Series) -> Dict[str, float]:
    """Calcula media, desviación estándar (muestral y poblacional), n"""
    x = series.dropna().astype(float)
    n = x.size
    media = x.mean() if n > 0 else float("nan")
    std_muestral = x.std(ddof=1) if n > 1 else float("nan")
    std_poblacional = x.std(ddof=0) if n > 0 else float("nan")
    return {"n": int(n), "mean": float(media), "std_m": float(std_muestral if not np.isnan(std_muestral) else 0.0), "std_p": float(std_poblacional if not np.isnan(std_poblacional) else 0.0)}

def pearson_r(x: pd.Series, y: pd.Series, n:int) -> float:
    """Calcula r de Pearson (ignora NaNs alineando índices)."""
    df = pd.concat([x, y], axis=1).dropna()
    if df.shape[0] < 2:
        return float("nan")
    x = df.iloc[:,0].astype(float)
    y = df.iloc[:,1].astype(float)
    xbar = x.mean()
    ybar = y.mean()
    num = ((x - xbar) * (y - ybar)).sum()
    denom = (n-1) * x.std(ddof=1) * y.std(ddof=1)
    if denom == 0:
        return float("nan")
    r = num / denom
    return float(r)

def linear_regression(x: pd.Series, y: pd.Series) -> Dict:
    """
    Ajuste de regresión lineal simple usando statsmodels.
    Devuelve slope, intercept, r, r2, pvalue_slope, stderr_slope, n, summary.
    """
    df = pd.concat([x, y], axis=1).dropna()
    if df.shape[0] < 2:
        return {}
    X = sm.add_constant(df.iloc[:,0])
    model = sm.OLS(df.iloc[:,1], X).fit()
    slope = float(model.params[1])
    intercept = float(model.params[0])
    r = pearson_r(df.iloc[:,0], df.iloc[:,1], int(df.shape[0]))
    pvalue = float(model.pvalues[1])
    stderr = float(model.bse[1])
    summary = model.summary().as_text()
    return {"slope": slope, "intercept": intercept, "r": r, "pvalue": pvalue, "stderr": stderr, "n": int(df.shape[0]), "summary": summary}

def classification_by_r(r_value: float) -> str:
    """Clasifica la fuerza de la correlación según |r|"""
    if np.isnan(r_value):
        return "No calculable (pocos datos)"
    r = abs(r_value)
    if r >= 0.95:
        return "Perfecta o casi perfecta"
    if r >= 0.9:
        return "Muy fuerte"
    if r >= 0.7:
        return "Fuerte"
    if r >= 0.5:
        return "Moderada"
    if r >= 0.3:
        return "Débil"
    return "Muy débil o nula"

def test_mean_z(xbar: float, mu0: float, sigma: float, n: int) -> Dict:
    """Estadístico z y p-valor para diferencia de medias (sigma poblacional conocido)"""
    se = sigma / np.sqrt(n)
    z = (xbar - mu0) / se
    return {"stat": float(z), "se": float(se)}

def test_mean_t(xbar: float, mu0: float, s: float, n: int) -> Dict:
    """Estadístico t y se (s es desviación muestral)"""
    se = s / np.sqrt(n)
    t = (xbar - mu0) / se
    df = n - 1
    return {"stat": float(t), "se": float(se), "df": int(df)}

def p_value_from_stat(stat: float, df: int = None, test_type: str = "two-sided", use_t: bool=False) -> float:
    """
    Calcula el p-valor según el estadístico.
    test_type: "two-sided", "greater", "less"
    use_t: si True usa la t-student con df, si False usa la normal
    """
    if use_t:
        if df is None:
            raise ValueError("df required for t distribution")
        if test_type == "two-sided":
            p = 2 * stats.t.sf(abs(stat), df)
        elif test_type == "greater":
            p = stats.t.sf(stat, df)
        else:  # less
            p = stats.t.cdf(stat, df)
    else:
        if test_type == "two-sided":
            p = 2 * stats.norm.sf(abs(stat))
        elif test_type == "greater":
            p = stats.norm.sf(stat)
        else:
            p = stats.norm.cdf(stat)
    return float(p)
