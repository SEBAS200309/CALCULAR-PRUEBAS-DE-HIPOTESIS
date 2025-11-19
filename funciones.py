# funciones.py
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from typing import Tuple, Dict

def estadisticas_basicas(series: pd.Series) -> Dict[str, float]:
    """Calcula media, desviación estándar (muestral y poblacional), n"""
    x = series.dropna().astype(float)
    n = x.size
    media = x.mean()
    std_muestral = x.std(ddof=1)
    std_poblacional = x.std(ddof=0)
    return {"n": int(n), "mean": float(media), "std_m": float(std_muestral), "std_p": float(std_poblacional)}

def pearson_r(x: pd.Series, y: pd.Series) -> float:
    """Calcula r de Pearson (ignora NaNs alineando índices)."""
    df = pd.concat([x, y], axis=1).dropna()
    if df.shape[0] < 2:
        return float("nan")
    return float(df.iloc[:,0].corr(df.iloc[:,1]))

def linear_regression(x: pd.Series, y: pd.Series) -> Dict:
    """
    Ajuste de regresión lineal simple usando statsmodels.
    Devuelve slope, intercept, r, r2, pvalue_slope, stderr_slope, summary (texto opcional)
    """
    df = pd.concat([x, y], axis=1).dropna()
    if df.shape[0] < 2:
        return {}
    X = sm.add_constant(df.iloc[:,0])
    model = sm.OLS(df.iloc[:,1], X).fit()
    slope = float(model.params[1])
    intercept = float(model.params[0])
    r = float(df.iloc[:,0].corr(df.iloc[:,1]))
    r2 = float(model.rsquared)
    pvalue = float(model.pvalues[1])
    stderr = float(model.bse[1])
    summary = model.summary().as_text()
    return {"slope": slope, "intercept": intercept, "r": r, "r2": r2, "pvalue": pvalue, "stderr": stderr, "n": int(df.shape[0]), "summary": summary}

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
