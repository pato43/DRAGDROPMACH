import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time, hashlib
from datetime import datetime, timedelta

st.set_page_config(page_title="RetailLab Builder — ML Studio", page_icon="🧪", layout="wide")

# ---------- Estilos ----------
st.markdown("""
<style>
:root{ --bg:#0a0f1c; --panel:#0f1630; --ink:#e8edff; --muted:#a6b5ff; --line:#22305b; --chip:#111a3e; }
html,body,[data-testid="stAppViewContainer"]{background:radial-gradient(1200px 800px at 25% -10%, #0e1736 0%, var(--bg) 45%) fixed; color:var(--ink)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1228 0%, #091024 100%)}
.block-title{font-weight:900;font-size:2rem;margin:.1rem 0 .25rem}
.subtle{color:var(--muted);font-size:.95rem}
.card{border:1px solid var(--line); background:var(--panel); border-radius:18px; padding:16px}
.badge{display:inline-flex;gap:.4rem;align-items:center;padding:.2rem .5rem;border-radius:999px;border:1px solid var(--line);background:var(--chip);font-size:.78rem}
.logo-row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:.45rem 0 .8rem}
.logo-row img{height:44px;border-radius:8px;padding:5px;border:1px solid #22305b;background:#0f1630}
hr{border:none;border-top:1px solid var(--line);margin:.8rem 0}
</style>
""", unsafe_allow_html=True)

# ---------- Utils ----------
def _safe_rerun():
    try: st.rerun()
    except Exception:
        try: st.experimental_rerun()
        except Exception: pass

def _hash(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (10**8)

def make_demo_data(n_stores=25, n_days=60, n_products=40, seed=123):
    rng = np.random.default_rng(seed)
    start = datetime.today().date() - timedelta(days=n_days)
    regions = ["Norte","Sur","Oriente","Poniente","Centro"]
    cats = ["Abarrotes","Botanas","Bebidas","Lácteos","Limpieza","Higiene","Enlatados","Panificados"]
    stores = [f"S{1000+i}" for i in range(n_stores)]
    prods = [f"P{100+i}-{c[:3].upper()}" for i,c in zip(range(n_products), rng.choice(cats, n_products))]
    rows=[]
    for d in range(n_days):
        date = start + timedelta(days=d)
        for s in rng.choice(stores, size=rng.integers(n_stores-6, n_stores), replace=False):
            region = rng.choice(regions)
            for p in rng.choice(prods, size=rng.integers(6, 14), replace=False):
                cat = p.split("-")[1]
                price = float(rng.choice([9.9,12.5,14.9,19.9,24.9,29.9,34.9]))
                units = int(max(0, rng.normal(12, 6)))
                promo = rng.choice([0,1], p=[.8,.2])
                stockout = 1 if (rng.random()<0.04 and units<2) else 0
                sales = float(units * price * (1.2 if promo else 1.0))
                rows.append([str(date), s, region, p, cat, price, units, sales, promo, stockout])
    df = pd.DataFrame(rows, columns=["date","store_id","region","product","category","price","units","sales","promo","stockout"])
    df["date"] = pd.to_datetime(df["date"])
    return df

def derive_views(df: pd.DataFrame):
    v_store = (df.groupby("store_id")["sales"].sum().reset_index().sort_values("sales", ascending=False).head(50))
    v_prod  = (df.groupby(["product","category"])["sales"].sum().reset_index().sort_values("sales", ascending=False).head(50))
    v_daily = (df.groupby("date")["sales"].sum().reset_index())
    v_alerts= (df.groupby(["store_id","product"])["stockout"].sum().reset_index().rename(columns={"stockout":"stockout_days"})
                 .sort_values("stockout_days", ascending=False).head(50))
    return {"Ventas por tienda": v_store, "Top productos": v_prod, "Ventas diarias": v_daily, "Alertas stockout": v_alerts}

def nice_kpis(df):
    tot = df["sales"].sum()
    t_u = df["units"].sum()
    ticket = (df["sales"] / df["units"].replace(0, np.nan)).median()
    stores = df["store_id"].nunique()
    c = st.columns(4)
    c[0].metric("Ingresos totales", f"${tot:,.0f}")
    c[1].metric("Unidades vendidas", f"{t_u:,.0f}")
    c[2].metric("Ticket mediano", f"${ticket:,.2f}")
    c[3].metric("Tiendas activas", f"{stores}")

def gemma3_summary(df):
    cats = df["category"].value_counts().index.tolist()[:4]
    hot = df.groupby("product")["sales"].sum().sort_values(ascending=False).head(3).index.tolist()
    tip = np.random.choice([
        "Agrupa por región y día para ver estacionalidad.",
        "Precio y promo son fuertes; agrega lags 7/14.",
        "Clúster por ticket para detectar tiendas atípicas.",
        "Con un baseline bien validado tomas decisiones rápidas."
    ])
    st.chat_message("assistant").markdown(
        f"**Gemma3**: Categorías top ({', '.join(cats)}). "
        f"Productos con mayor tracción: {', '.join(hot)}. {tip}"
    )

def config_hash(cfg: dict) -> int:
    s = "|".join([f"{k}={cfg[k]}" for k in sorted(cfg.keys())])
    return _hash(s)

# ---- Simulaciones según modelo ----
def simulate_training(seed: int, df, cfg):
    rng = np.random.default_rng(seed + int(df["sales"].sum()) % 1000)
    metrics = {
        "RMSE ventas": round(rng.uniform(12, 35), 2),
        "MAPE %": round(rng.uniform(8, 18), 2),
        "ROC AUC (stockout)": round(rng.uniform(0.72, 0.92), 3),
        "Silhouette (clusters)": round(rng.uniform(0.32, 0.67), 3)
    }
    feats = ["price","promo","units_lag7","dow","region","category","promo_rolling"]
    imp_v = rng.random(len(feats)); imp_v /= imp_v.sum()
    imp = pd.DataFrame({"feature":feats,"importance":imp_v}).sort_values("importance", ascending=False)

    out = {}
    # Regresión/ARIMA: serie y residuales
    ts = df.groupby("date")["sales"].sum().reset_index().rename(columns={"sales":"y"})
    ts["yhat"] = ts["y"]*(1 + rng.normal(0, 0.08, size=len(ts)))
    out["reg"] = ts
    # Clasificación: probabilidades + matriz confusión + curva ROC (aprox)
    n = min(4000, len(df))
    samp = df.sample(n, random_state=seed).copy()
    z = 0.4*samp["promo"] + 0.3*(samp["units"]<2).astype(int) + 0.1*np.log1p(samp["price"]) + rng.normal(0,0.6,n)
    proba = 1/(1+np.exp(-z))
    samp["proba"] = proba
    samp["y_true"] = samp["stockout"].astype(int)
    samp["y_pred"] = (samp["proba"]>=0.5).astype(int)
    cm = pd.crosstab(samp["y_true"], samp["y_pred"]).reindex(index=[0,1], columns=[0,1], fill_value=0)
    thrs = np.linspace(0,1,25)
    tpr, fpr = [], []
    for t in thrs:
        yhat = (samp["proba"]>=t).astype(int)
        tp = ((yhat==1)&(samp["y_true"]==1)).sum()
        fp = ((yhat==1)&(samp["y_true"]==0)).sum()
        fn = ((yhat==0)&(samp["y_true"]==1)).sum()
        tn = ((yhat==0)&(samp["y_true"]==0)).sum()
        tpr.append(tp/(tp+fn+1e-9)); fpr.append(fp/(fp+tn+1e-9))
    out["clf"] = {"sample":samp, "cm":cm, "roc":pd.DataFrame({"fpr":fpr,"tpr":tpr})}
    # Clustering K-Means (proyección 2D por SVD)
    g = (df.groupby("store_id")
           .agg(sales=("sales","sum"), units=("units","sum"),
                promo_rate=("promo","mean"), price=("price","mean"))
           .reset_index())
    X = g[["sales","units","promo_rate","price"]].to_numpy()
    X = (X - X.mean(0))/X.std(0)
    U,S,Vt = np.linalg.svd(X, full_matrices=False)
    Z = U[:, :2]*S[:2]
    k = min(12, max(2, cfg["complexity"]+1))
    cent = Z[np.random.default_rng(seed).choice(len(Z), k, replace=False)]
    for _ in range(5):
        dist = ((Z[:,None,:]-cent[None,:,:])**2).sum(-1)
        lab = dist.argmin(1)
        for j in range(k):
            if (lab==j).any(): cent[j] = Z[lab==j].mean(0)
    # salida de clustering
    proj = pd.DataFrame({"x":Z[:,0],"y":Z[:,1],"cluster":lab,"store_id":g["store_id"]})
    sizes = pd.Series(lab).value_counts().sort_index()
    sizes = sizes.reset_index(name="size").rename(columns={"index":"cluster"})
    sizes["cluster"] = sizes["cluster"].astype(str)  # <- evita error de columnas para plotly
    out["clu"] = {"proj":proj, "sizes":sizes}
    # Forecast ARIMA-like naive
    horizon = int(cfg.get("horizon",14))
    ts2 = ts.copy()
    baseline = ts2["y"].rolling(7, min_periods=1).mean()
    future = []
    last_day = ts2["date"].iloc[-1]
    for h in range(1, horizon+1):
        val = float(baseline.iloc[-1]*(1+np.random.normal(0,0.02)))
        future.append({"date": last_day + timedelta(days=h), "yhat": val})
    out["fc"] = {"history":ts2[["date","y"]], "future":pd.DataFrame(future)}
    return metrics, imp, out

# ---------- Estado ----------
if "gate_ready" not in st.session_state: st.session_state.gate_ready = False
if "source_type" not in st.session_state: st.session_state.source_type = None
if "df_main" not in st.session_state: st.session_state.df_main = None
if "dfs" not in st.session_state: st.session_state.dfs = {}
if "auto_run" not in st.session_state: st.session_state.auto_run = True
if "model_cfg" not in st.session_state:
    st.session_state.model_cfg = {"target":"sales","algo":"Bosque aleatorio","cv":"KFold-5","complexity":5,"horizon":14}
if "last_hash" not in st.session_state: st.session_state.last_hash = None
if "last_metrics" not in st.session_state: st.session_state.last_metrics = None
if "last_imp" not in st.session_state: st.session_state.last_imp = None
if "last_out" not in st.session_state: st.session_state.last_out = None

# ---------- Sidebar ----------
def reset_all():
    for k in list(st.session_state.keys()):
        if k not in ["_is_running_with_streamlit"]: del st.session_state[k]
    _safe_rerun()

with st.sidebar:
    st.markdown("### 🧪 RetailLab Builder")
    st.toggle("Auto-ejecutar", key="auto_run", value=st.session_state.auto_run)
    st.button("🔄 Reiniciar", on_click=reset_all)
    st.markdown("---")
    st.markdown("**Plantillas rápidas**")
    c1,c2 = st.columns(2)
    if c1.button("Predicción ventas"):
        st.session_state.model_cfg.update({"target":"sales","algo":"Bosque aleatorio","cv":"KFold-5"})
        _safe_rerun()
    if c2.button("Alerta stockout"):
        st.session_state.model_cfg.update({"target":"stockout","algo":"Clasificador logístico","cv":"KFold-10"})
        _safe_rerun()
    if c1.button("Segmentar tiendas"):
        st.session_state.model_cfg.update({"target":"sales","algo":"K-Means","cv":"Hold-out 80/20"})
        _safe_rerun()
    if c2.button("Pronóstico 14d"):
        st.session_state.model_cfg.update({"target":"sales","algo":"ARIMA","cv":"Hold-out 80/20","horizon":14})
        _safe_rerun()

# ---------- Header + Logos en línea ----------
st.markdown('<div class="block-title">RetailLab Builder — ML Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle">Estudio visual para construir y evaluar modelos de ML orientados a retail.'
    ' Flujo: <b>1)</b> Carga una fuente, <b>2)</b> Elige modelo, <b>3)</b> Ejecuta y explora métricas y gráficos.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="logo-row">'
    '<img src="https://ai.google.dev/static/gemma/images/gemma3.png?hl=es-419" title="Gemma3" />'
    '<img src="https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/dark/mcp.png" title="MCP" />'
    '<img src="https://assets.streamlinehq.com/image/private/w_300,h_300,ar_1/f_auto/v1/icons/logos/langchain-ipuhh4qo1jz5ssl4x0g2a.png/langchain-dp1uxj2zn3752pntqnpfu2.png?_a=DATAg1AAZAA0" title="LangChain" />'
    '<img src="https://google.github.io/adk-docs/assets/agent-development-kit.png" title="Google ADK" />'
    '<img src="https://cdn-icons-png.flaticon.com/512/9159/9159105.png" title="CSV" />'
    '<img src="https://img.icons8.com/?size=512&id=117561&format=png" title="Excel" />'
    '<img src="https://mailmeteor.com/logos/assets/PNG/Google_Sheets_Logo_512px.png" title="Sheets" />'
    '<img src="https://upload.wikimedia.org/wikipedia/commons/a/ad/Logo_PostgreSQL.png" title="PostgreSQL" />'
    '<img src="https://cdn-icons-png.flaticon.com/512/5968/5968364.png" title="SQL Server" />'
    '</div>',
    unsafe_allow_html=True
)
st.caption("Gemma3, MCP, LangChain, Google ADK, y conectores típicos (CSV/Excel/Sheets/BBDD) — ilustrativo del stack previsto.")
st.divider()

# ==========================================================
#           PASO 1: PANTALLA DE FUENTE Y CARGA
# ==========================================================
if not st.session_state.gate_ready:
    st.markdown("**Selecciona fuente y carga el archivo ilustrativo. Gemma3 prepara vistas automáticamente.**")
    src = st.radio("Fuente de datos", ["CSV (local)","Base de datos","Excel Online / OneDrive","Google Sheets"], horizontal=True)
    st.session_state.source_type = src
    size = st.selectbox("Tamaño de muestra", ["Pequeño (demo)","Medio","Grande"], index=1)
    size_map = {"Pequeño (demo)": (12, 30, 18), "Medio": (25, 60, 40), "Grande": (40, 90, 70)}

    if src == "CSV (local)":
        file = st.file_uploader("Sube tu CSV ilustrativo", type=["csv"])
        if file and st.button("Procesar"):
            st.session_state.seed = _hash(file.name)
            with st.status("Gemma3 analizando el archivo…", expanded=True) as status:
                for t in ["Leyendo metadatos","Perfilando columnas","Detectando outliers","Generando vistas"]:
                    st.write("•", t); time.sleep(0.6)
                status.update(label="¡Listo! Data preparada.", state="complete")
            ns, nd, npd = size_map[size]
            df = make_demo_data(ns, nd, npd, seed=st.session_state.seed)
            st.session_state.df_main = df
            st.session_state.dfs = derive_views(df)
            st.session_state.gate_ready = True
            st.toast("Conexión completada", icon="✅")
            _safe_rerun()

    elif src == "Base de datos":
        st.markdown("#### Conectar a BBDD (ilustrativo)")
        c1,c2 = st.columns(2)
        with c1:
            st.selectbox("Motor", ["PostgreSQL","MySQL","SQL Server","Oracle","SQLite"])
            st.text_input("Host / Endpoint", "db.example.com")
            st.text_input("Puerto", "5432")
            st.text_input("Base de datos", "retail_demo")
        with c2:
            st.text_input("Usuario", "demo_user")
            st.text_input("Password", type="password")
            st.text_input("Tabla principal", "ventas")
        if st.button("Conectar y preparar"):
            with st.status("Conectando…", expanded=True) as status:
                for t in ["Autenticando","Leyendo esquema","Muestreando tabla","Preparando vistas"]:
                    st.write("•", t); time.sleep(0.6)
                status.update(label="Conexión establecida", state="complete")
            ns, nd, npd = size_map[size]
            df = make_demo_data(ns, nd, npd, seed=st.session_state.seed)
            st.session_state.df_main = df
            st.session_state.dfs = derive_views(df)
            st.session_state.gate_ready = True
            st.toast("Datos listos", icon="✅")
            _safe_rerun()

    elif src == "Excel Online / OneDrive":
        st.text_input("URL compartida del libro", "https://1drv.ms/x/s!demo")
        st.text_input("Hoja", "ventas")
        if st.button("Vincular y preparar"):
            with st.status("Vinculando Excel…", expanded=True) as status:
                for t in ["Verificando acceso","Leyendo rangos","Normalizando columnas","Creando vistas"]:
                    st.write("•", t); time.sleep(0.6)
                status.update(label="Libro vinculado", state="complete")
            ns, nd, npd = size_map[size]
            df = make_demo_data(ns, nd, npd, seed=st.session_state.seed)
            st.session_state.df_main = df
            st.session_state.dfs = derive_views(df)
            st.session_state.gate_ready = True
            st.toast("Datos listos", icon="✅")
            _safe_rerun()

    else:  # Google Sheets
        st.text_input("URL/ID de la hoja", "https://docs.google.com/spreadsheets/d/XXXXXXXX")
        st.text_input("Pestaña", "ventas")
        if st.button("Sincronizar y preparar"):
            with st.status("Sincronizando Sheets…", expanded=True) as status:
                for t in ["Validando hoja","Leyendo pestaña","Ajustando tipos","Derivando vistas"]:
                    st.write("•", t); time.sleep(0.6)
                status.update(label="Hoja sincronizada", state="complete")
            ns, nd, npd = size_map[size]
            df = make_demo_data(ns, nd, npd, seed=st.session_state.seed)
            st.session_state.df_main = df
            st.session_state.dfs = derive_views(df)
            st.session_state.gate_ready = True
            st.toast("Datos listos", icon="✅")
            _safe_rerun()

# ==========================================================
#     PASO 2: STUDIO NO-CODE + MÉTRICAS + GRÁFICOS DINÁMICOS
# ==========================================================
else:
    df = st.session_state.df_main

    st.info("**Cómo usar:** 1) Elige *Objetivo* y *Modelo*. 2) Ajusta *Validación*, *Complejidad* y *Horizonte*. "
            "3) Pulsa **Ejecutar** (o activa Auto-ejecutar). Mira **Métricas**, **Importancia** y **Gráficos**, que cambian según el modelo.")

    # KPIs + Gemma3
    c = st.columns([3,2])
    with c[0]: nice_kpis(df)
    with c[1]: gemma3_summary(df)
    st.divider()

    tab_build, tab_data, tab_graphs = st.tabs(["🎛️ Builder","🧾 DataFrames","📈 Gráficos"])

    # ---- BUILDER ----
    with tab_build:
        c1,c2,c3 = st.columns(3)
        with c1: st.selectbox("Objetivo", ["sales","units","stockout"], key="target", index=["sales","units","stockout"].index(st.session_state.model_cfg["target"]))
        with c2: st.selectbox("Modelo", ["Regresión lineal","Bosque aleatorio","XGBoost","Clasificador logístico","K-Means","ARIMA"], key="algo", index=["Regresión lineal","Bosque aleatorio","XGBoost","Clasificador logístico","K-Means","ARIMA"].index(st.session_state.model_cfg["algo"]))
        with c3: st.selectbox("Validación", ["Hold-out 80/20","KFold-5","KFold-10"], key="cv", index=["Hold-out 80/20","KFold-5","KFold-10"].index(st.session_state.model_cfg["cv"]))
        cc1,cc2 = st.columns(2)
        with cc1: st.slider("Complejidad", 1, 10, st.session_state.model_cfg["complexity"], key="complexity")
        with cc2: st.slider("Horizonte (días)", 7, 60, st.session_state.model_cfg["horizon"], key="horizon")

        st.session_state.model_cfg.update({
            "target": st.session_state.target,
            "algo": st.session_state.algo,
            "cv": st.session_state.cv,
            "complexity": st.session_state.complexity,
            "horizon": st.session_state.horizon
        })

        cfg_hash = config_hash(st.session_state.model_cfg)
        run_now = st.button("▶️ Ejecutar") or (st.session_state.auto_run and cfg_hash != st.session_state.last_hash)

        if run_now:
            with st.spinner("Gemma3 orquestando el flujo…"):
                time.sleep(0.25)
                m, imp, out = simulate_training(cfg_hash, df, st.session_state.model_cfg)
            st.session_state.last_hash = cfg_hash
            st.session_state.last_metrics = m
            st.session_state.last_imp = imp
            st.session_state.last_out = out

        if st.session_state.last_metrics is not None:
            m = st.session_state.last_metrics
            b1,b2,b3,b4 = st.columns(4)
            b1.metric("RMSE ventas", m["RMSE ventas"])
            b2.metric("MAPE %", m["MAPE %"])
            b3.metric("ROC AUC", m["ROC AUC (stockout)"])
            b4.metric("Silhouette", m["Silhouette (clusters)"])
            st.markdown("**Importancia de features**")
            fig_imp = px.bar(st.session_state.last_imp, x="importance", y="feature", orientation="h")
            fig_imp.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("#### Resumen rápido")
        cquick1,cquick2 = st.columns([3,2])
        with cquick1:
            g1 = px.line(df.groupby("date")["sales"].sum().reset_index(), x="date", y="sales", title="Ventas diarias")
            g1.update_layout(height=280, margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(g1, use_container_width=True)
        with cquick2:
            g2 = px.treemap(df, path=["category","product"], values="sales", title="Mix por categoría")
            g2.update_layout(height=280, margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(g2, use_container_width=True)

    # ---- DATA ----
    with tab_data:
        st.markdown("**DF general (editable)**")
        st.data_editor(df.head(300), use_container_width=True, height=320, num_rows="dynamic")
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Ventas por tienda**")
            st.data_editor(st.session_state.dfs["Ventas por tienda"], use_container_width=True, height=260)
            st.markdown("**Alertas stockout**")
            st.data_editor(st.session_state.dfs["Alertas stockout"], use_container_width=True, height=260)
        with c2:
            st.markdown("**Top productos**")
            st.data_editor(st.session_state.dfs["Top productos"], use_container_width=True, height=260)
            st.markdown("**Ventas diarias**")
            st.data_editor(st.session_state.dfs["Ventas diarias"], use_container_width=True, height=260)

    # ---- GRÁFICOS DINÁMICOS POR MODELO ----
    with tab_graphs:
        out = st.session_state.last_out
        algo = st.session_state.model_cfg["algo"]
        st.caption("Estas visualizaciones cambian según el modelo seleccionado en el Builder.")
        if out is None:
            st.info("Ejecuta el Builder para ver gráficos específicos del modelo.")
        else:
            if "Regresión" in algo or algo == "XGBoost":
                st.markdown("**Regresión: Real vs Predicción y residuales**")
                ts = out["reg"]
                f1 = go.Figure()
                f1.add_trace(go.Scatter(x=ts["date"], y=ts["y"], name="Real", mode="lines"))
                f1.add_trace(go.Scatter(x=ts["date"], y=ts["yhat"], name="Predicción", mode="lines"))
                f1.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(f1, use_container_width=True)
                resid = ts.copy(); resid["resid"] = resid["y"]-resid["yhat"]
                f2 = px.scatter(resid, x="yhat", y="resid", title="Residuales vs yhat", opacity=.7)
                f2.update_layout(height=320, margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(f2, use_container_width=True)

            elif "Clasificador" in algo:
                st.markdown("**Clasificación: Matriz de confusión y ROC**")
                cm = out["clf"]["cm"]
                z = cm.values
                f1 = go.Figure(data=go.Heatmap(z=z, x=["Pred 0","Pred 1"], y=["Real 0","Real 1"],
                                               colorscale="Blues", text=z, texttemplate="%{text}"))
                f1.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(f1, use_container_width=True)
                roc = out["clf"]["roc"]
                f2 = go.Figure()
                f2.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="ROC"))
                f2.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Azar", line=dict(dash="dash")))
                f2.update_layout(height=320, xaxis_title="FPR", yaxis_title="TPR",
                                 margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(f2, use_container_width=True)

            elif "K-Means" in algo:
                st.markdown("**Clustering: proyección 2D y tamaño de clúster**")
                proj = out["clu"]["proj"]
                f1 = px.scatter(proj, x="x", y="y", color=proj["cluster"].astype(str),
                                hover_data=["store_id"], title="Tiendas en 2D (SVD)")
                f1.update_layout(height=340, margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(f1, use_container_width=True)
                sizes = out["clu"]["sizes"]
                f2 = px.bar(sizes, x="cluster", y="size", title="Tamaño de clúster")
                f2.update_layout(height=280, margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(f2, use_container_width=True)

            elif "ARIMA" in algo:
                st.markdown("**Forecast: histórico y horizonte futuro**")
                hist = out["fc"]["history"]; fut = out["fc"]["future"]
                f1 = go.Figure()
                f1.add_trace(go.Scatter(x=hist["date"], y=hist["y"], name="Histórico", mode="lines"))
                f1.add_trace(go.Scatter(x=fut["date"], y=fut["yhat"], name="Pronóstico", mode="lines"))
                f1.update_layout(height=340, margin=dict(l=10,r=10,t=30,b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(f1, use_container_width=True)
            else:
                st.info("Selecciona un modelo para ver sus visualizaciones dedicadas.")
