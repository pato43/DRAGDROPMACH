import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time, hashlib, random
from datetime import datetime, timedelta

st.set_page_config(page_title="RetailLab Builder — ML Studio", page_icon="🧪", layout="wide")

# ---------- Estilos ----------
st.markdown("""
<style>
:root{ --bg:#0a0f1c; --panel:#0f1630; --ink:#e8edff; --muted:#a6b5ff; --line:#22305b; --chip:#111a3e; }
html,body,[data-testid="stAppViewContainer"]{background:radial-gradient(1200px 800px at 25% -10%, #0e1736 0%, var(--bg) 45%) fixed; color:var(--ink)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1228 0%, #091024 100%)}
.block-title{font-weight:900;font-size:2rem;margin:.1rem 0 .4rem}
.subtle{color:var(--muted);font-size:.95rem}
.card{border:1px solid var(--line); background:var(--panel); border-radius:18px; padding:16px}
.kpi{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.kpi>div{border:1px solid var(--line);background:#0e1736;border-radius:14px;padding:12px}
hr{border:none;border-top:1px solid var(--line);margin:1rem 0}
.logo-grid{display:grid;grid-template-columns:repeat(6, minmax(0,1fr));gap:10px;align-items:center}
.logo-grid img{width:100%;max-height:48px;object-fit:contain;background:#0f1630;border-radius:10px;padding:6px;border:1px solid #22305b}
.hero{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:10px}
.badge{display:inline-flex;gap:.4rem;align-items:center;padding:.25rem .6rem;border-radius:999px;border:1px solid var(--line);background:var(--chip);font-size:.78rem}
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
    cols = st.columns(4)
    with cols[0]: st.metric("Ingresos totales", f"${tot:,.0f}")
    with cols[1]: st.metric("Unidades vendidas", f"{t_u:,.0f}")
    with cols[2]: st.metric("Ticket mediano", f"${ticket:,.2f}")
    with cols[3]: st.metric("Tiendas activas", f"{stores}")

def gemma3_summary(df):
    cats = df["category"].value_counts().index.tolist()[:4]
    hot = df.groupby("product")["sales"].sum().sort_values(ascending=False).head(3).index.tolist()
    tip = np.random.choice([
        "Agrupa por región y día para encontrar ‘lunes flojos’.",
        "Precio y promo son features clave; añade lags de 7/14 días.",
        "Un clúster por ticket ayuda a detectar tiendas con mix atípico.",
        "Con regresión baseline + validación cruzada ya decides prioridades."
    ])
    st.chat_message("assistant").markdown(
        f"**Gemma3**: Detecté categorías dominantes ({', '.join(cats)}). "
        f"Productos con mayor tracción: {', '.join(hot)}. {tip}"
    )

def simulate_training(config_seed: int, df):
    rng = np.random.default_rng(config_seed + int(df["sales"].sum()) % 1000)
    metrics = {
        "RMSE ventas": round(rng.uniform(12, 35), 2),
        "MAPE %": round(rng.uniform(8, 18), 2),
        "ROC AUC (stockout)": round(rng.uniform(0.72, 0.92), 3),
        "Silhouette (clusters)": round(rng.uniform(0.32, 0.67), 3)
    }
    feats = ["price","promo","units_lag7","dow","region","category","promo_rolling"]
    importances = rng.random(len(feats)); importances /= importances.sum()
    imp = pd.DataFrame({"feature":feats, "importance":importances}).sort_values("importance", ascending=False)
    return metrics, imp

def config_hash(cfg: dict) -> int:
    s = "|".join([f"{k}={cfg[k]}" for k in sorted(cfg.keys())])
    return _hash(s)

def plot_sales(df):
    c1, c2 = st.columns([3,2])
    with c1:
        g1 = px.line(df.groupby("date")["sales"].sum().reset_index(), x="date", y="sales", title="Ventas diarias")
        g1.update_layout(height=360, margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(g1, use_container_width=True)
    with c2:
        g2 = px.treemap(df, path=["category","product"], values="sales", title="Mix por categoría")
        g2.update_layout(height=360, margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(g2, use_container_width=True)

def plot_more_graphs(df):
    c1,c2 = st.columns(2)
    with c1:
        top_stores = df.groupby("store_id")["sales"].sum().nlargest(12).reset_index()
        fig = px.bar(top_stores, x="store_id", y="sales", title="Top tiendas por ventas")
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        df2 = df.sample(min(2000, len(df)), random_state=42)
        fig2 = px.scatter(df2, x="price", y="units", color="category", opacity=.6, title="Precio vs Unidades")
        fig2.update_layout(height=360, margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
    w = df.assign(dow=df["date"].dt.day_name()).groupby("dow")["sales"].sum().reindex(
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    ).reset_index().rename(columns={"dow":"día"})
    w["día"] = w["día"].map({
        "Monday":"Lunes","Tuesday":"Martes","Wednesday":"Miércoles","Thursday":"Jueves",
        "Friday":"Viernes","Saturday":"Sábado","Sunday":"Domingo"})
    fig3 = px.bar(w, x="día", y="sales", title="Ventas por día de la semana")
    fig3.update_layout(height=320, margin=dict(l=10,r=10,t=40,b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)

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

# ---------- Sidebar ----------
def reset_all():
    for k in list(st.session_state.keys()):
        if k not in ["_is_running_with_streamlit"]: del st.session_state[k]
    _safe_rerun()

with st.sidebar:
    st.markdown("### 🧪 RetailLab Builder")
    st.toggle("Auto-ejecutar", key="auto_run", value=st.session_state.auto_run)  # ÚNICO toggle con esta key
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

# ==========================================================
#           PASO 1: PANTALLA DE FUENTE Y CARGA
# ==========================================================
if not st.session_state.gate_ready:
    st.markdown('<div class="block-title">RetailLab Builder — ML Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero">'
        '<span class="badge">Gemma3 · Analítica asistida</span>'
        '<span class="badge">MCP · Conectores</span>'
        '<span class="badge">APIs · Integración</span>'
        '<span class="badge">ADK Agents</span>'
        '<span class="badge">LangChain · Orquestación</span>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="subtle">Selecciona una fuente y carga un archivo. Luego Gemma3 prepara vistas y gráficos listos para usar.</div>', unsafe_allow_html=True)
    st.divider()

    # Logos del stack
    with st.expander("Stack & Conectores (ilustrativo)"):
        cols = st.columns(6)
        cols[0].image("https://ai.google.dev/static/gemma/images/gemma3.png?hl=es-419", caption="Gemma3", use_column_width=True)
        cols[1].image("https://assets.streamlinehq.com/image/private/w_300,h_300,ar_1/f_auto/v1/icons/logos/langchain-ipuhh4qo1jz5ssl4x0g2a.png/langchain-dp1uxj2zn3752pntqnpfu2.png?_a=DATAg1AAZAA0", caption="LangChain", use_column_width=True)
        cols[2].image("https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/dark/mcp.png", caption="MCP", use_column_width=True)
        cols[3].image("https://cdn-icons-png.flaticon.com/512/9159/9159105.png", caption="CSV", use_column_width=True)
        cols[4].image("https://img.icons8.com/?size=512&id=117561&format=png", caption="Excel", use_column_width=True)
        cols[5].image("https://mailmeteor.com/logos/assets/PNG/Google_Sheets_Logo_512px.png", caption="Sheets", use_column_width=True)

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
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Motor", ["PostgreSQL","MySQL","SQL Server","Oracle","SQLite"])
            st.text_input("Host / Endpoint", "db.example.com")
            st.text_input("Puerto", "5432")
            st.text_input("Base de datos", "retail_demo")
        with col2:
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
        st.markdown("#### Conectar Excel Online / OneDrive (ilustrativo)")
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
        st.markdown("#### Conectar Google Sheets (ilustrativo)")
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
#           PASO 2: STUDIO NO-CODE + VISUALIZACIÓN
# ==========================================================
else:
    st.markdown('<div class="block-title">RetailLab Builder — ML Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Datos listos. Gemma3 organizó un DF general y vistas derivadas. Ajusta tu modelo sin código y observa métricas y gráficos en vivo.</div>', unsafe_allow_html=True)
    st.divider()

    df = st.session_state.df_main

    cols = st.columns([3,2,1])
    with cols[0]: nice_kpis(df)
    with cols[1]: gemma3_summary(df)
    with cols[2]:
        st.markdown("**Fuente**"); st.write(st.session_state.source_type); st.write("Semilla:", st.session_state.seed)

    tab_build, tab_data, tab_graphs, tab_about = st.tabs(["🎛️ Builder","🧾 DataFrames","📈 Gráficos","ℹ️ Stack & Conectores"])

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
        manual = st.button("▶️ Ejecutar")
        run_now = manual or (st.session_state.auto_run and cfg_hash != st.session_state.last_hash)

        if run_now:
            with st.spinner("Gemma3 orquestando el flujo…"):
                time.sleep(0.2)
                metrics, imp = simulate_training(cfg_hash, df)
            st.session_state.last_hash = cfg_hash
            st.session_state.last_metrics = metrics
            st.session_state.last_imp = imp

        if st.session_state.last_metrics is not None:
            m = st.session_state.last_metrics
            b1,b2,b3,b4 = st.columns(4)
            b1.metric("RMSE ventas", m["RMSE ventas"])
            b2.metric("MAPE %", m["MAPE %"])
            b3.metric("ROC AUC", m["ROC AUC (stockout)"])
            b4.metric("Silhouette", m["Silhouette (clusters)"])
            st.markdown("**Importancia de features**")
            fig_imp = px.bar(st.session_state.last_imp, x="importance", y="feature", orientation="h")
            fig_imp.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("#### Visuales rápidos")
        plot_sales(df)

    # ---- DATA ----
    with tab_data:
        st.markdown("#### DF general (editable)")
        st.data_editor(df.head(300), use_container_width=True, height=320, num_rows="dynamic")
        v1, v2 = st.columns(2)
        with v1:
            st.markdown("**Ventas por tienda**")
            st.data_editor(st.session_state.dfs["Ventas por tienda"], use_container_width=True, height=260)
            st.markdown("**Alertas stockout**")
            st.data_editor(st.session_state.dfs["Alertas stockout"], use_container_width=True, height=260)
        with v2:
            st.markdown("**Top productos**")
            st.data_editor(st.session_state.dfs["Top productos"], use_container_width=True, height=260)
            st.markdown("**Ventas diarias**")
            st.data_editor(st.session_state.dfs["Ventas diarias"], use_container_width=True, height=260)

    # ---- GRÁFICOS ----
    with tab_graphs:
        plot_more_graphs(df)

    # ---- ABOUT ----
    with tab_about:
        st.markdown("### Arquitectura y Conectores (ilustrativo)")
        st.markdown("- **Gemma3** como analista de datos asistido.\n- **MCP** para conectores unificados.\n- **APIs** para ingestión/serving.\n- **ADK Agents** para flujos asistidos.\n- **LangChain** para orquestación de herramientas.")
        st.markdown("#### Logos")
        grid = st.container()
        with grid:
            c = st.columns(6)
            c[0].image("https://ai.google.dev/static/gemma/images/gemma3.png?hl=es-419", caption="Gemma3", use_column_width=True)
            c[1].image("https://assets.streamlinehq.com/image/private/w_300,h_300,ar_1/f_auto/v1/icons/logos/langchain-ipuhh4qo1jz5ssl4x0g2a.png/langchain-dp1uxj2zn3752pntqnpfu2.png?_a=DATAg1AAZAA0", caption="LangChain", use_column_width=True)
            c[2].image("https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/dark/mcp.png", caption="MCP", use_column_width=True)
            c[3].image("https://cdn-icons-png.flaticon.com/512/9159/9159105.png", caption="CSV", use_column_width=True)
            c[4].image("https://img.icons8.com/?size=512&id=117561&format=png", caption="Excel", use_column_width=True)
            c[5].image("https://mailmeteor.com/logos/assets/PNG/Google_Sheets_Logo_512px.png", caption="Sheets", use_column_width=True)
        st.caption("Las integraciones mostradas corresponden al stack previsto de la plataforma.")
