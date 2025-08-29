import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time, hashlib, random
from datetime import datetime, timedelta

st.set_page_config(page_title="RetailLab Builder — ML Studio", page_icon="🧪", layout="wide")

# --------- Estilos ---------
st.markdown("""
<style>
:root{
  --bg:#0a0f1c; --panel:#0f1630; --ink:#e8edff; --muted:#a6b5ff; --line:#22305b;
  --chip:#111a3e; --accent:#6ee7ff; --accent2:#a78bfa; --ok:#22c55e; --warn:#f59e0b;
}
html, body, [data-testid="stAppViewContainer"]{background:radial-gradient(1200px 800px at 25% -10%, #0e1736 0%, var(--bg) 45%) fixed; color:var(--ink)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1228 0%, #091024 100%)}
.block-title{font-weight:900;font-size:2rem;margin:.1rem 0 .4rem}
.subtle{color:var(--muted);font-size:.95rem}
.card{border:1px solid var(--line); background:var(--panel); border-radius:18px; padding:16px}
.badge{display:inline-flex;gap:.45rem;align-items:center;padding:.25rem .6rem;border-radius:999px;
  border:1px solid var(--line);background:var(--chip);font-size:.78rem}
.kpi{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.kpi>div{border:1px solid var(--line);background:#0e1736;border-radius:14px;padding:12px}
a{color:#7dd3fc}
hr{border:none;border-top:1px solid var(--line);margin:1rem 0}
.small{font-size:.9rem}
.btn-row{display:flex;gap:.5rem;flex-wrap:wrap}
</style>
""", unsafe_allow_html=True)

# --------- Estado ---------
if "seed" not in st.session_state: st.session_state.seed = 123
if "source_done" not in st.session_state: st.session_state.source_done = False
if "source_type" not in st.session_state: st.session_state.source_type = None
if "df_main" not in st.session_state: st.session_state.df_main = None
if "dfs" not in st.session_state: st.session_state.dfs = {}
if "pipeline" not in st.session_state: st.session_state.pipeline = []
if "report_log" not in st.session_state: st.session_state.report_log = []

random.seed(st.session_state.seed)
np.random.seed(st.session_state.seed)

# --------- Utils & demo data ---------
def _hash(s: str) -> int:
    import hashlib
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (10**8)

def make_demo_data(n_stores=25, n_days=60, n_products=40):
    rng = np.random.default_rng(st.session_state.seed)
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
    v_store = (df.groupby("store_id")["sales"].sum().reset_index()
                 .sort_values("sales", ascending=False).head(50))
    v_prod = (df.groupby(["product","category"])["sales"].sum().reset_index()
                .sort_values("sales", ascending=False).head(50))
    v_daily = (df.groupby("date")["sales"].sum().reset_index())
    v_alerts = (df.groupby(["store_id","product"])["stockout"].sum().reset_index()
                  .rename(columns={"stockout":"stockout_days"})
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

def simulate_training(pipeline_steps, df):
    seed = len("".join(pipeline_steps)) + int(df["sales"].sum()) % 1000
    rng = np.random.default_rng(seed)
    time.sleep(0.7)
    metrics = {
        "RMSE ventas": round(rng.uniform(12, 35), 2),
        "MAPE %": round(rng.uniform(8, 18), 2),
        "ROC AUC (stockout)": round(rng.uniform(0.72, 0.92), 3),
        "Silhouette (clusters)": round(rng.uniform(0.32, 0.67), 3)
    }
    feats = ["price","promo","units_lag7","dow","region","category","promo_rolling"]
    importances = rng.random(len(feats)); importances /= importances.sum()
    return metrics, pd.DataFrame({"feature":feats, "importance":importances}).sort_values("importance", ascending=False)

def plot_feature_importance(df_imp):
    fig = px.bar(df_imp, x="importance", y="feature", orientation="h")
    fig.update_layout(height=380, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

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

def reset_all():
    for k in ["seed","source_done","source_type","df_main","dfs","pipeline","report_log"]:
        if k in st.session_state: del st.session_state[k]
    st.experimental_rerun()

# --------- Sidebar ---------
with st.sidebar:
    st.markdown("### 🧪 RetailLab Builder")
    st.button("🔄 Reiniciar", on_click=reset_all)
    st.markdown("---")
    st.markdown("**Modo asistido**")
    if "pipeline" not in st.session_state: st.session_state.pipeline = []
    add = st.selectbox("Agregar bloque", ["—","Limpiar nulos","One-Hot Encode","Escalar numéricos","Dividir Train/Test",
                                           "Regresión (Ventas)","Clasificación (Stockout)","Clustering (Tiendas)",
                                           "Forecast (Temporal)","Explicabilidad","AutoTune"], index=0)
    if add != "—":
        st.session_state.pipeline.append(add)
        st.experimental_rerun()

# ==========================================================
#                PASO 1: ELECCIÓN DE FUENTE
# ==========================================================
if not st.session_state.source_done:

    st.markdown('<div class="block-title">RetailLab Builder — ML Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Primero selecciona una fuente de datos. Tras cargarla, <span class="badge">Gemma3</span> la analizará y preparará vistas.</div>', unsafe_allow_html=True)
    st.divider()

    src = st.radio("Fuente de datos", ["CSV (local)","Base de datos","Excel Online / OneDrive","Google Sheets"], horizontal=True)
    st.session_state.source_type = src

    size = st.selectbox("Tamaño de muestra", ["Pequeño (demo)","Medio","Grande"], index=1)
    size_map = {"Pequeño (demo)": (12, 30, 18), "Medio": (25, 60, 40), "Grande": (40, 90, 70)}

    if src == "CSV (local)":
        st.markdown("#### Carga CSV")
        file = st.file_uploader("Sube tu CSV ilustrativo", type=["csv"])
        if file:
            st.session_state.seed = _hash(file.name)
            with st.status("Gemma3 analizando el archivo…", expanded=True) as status:
                for t in ["Leyendo metadatos","Perfilando columnas","Detectando outliers","Generando vistas"]:
                    st.write("•", t); time.sleep(0.6)
                status.update(label="¡Listo! Data preparada.", state="complete")
            ns, nd, npd = size_map[size]
            st.session_state.df_main = make_demo_data(ns, nd, npd)
            st.session_state.dfs = derive_views(st.session_state.df_main)
            st.session_state.source_done = True
            st.toast("Conexión completada", icon="✅")
            st.experimental_rerun()

    elif src == "Base de datos":
        st.markdown("#### Conectar a BBDD")
        col1, col2 = st.columns(2)
        with col1:
            db_type = st.selectbox("Motor", ["PostgreSQL","MySQL","SQL Server","Oracle","SQLite"])
            host = st.text_input("Host / Endpoint", "db.example.com")
            port = st.text_input("Puerto", {"PostgreSQL":"5432","MySQL":"3306","SQL Server":"1433","Oracle":"1521","SQLite":"—"}[db_type])
            dbname = st.text_input("Base de datos", "retail_demo")
        with col2:
            user = st.text_input("Usuario", "demo_user")
            pwd = st.text_input("Password", type="password")
            table = st.text_input("Tabla principal", "ventas")
        if st.button("Conectar y preparar"):
            with st.status(f"Conectando a {db_type}…", expanded=True) as status:
                for t in ["Autenticando","Leyendo esquema","Muestreando tabla","Preparando vistas"]:
                    st.write("•", t); time.sleep(0.6)
                status.update(label="Conexión establecida", state="complete")
            ns, nd, npd = size_map[size]
            st.session_state.df_main = make_demo_data(ns, nd, npd)
            st.session_state.dfs = derive_views(st.session_state.df_main)
            st.session_state.source_done = True
            st.toast("Datos listos", icon="✅")
            st.experimental_rerun()

    elif src == "Excel Online / OneDrive":
        st.markdown("#### Conectar a Excel Online / OneDrive")
        url = st.text_input("URL compartida del libro", "https://1drv.ms/x/s!demo")
        sheet = st.text_input("Hoja", "ventas")
        if st.button("Vincular y preparar"):
            with st.status("Vinculando Excel…", expanded=True) as status:
                for t in ["Verificando acceso","Leyendo rangos","Normalizando columnas","Creando vistas"]:
                    st.write("•", t); time.sleep(0.6)
                status.update(label="Libro vinculado", state="complete")
            ns, nd, npd = size_map[size]
            st.session_state.df_main = make_demo_data(ns, nd, npd)
            st.session_state.dfs = derive_views(st.session_state.df_main)
            st.session_state.source_done = True
            st.toast("Datos listos", icon="✅")
            st.experimental_rerun()

    else:  # Google Sheets
        st.markdown("#### Conectar a Google Sheets")
        gurl = st.text_input("URL/ID de la hoja", "https://docs.google.com/spreadsheets/d/XXXXXXXX")
        gtab = st.text_input("Nombre de pestaña", "ventas")
        if st.button("Sincronizar y preparar"):
            with st.status("Sincronizando Sheets…", expanded=True) as status:
                for t in ["Validando hoja","Leyendo pestaña","Ajustando tipos","Derivando vistas"]:
                    st.write("•", t); time.sleep(0.6)
                status.update(label="Hoja sincronizada", state="complete")
            ns, nd, npd = size_map[size]
            st.session_state.df_main = make_demo_data(ns, nd, npd)
            st.session_state.dfs = derive_views(st.session_state.df_main)
            st.session_state.source_done = True
            st.toast("Datos listos", icon="✅")
            st.experimental_rerun()

# ==========================================================
#                PASO 2: APP COMPLETA
# ==========================================================
else:
    st.markdown('<div class="block-title">RetailLab Builder — ML Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Datos listos. <span class="badge">Gemma3</span> ya organizó un DF general y vistas derivadas. Configura tu pipeline sin programar.</div>', unsafe_allow_html=True)
    st.divider()

    # KPIs + resumen
    cols = st.columns([3,2,1])
    with cols[0]:
        nice_kpis(st.session_state.df_main)
    with cols[1]:
        gemma3_summary(st.session_state.df_main)
    with cols[2]:
        st.markdown("**Fuente**")
        st.write(st.session_state.source_type)
        st.write("Semilla:", st.session_state.seed)

    # Tabs
    tab_studio, tab_data, tab_models, tab_docs = st.tabs(["🎛️ Builder", "🧾 DataFrames", "🤖 Modelos", "📚 Doc simple"])

    # ---------- BUILDER ----------
    with tab_studio:
        st.markdown("#### Pipeline sin programación")
        if not st.session_state.pipeline:
            st.info("Agrega bloques desde la barra lateral.")
        else:
            # listado con reordenar
            to_remove = None
            for i, step in enumerate(st.session_state.pipeline):
                c1,c2,c3,c4 = st.columns([6,1,1,1])
                with c1: st.markdown(f"- **#{i+1}** {step}")
                with c2:
                    if st.button("⬆️", key=f"up{i}") and i>0:
                        st.session_state.pipeline[i-1], st.session_state.pipeline[i] = st.session_state.pipeline[i], st.session_state.pipeline[i-1]
                        st.experimental_rerun()
                with c3:
                    if st.button("⬇️", key=f"down{i}") and i < len(st.session_state.pipeline)-1:
                        st.session_state.pipeline[i+1], st.session_state.pipeline[i] = st.session_state.pipeline[i], st.session_state.pipeline[i+1]
                        st.experimental_rerun()
                with c4:
                    if st.button("🗑️", key=f"del{i}"): to_remove = i
            if to_remove is not None:
                st.session_state.pipeline.pop(to_remove); st.experimental_rerun()

        st.markdown("#### Visuales rápidos")
        plot_sales(st.session_state.df_main)

    # ---------- DATA ----------
    with tab_data:
        st.markdown("#### DF general (editable)")
        st.data_editor(st.session_state.df_main.head(300), use_container_width=True, height=320, num_rows="dynamic")
        st.markdown("#### Vistas derivadas")
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

    # ---------- MODELOS ----------
    with tab_models:
        left, right = st.columns([2,1])
        with left:
            st.markdown("#### Ejecutar flujo")
            st.write("Pasos:", " → ".join(st.session_state.pipeline) or "Añade bloques en el Builder")
            # Config rápida por tipo
            st.markdown("##### Configuración")
            c1,c2,c3 = st.columns(3)
            with c1:
                target = st.selectbox("Objetivo", ["sales","units","stockout"], index=0)
            with c2:
                algo = st.selectbox("Modelo", ["Regresión lineal","Bosque aleatorio","XGBoost","Clasificador logístico","K-Means","ARIMA"], index=1)
            with c3:
                cv = st.selectbox("Validación", ["Hold-out 80/20","KFold-5","KFold-10"], index=1)
            st.slider("Complejidad", 1, 10, 5, key="complexity")
            st.slider("Horizonte forecast (días)", 7, 60, 14, key="horizon")
            run = st.button("▶️ Ejecutar ahora", type="primary")
            if run:
                with st.spinner("Gemma3 orquestando el flujo…"):
                    prog = st.progress(0)
                    for i in range(1,6):
                        time.sleep(0.35); prog.progress(i/5)
                    metrics, imp = simulate_training(st.session_state.pipeline or ["Baseline"], st.session_state.df_main)
                st.success("Ejecución finalizada")
                c1,c2 = st.columns([1,1])
                with c1:
                    st.markdown("**Métricas**")
                    for k,v in metrics.items(): st.markdown(f"- {k}: **{v}**")
                with c2:
                    st.markdown("**Importancia de features**")
                    plot_feature_importance(imp)
                st.markdown("**Distribuciones y curvas**")
                g = px.histogram(st.session_state.df_main, x="price", color="category", barmode="overlay", opacity=.6, nbins=20)
                g.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(g, use_container_width=True)
        with right:
            st.markdown("#### Ajustes rápidos")
            st.toggle("Balancear clases (si aplica)", value=True)
            st.toggle("One-Hot automático", value=True)
            st.toggle("Escalado MinMax", value=False)
            st.text_input("Semilla", value=str(st.session_state.seed))
            st.text_area("Notas de la corrida", value="Baseline retail ventas/stockout")

    # ---------- DOCS ----------
    with tab_docs:
        st.markdown("### Documentación rápida")
        st.markdown("""
**Regresión (Ventas)**  
Predice una variable numérica (`sales`) con features como `price`, `promo`, `dow`.  
*Ejemplo:* si `price` baja y `promo`=1, se espera mayor venta.

**Clasificación (Stockout)**  
Predice si habrá rotura de stock (`stockout=1`) para alertas de reabasto.  
*Ejemplo:* al bajar `units` y aumentar la demanda, sube la probabilidad.

**Clustering (Tiendas)**  
Agrupa tiendas por comportamiento (ticket, mix, estacionalidad).  
*Ejemplo:* clúster de alto ticket y baja frecuencia → clientes ocasionales premium.

**Forecast (Temporal)**  
Proyecta ventas por día.  
*Ejemplo:* horizonte 14 días usando tendencia + estacionalidad (días de la semana).

**Sugerencias**  
1) Limpia y codifica variables. 2) Divide train/test. 3) Entrena (Regresión/Clasificación).  
4) Explica con importancia de features. 5) Añade Forecast para proyección.
        """)
        st.info("Tip: edita celdas en los DataFrames para simular escenarios y vuelve a ejecutar el flujo.")
