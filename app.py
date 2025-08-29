import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time, hashlib, random, string, json
from datetime import datetime, timedelta

st.set_page_config(page_title="RetailLab Builder — ML Studio", page_icon="🧪", layout="wide")

# ---------- Estilos ----------
st.markdown("""
<style>
:root{
  --bg:#0a0f1c; --panel:#0f1630; --ink:#e8edff; --muted:#9fb2ff; --line:#22305b;
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
.drag-hint{opacity:.8;border:1px dashed #4f5d9c;border-radius:12px;padding:10px}
hr{border:none;border-top:1px solid var(--line);margin:1rem 0}
</style>
""", unsafe_allow_html=True)

# ---------- Intento de componentes drag&drop (opcional) ----------
try:
    from streamlit_elements import elements, mui, dashboard, html
    HAS_ELEMENTS = True
except Exception:
    HAS_ELEMENTS = False

# ---------- Estado ----------
if "seed" not in st.session_state: st.session_state.seed = 123
if "pipeline" not in st.session_state: st.session_state.pipeline = []
if "layout" not in st.session_state: st.session_state.layout = []
if "df_main" not in st.session_state: st.session_state.df_main = None
if "dfs" not in st.session_state: st.session_state.dfs = {}
if "last_upload" not in st.session_state: st.session_state.last_upload = None

random.seed(st.session_state.seed)
np.random.seed(st.session_state.seed)

# ---------- Utilidades ----------
def _hash(s: str) -> int:
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
    return {
        "Ventas por tienda": v_store,
        "Top productos": v_prod,
        "Ventas diarias": v_daily,
        "Alertas stockout": v_alerts
    }

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
    tip = random.choice([
        "Te conviene agrupar por región y día para detectar ‘lunes flojos’.",
        "Considera precio y promo como features clave: suelen explicar >30% de la varianza simulada.",
        "Un cluster por ticket ayuda a detectar tiendas con ‘mix’ atípico.",
        "Un modelo regresión simple ya da baseline decente; evalúa MAE y MAPE."
    ])
    st.chat_message("assistant").markdown(
        f"**Gemma3**: Detecté {len(cats)} categorías dominantes ({', '.join(cats)}). "
        f"Los productos con mayor tracción son: {', '.join(hot)}. {tip}"
    )

def simulate_training(pipeline_steps, df):
    seed = len("".join(pipeline_steps)) + int(df["sales"].sum()) % 1000
    rng = np.random.default_rng(seed)
    time.sleep(0.7)
    metrics = {
        "RMSE ventas": round(rng.uniform(12, 35), 2),
        "MAPE %": round(rng.uniform(8, 18), 2),
        "ROC AUC (rotura stock)": round(rng.uniform(0.72, 0.92), 3),
        "Silhouette (clusters)": round(rng.uniform(0.32, 0.67), 3)
    }
    feats = ["price","promo","units_lag7","dow","region","category","promo_rolling"]
    importances = rng.random(len(feats)); importances /= importances.sum()
    return metrics, pd.DataFrame({"feature":feats, "importance":importances}).sort_values("importance", ascending=False)

def plot_feature_importance(df_imp):
    fig = px.bar(df_imp, x="importance", y="feature", orientation="h")
    fig.update_layout(height=380, margin=dict(l=10,r=10,t=10,b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
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

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 🧪 RetailLab Builder")
    uploaded = st.file_uploader("Sube un CSV de ventas", type=["csv"])
    preset = st.selectbox("Tamaño de muestra", ["Pequeño (demo)","Medio","Grande"], index=1)
    st.markdown("### 🎛️ Modo de bloques")
    palette = ["Limpiar nulos", "One-Hot Encode", "Escalar numéricos", "Dividir Train/Test",
               "Regresión (Ventas)", "Clasificación (Stockout)", "Clustering (Tiendas)",
               "Forecast (Temporal)", "Explicabilidad (SHAP)", "AutoTune (búsqueda)"]
    add_block = st.selectbox("Agregar bloque", ["—"] + palette)
    if add_block != "—":
        st.session_state.pipeline.append(add_block)
        st.rerun()
    if st.button("Limpiar pipeline"):
        st.session_state.pipeline = []

# ---------- Header ----------
st.markdown('<div class="block-title">RetailLab Builder — ML Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Construye visualmente pipelines de aprendizaje automático para retail. Sube un CSV, deja que <span class="badge">Gemma3</span> organice la info y prueba bloques arrastrables.</div>', unsafe_allow_html=True)
st.divider()

# ---------- Data bootstrap ----------
if uploaded and uploaded != st.session_state.last_upload:
    st.session_state.last_upload = uploaded
    st.session_state.seed = _hash(uploaded.name)
    with st.status("Gemma3 analizando el archivo…", expanded=True) as status:
        for t in ["Leyendo metadatos","Perfilando columnas","Detectando outliers","Creando vistas"]:
            st.write("•", t); time.sleep(0.5)
        status.update(label="¡Listo! Data preparada.", state="complete")
    # demo data (no dependemos del CSV)
    size_map = {"Pequeño (demo)": (12, 30, 18), "Medio": (25, 60, 40), "Grande": (40, 90, 70)}
    n_stores, n_days, n_products = size_map[preset]
    st.session_state.df_main = make_demo_data(n_stores, n_days, n_products)
    st.session_state.dfs = derive_views(st.session_state.df_main)

# Si no hay upload, genera demo inicial
if st.session_state.df_main is None:
    st.info("Carga un CSV para comenzar. Puedes usar cualquiera; se organizará automáticamente.")
    st.session_state.df_main = make_demo_data()
    st.session_state.dfs = derive_views(st.session_state.df_main)

# ---------- KPIs + resumen Gemma3 ----------
with st.container():
    cols = st.columns([3,2])
    with cols[0]:
        nice_kpis(st.session_state.df_main)
    with cols[1]:
        gemma3_summary(st.session_state.df_main)

# ---------- Tabs principales ----------
tab_studio, tab_data, tab_models, tab_docs = st.tabs(["🎛️ Studio", "🧾 DataFrames", "🤖 Modelos", "📚 Doc simple"])

# ====== STUDIO ======
with tab_studio:
    st.markdown("#### Canvas de pipeline")
    if HAS_ELEMENTS:
        with elements("studio"):
            # Layout base + tarjetas dinámicas
            base = [
                dashboard.Item("lib", x=0, y=0, w=3, h=8, isDraggable=False, isResizable=False),
                dashboard.Item("out", x=9, y=0, w=3, h=8, isDraggable=False, isResizable=True),
            ]
            cards = []
            for i, step in enumerate(st.session_state.pipeline):
                cards.append(dashboard.Item(f"card_{i}", x=3+(i%6), y=(i//6)*2, w=2, h=2))
            layout = base + cards
            with dashboard.Grid(layout, draggableHandle=".drag-handle"):
                # Librería
                with mui.Card(key="lib"):
                    with mui.CardHeader(title="Bloques disponibles", subheader="Da clic en la barra lateral para añadir"):
                        pass
                    with mui.CardContent():
                        for b in palette:
                            mui.Chip(label=b, color="primary", variant="outlined", style={"margin":"4px"})
                # Output simulado
                with mui.Card(key="out"):
                    mui.CardHeader(title="Salida / Log", subheader="Mensajes de ejecución")
                    with mui.CardContent():
                        log = "\\n".join([f"[✓] {s}" for s in st.session_state.pipeline]) if st.session_state.pipeline else "Añade bloques para comenzar"
                        mui.Typography(log, component="pre", sx={"whiteSpace":"pre-wrap"})
                # Tarjetas de pipeline
                for i, step in enumerate(st.session_state.pipeline):
                    with mui.Card(key=f"card_{i}", sx={"overflow":"hidden"}):
                        with mui.CardHeader(
                            title=step,
                            subheader=f"#{i+1}",
                            avatar=mui.Avatar(str(i+1)),
                            className="drag-handle",
                            action=mui.IconButton(mui.icon.Delete(), onClick=html.capture(f"remove_{i}"))
                        ): pass
                        with mui.CardContent():
                            # Simples controles por bloque
                            if "Regresión" in step:
                                mui.TextField(label="Target", defaultValue="sales", size="small")
                                mui.Slider(defaultValue=80, step=5, min=50, max=95, marks=True)
                            elif "Clasificación" in step:
                                mui.TextField(label="Objetivo", defaultValue="stockout>0", size="small")
                                mui.Select(mui.MenuItem("Logistic"), mui.MenuItem("RandomForest"), defaultValue="Logistic")
                            elif "Clustering" in step:
                                mui.Slider(defaultValue=4, step=1, min=2, max=12, marks=True)
                            elif "Forecast" in step:
                                mui.TextField(label="Horizonte (días)", defaultValue="14", size="small")
                            else:
                                mui.Typography("Opciones rápidas listas.", variant="body2", color="text.secondary")

            # Botones para borrar tarjetas
            for i,_ in enumerate(st.session_state.pipeline):
                if html.event(f"remove_{i}"):
                    st.session_state.pipeline.pop(i)
                    st.rerun()
        st.caption("Tip: arrastra las tarjetas para ordenar visualmente tu flujo.")
    else:
        st.warning("Interfaz drag-and-drop avanzada requiere `streamlit-elements`. Usa la barra lateral para construir el pipeline en modo simple.")
        st.write("Bloques actuales:", " → ".join(st.session_state.pipeline) if st.session_state.pipeline else "—")

    st.markdown("#### Visuales rápidos")
    plot_sales(st.session_state.df_main)

# ====== DATAFRAMES ======
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

# ====== MODELOS ======
with tab_models:
    cols = st.columns([2,1])
    with cols[0]:
        st.markdown("#### Ejecutar flujo")
        st.write("Pasos:", " → ".join(st.session_state.pipeline) or "Añade bloques en el Studio")
        run = st.button("▶️ Ejecutar ahora", type="primary")
        if run and st.session_state.pipeline:
            with st.spinner("Gemma3 orquestando el flujo, por favor espera…"):
                prog = st.progress(0)
                for i in range(1, 6):
                    time.sleep(0.35); prog.progress(i/5)
                metrics, imp = simulate_training(st.session_state.pipeline, st.session_state.df_main)
            st.success("Ejecución finalizada")
            c1,c2 = st.columns([1,1])
            with c1:
                st.markdown("**Métricas**")
                for k,v in metrics.items():
                    st.markdown(f"- {k}: **{v}**")
            with c2:
                st.markdown("**Importancia de features**")
                plot_feature_importance(imp)
            st.markdown("**Curvas y distribuciones**")
            g = px.histogram(st.session_state.df_main, x="price", color="category", barmode="overlay", opacity=.6, nbins=20)
            g.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(g, use_container_width=True)
        elif run:
            st.info("Primero agrega bloques a tu pipeline en el Studio.")
    with cols[1]:
        st.markdown("#### Config rápida")
        st.toggle("Validación cruzada", value=True)
        st.select_slider("Complejidad", options=["Baja","Media","Alta"], value="Media")
        st.slider("Porc. entrenamiento", 50, 95, 80)
        st.text_input("Semilla", value=str(st.session_state.seed))
        st.text_area("Notas de la corrida", value="Baseline retail ventas/stockout")

# ====== DOCS ======
with tab_docs:
    st.markdown("### Documentación rápida")
    st.markdown("""
**Regresión (Ventas)**  
Predice una variable numérica (ej. `sales`) a partir de otras (`price`, `promo`, `dow`, etc.).  
*Ejemplo corto:* si `price` baja y `promo`=1, el modelo espera más ventas.

**Clasificación (Stockout)**  
Clasifica si habrá rotura de stock (`stockout=1`). Útil para alertas.  
*Ejemplo:* si `units` van a la baja y la reposición tarda, aumenta probabilidad de 1.

**Clustering (Tiendas)**  
Agrupa tiendas con comportamiento similar (ticket, mix, estacionalidad).  
*Ejemplo:* clúster con alto ticket y baja frecuencia → clientes “ocasionales premium”.

**Forecast (Temporal)**  
Proyecta ventas por día (serie de tiempo).  
*Ejemplo:* próximas 2 semanas usando tendencia + estacionalidad (días de la semana).

**Explicabilidad**  
Mide el peso de cada feature en la predicción. La gráfica de importancia orienta prioridades.

**Sugerencias de uso**  
1) Comienza con: Limpiar → One-Hot → Escalar → Dividir.  
2) Luego agrega: Regresión o Clasificación según objetivo.  
3) Cierra con: Explicabilidad y un Forecast a 14 días.
    """)
    st.info("Consejo: edita celdas en los DataFrames para experimentar con escenarios y vuelve a ‘Ejecutar flujo’.")
