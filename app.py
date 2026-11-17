"""
MVP · Registro de clientes para la campaña de depósitos a plazo (Bank Marketing, UCI).

Un usuario registra un cliente con las mismas 20 variables del dataset del proyecto; el registro
se guarda en la base de datos (SQLite) junto a los 41 188 originales y el Random Forest final
le asigna una probabilidad de contratar y una recomendación (llamar / no llamar, umbral 0.296).

Ejecutar en local:   streamlit run app.py
"""
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from train_model import preparar

BASE = Path(__file__).parent
CSV_ORIGINAL = BASE / "bank-additional-full.csv"
MODELO = BASE / "modelo" / "rf_bank.joblib"
DB = Path(os.environ.get("MVP_DB", BASE / "datos_mvp" / "clientes.db"))

# Paleta del proyecto (misma del notebook)
MEDIANOCHE, GRIS_AZUL, LADRILLO = "#1B2A49", "#B7C4D6", "#A93F4A"

COLUMNAS = ["age", "job", "marital", "education", "default", "housing", "loan", "contact",
            "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome",
            "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y"]

# Valores permitidos = los que existen en el dataset (el modelo no conoce otros)
OPCIONES = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
            "self-employed", "services", "student", "technician", "unemployed", "unknown"],
    "marital": ["married", "single", "divorced", "unknown"],
    "education": ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
                  "professional.course", "university.degree", "unknown"],
    "default": ["no", "yes", "unknown"],
    "housing": ["no", "yes", "unknown"],
    "loan": ["no", "yes", "unknown"],
    "contact": ["cellular", "telephone"],
    "month": ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
    "poutcome": ["nonexistent", "failure", "success"],
}
ETIQUETAS = {
    "job": {"admin.": "Administrativo", "blue-collar": "Obrero", "entrepreneur": "Emprendedor",
            "housemaid": "Trabajo doméstico", "management": "Gerencia", "retired": "Jubilado",
            "self-employed": "Independiente", "services": "Servicios", "student": "Estudiante",
            "technician": "Técnico", "unemployed": "Desempleado", "unknown": "Desconocido"},
    "marital": {"married": "Casado/a", "single": "Soltero/a", "divorced": "Divorciado/a o viudo/a",
                "unknown": "Desconocido"},
    "education": {"basic.4y": "Básica 4 años", "basic.6y": "Básica 6 años", "basic.9y": "Básica 9 años",
                  "high.school": "Secundaria", "illiterate": "Sin estudios",
                  "professional.course": "Técnica profesional", "university.degree": "Universitaria",
                  "unknown": "Desconocido"},
    "si_no": {"no": "No", "yes": "Sí", "unknown": "Desconocido"},
    "contact": {"cellular": "Celular", "telephone": "Teléfono fijo"},
    "month": {"mar": "Marzo", "apr": "Abril", "may": "Mayo", "jun": "Junio", "jul": "Julio",
              "aug": "Agosto", "sep": "Septiembre", "oct": "Octubre", "nov": "Noviembre", "dec": "Diciembre"},
    "day_of_week": {"mon": "Lunes", "tue": "Martes", "wed": "Miércoles", "thu": "Jueves", "fri": "Viernes"},
    "poutcome": {"failure": "Fracaso (no contrató)", "success": "Éxito (contrató)"},
}
# Contexto macro por defecto = último registro del dataset (noviembre 2010)
MACRO_DEFECTO = {"emp.var.rate": -1.1, "cons.price.idx": 94.767, "cons.conf.idx": -50.8,
                 "euribor3m": 1.028, "nr.employed": 4963.6}


# ------------------------------------------------------------------ base de datos
def conexion():
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    existe = con.execute("SELECT name FROM sqlite_master WHERE name='clientes'").fetchone()
    if not existe:   # primera ejecución: la base arranca con el dataset original del proyecto
        df = pd.read_csv(CSV_ORIGINAL, sep=";")
        df["origen"], df["registrado"] = "dataset original", None
        df["probabilidad"], df["recomendacion"] = float("nan"), None
        df.to_sql("clientes", con, index_label="id")
    return con


def guardar(registros: pd.DataFrame) -> int:
    with conexion() as con:
        siguiente = con.execute("SELECT MAX(id) + 1 FROM clientes").fetchone()[0]
        registros = registros.copy()
        registros.index = range(siguiente, siguiente + len(registros))
        registros.to_sql("clientes", con, if_exists="append", index_label="id")
    return len(registros)


def leer(solo_nuevos: bool = False) -> pd.DataFrame:
    filtro = "WHERE origen != 'dataset original'" if solo_nuevos else ""
    with conexion() as con:
        return pd.read_sql(f"SELECT * FROM clientes {filtro} ORDER BY id DESC", con, index_col="id")


# ------------------------------------------------------------------ modelo
@st.cache_resource
def cargar_modelo():
    return joblib.load(MODELO)


def puntuar(registros: pd.DataFrame) -> pd.DataFrame:
    m = cargar_modelo()
    X = preparar(registros[COLUMNAS[:-1]], columnas=m["columnas"])
    salida = registros.copy()
    salida["probabilidad"] = m["modelo"].predict_proba(X)[:, 1].round(4)
    salida["recomendacion"] = ["llamar" if p >= m["umbral"] else "no llamar" for p in salida["probabilidad"]]
    return salida


def validar(df: pd.DataFrame) -> list[str]:
    """Reglas de consistencia del dataset; devuelve los errores encontrados (vacío = válido)."""
    errores = []
    faltan = [c for c in COLUMNAS[:-1] if c not in df.columns]
    if faltan:
        return [f"Faltan columnas: {', '.join(faltan)}"]
    for col, permitidos in OPCIONES.items():
        malos = sorted(set(df[col].astype(str)) - set(permitidos))
        if malos:
            errores.append(f"`{col}` tiene valores no válidos: {', '.join(malos)}")
    for col, (lo, hi) in {"age": (17, 100), "campaign": (1, 60), "previous": (0, 10),
                          "pdays": (0, 999), "duration": (0, 6000)}.items():
        v = pd.to_numeric(df[col], errors="coerce")
        if v.isna().any() or (v < lo).any() or (v > hi).any():
            errores.append(f"`{col}` debe ser numérico entre {lo} y {hi}")
    if not errores:
        sin_historia = df["previous"].astype(int) == 0
        if ((df["poutcome"] == "nonexistent") != sin_historia).any():
            errores.append("`poutcome` = nonexistent si y solo si `previous` = 0")
        if ((df["poutcome"] == "success") & (df["pdays"].astype(int) == 999)).any():
            errores.append("Un éxito anterior necesita los días desde el último contacto (`pdays` ≠ 999)")
        if (sin_historia & (df["pdays"].astype(int) != 999)).any():
            errores.append("Sin contactos previos, `pdays` debe ser 999")
    return errores


# ------------------------------------------------------------------ interfaz
st.set_page_config(page_title="MVP · Registro de clientes", page_icon="🏦", layout="wide")
st.title("Registro de clientes · Campaña de depósitos a plazo")
st.caption("MVP del proyecto de Applied Machine Learning (UTEC 2026-2) · Dataset Bank Marketing (UCI) · "
           "Modelo: Random Forest (AUC-PR 0.494)")

tab_reg, tab_base, tab_masiva, tab_info = st.tabs(
    ["➕ Registrar cliente", "🗂️ Base de datos", "📤 Carga masiva (CSV)", "ℹ️ Sobre el modelo"])


def selector(etiqueta, col, clave_etq=None, **kw):
    nombres = ETIQUETAS.get(clave_etq or col, {})
    return st.selectbox(etiqueta, OPCIONES[col], format_func=lambda v: nombres.get(v, v), **kw)


with tab_reg:
    st.subheader("Datos del cliente")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Perfil**")
            age = st.number_input("Edad", 17, 100, 40)
            job = selector("Ocupación", "job")
            marital = selector("Estado civil", "marital")
            education = selector("Educación", "education", index=6)
        with c2:
            st.markdown("**Situación financiera**")
            default = selector("¿Crédito en mora?", "default", "si_no")
            housing = selector("¿Préstamo hipotecario?", "housing", "si_no")
            loan = selector("¿Préstamo personal?", "loan", "si_no")
            contact = selector("Medio de contacto", "contact", index=0)
        with c3:
            st.markdown("**Campaña actual**")
            month = selector("Mes de la llamada", "month", index=8)
            day = selector("Día de la llamada", "day_of_week")
            campaign = st.number_input("N.º de llamadas en esta campaña (incluida esta)", 1, 60, 1)
            duration = st.number_input("Duración de la última llamada (s) · solo registro, el modelo no la usa",
                                       0, 6000, 0, help="Se conoce después de llamar: usarla sería fuga de información.")

        st.markdown("**Historial con campañas anteriores**")
        h1, h2, h3 = st.columns(3)
        with h1:
            previous = st.number_input("Contactos en campañas anteriores (0 = nunca)", 0, 10, 0)
        with h2:
            poutcome = st.selectbox("Resultado de la campaña anterior",
                                    ["failure", "success"] if previous else ["nonexistent"],
                                    format_func=lambda v: ETIQUETAS["poutcome"].get(v, "Sin campaña anterior"),
                                    disabled=not previous)
        with h3:
            conoce_dias = st.checkbox("Se conoce la fecha del último contacto", value=bool(previous),
                                      disabled=not previous)
            pdays = st.number_input("Días desde el último contacto", 0, 998, 10,
                                    disabled=not (previous and conoce_dias))

        with st.expander("Contexto macroeconómico del mes (por defecto: último dato disponible, nov 2010)"):
            m1, m2, m3, m4, m5 = st.columns(5)
            macro = {
                "emp.var.rate": m1.number_input("Var. tasa de empleo", -5.0, 5.0, MACRO_DEFECTO["emp.var.rate"], .1),
                "cons.price.idx": m2.number_input("Índice de precios", 90.0, 96.0, MACRO_DEFECTO["cons.price.idx"], .001, format="%.3f"),
                "cons.conf.idx": m3.number_input("Confianza del consumidor", -60.0, -20.0, MACRO_DEFECTO["cons.conf.idx"], .1),
                "euribor3m": m4.number_input("Euribor 3 meses (%)", 0.0, 6.0, MACRO_DEFECTO["euribor3m"], .001, format="%.3f"),
                "nr.employed": m5.number_input("N.º de empleados (miles)", 4900.0, 5300.0, MACRO_DEFECTO["nr.employed"], .1),
            }
        y_real = st.radio("¿Ya se conoce si contrató el depósito?", ["Aún no", "Sí", "No"], horizontal=True)
        enviado = st.button("Calcular y guardar", type="primary")

    if enviado:
        registro = pd.DataFrame([{
            "age": age, "job": job, "marital": marital, "education": education, "default": default,
            "housing": housing, "loan": loan, "contact": contact, "month": month, "day_of_week": day,
            "duration": duration, "campaign": campaign,
            "pdays": pdays if previous and conoce_dias else 999,
            "previous": previous, "poutcome": poutcome if previous else "nonexistent", **macro,
            "y": {"Sí": "yes", "No": "no"}.get(y_real),
        }])
        errores = validar(registro)
        if errores:
            for e in errores:
                st.error(e)
        else:
            registro = puntuar(registro)
            registro["origen"] = "formulario"
            registro["registrado"] = datetime.now().isoformat(timespec="seconds")
            guardar(registro)
            p, rec = registro.loc[0, "probabilidad"], registro.loc[0, "recomendacion"]
            umbral, base = cargar_modelo()["umbral"], cargar_modelo()["tasa_base"]
            st.success("Cliente guardado en la base de datos.")
            r1, r2, r3 = st.columns(3)
            r1.metric("Puntaje del modelo", f"{p:.1%}")
            r2.metric("Recomendación", rec.upper())
            r3.metric("Umbral de decisión", f"{umbral:.3f}")
            st.caption(f"El puntaje ordena clientes, no es una probabilidad calibrada: los pesos de clase lo inflan. "
                       f"Tasa real de contratación en el dataset: {base:.1%}. "
                       f"El umbral {umbral} maximiza el beneficio ilustrativo (5 € por llamada, 80 € por depósito).")

with tab_base:
    solo_nuevos = st.toggle("Mostrar solo los registros añadidos en el MVP", value=True)
    datos = leer(solo_nuevos)
    total = leer(False)
    k1, k2, k3 = st.columns(3)
    k1.metric("Registros en la base", f"{len(total):,}")
    k2.metric("Añadidos desde el MVP", f"{(total['origen'] != 'dataset original').sum():,}")
    nuevos = total[total["origen"] != "dataset original"]
    k3.metric("Recomendados para llamar", f"{(nuevos['recomendacion'] == 'llamar').sum():,}")
    if datos.empty:
        st.info("Aún no hay registros añadidos. Use la pestaña «Registrar cliente».")
    else:
        st.dataframe(datos, width="stretch", height=420)
        st.download_button("Descargar en formato del dataset (CSV con ';')",
                           datos[COLUMNAS].to_csv(sep=";", index=False).encode("utf-8"),
                           "clientes_mvp.csv", "text/csv")

with tab_masiva:
    st.write("Suba un CSV con las mismas columnas que `bank-additional-full.csv` (separador `;`). "
             "La columna `y` es opcional. Cada fila se valida, se puntúa y se guarda.")
    plantilla = pd.read_csv(CSV_ORIGINAL, sep=";", nrows=3)
    st.download_button("Descargar plantilla (3 filas de ejemplo)",
                       plantilla.to_csv(sep=";", index=False).encode("utf-8"), "plantilla_clientes.csv")
    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        lote = pd.read_csv(archivo, sep=None, engine="python")
        errores = validar(lote)
        if errores:
            for e in errores:
                st.error(e)
        else:
            if "y" not in lote:
                lote["y"] = None
            lote = puntuar(lote[COLUMNAS])
            st.dataframe(lote, width="stretch")
            if st.button(f"Guardar {len(lote)} registros", type="primary"):
                lote["origen"] = "carga CSV"
                lote["registrado"] = datetime.now().isoformat(timespec="seconds")
                st.success(f"{guardar(lote)} registros guardados.")

with tab_info:
    st.markdown(f"""
**Qué hace este MVP.** Permite alimentar la base de datos del proyecto con clientes nuevos, usando exactamente
las variables del dataset *Bank Marketing* (UCI), y devuelve la recomendación del modelo final.

**Modelo.** Random Forest (300 árboles, `max_features=0.3`, `min_samples_leaf=20`, pesos de clase
`balanced_subsample`), entrenado con la misma preparación y partición del informe: sin `duration`
(fuga de información), sin `emp.var.rate` ni `nr.employed` (VIF 33 y 31), escalado dentro del Pipeline.
AUC-PR en prueba 0.494 (tasa base 0.113).

**Regla de decisión.** Llamar si el puntaje ≥ 0.296: el umbral que maximiza el beneficio ilustrativo del §5.3
(llamar al 38 % de la base alcanza al 78 % de los contratantes).

**Limitaciones.** Sin historial de contacto el modelo apenas discrimina (§5.2); el contexto macro marca
*cuándo* se llama más que *a quién*. Los meses enero y febrero no existen en el dataset.
""")
