"""Ejecutar con: python -m streamlit run app.py"""
from functools import partial
from pathlib import Path
import inspect
import json
import os

import streamlit as st

from analisis import ETIQUETAS, ENCODER_REVISION, analizar, cargar_corpus, texto_visible
from vistas import grafico_mapa, grafico_vecinos


BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="IMDb · Explorador de reseñas", page_icon="🎬", layout="wide")


def ancho_completo(funcion):
    """Streamlit antiguo usa use_container_width; las versiones nuevas, width."""
    if "width" in inspect.signature(funcion).parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}


if "on_select" not in inspect.signature(st.plotly_chart).parameters:
    st.error("Esta versión de Streamlit no permite seleccionar puntos en el mapa.")
    st.info("Cierra este servidor y abre iniciar_windows.bat para instalar y usar las dependencias del proyecto.")
    st.stop()

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2.5rem; max-width: 1500px;}
h1 {letter-spacing: -0.035em;}
[data-testid="stMetricValue"] {font-size: 1.6rem;}
[data-testid="stCaptionContainer"] {line-height: 1.5;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def cargar_modelo(ruta, modificacion):
    return cargar_corpus(ruta)


@st.cache_resource(show_spinner=False)
def cargar_encoder(nombre, revision):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(nombre, revision=revision, device="cpu")


defaults = {
    "resultado": None, "seleccionado": None, "filtro": "Todos", "consulta_num": 0,
    "mapa_version": 0, "vecinos_version": 0, "mapa_selector": None,
    "vecino_selector": None, "texto_entrada": "", "aviso": None,
}
for clave, valor in defaults.items():
    st.session_state.setdefault(clave, valor)

st.title("Explorador de reseñas IMDb")
st.write("Analiza una opinión en inglés, explora su contexto y compárala con otras reseñas.")

ruta = Path(os.environ.get("IMDB_MODEL_PATH", str(BASE / "modelo_sentimientos_imdb.pkl")))
try:
    corpus = cargar_modelo(str(ruta.resolve()), ruta.stat().st_mtime_ns)
except Exception as exc:
    st.error("No se pudo cargar el modelo del proyecto.")
    st.info("Conserva modelo_sentimientos_imdb.pkl junto a app.py e instala requirements.txt en un entorno virtual nuevo.")
    with st.expander("Detalle para resolver el problema"):
        st.code(str(exc), language=None)
    st.stop()


def marcar(indice, origen):
    """Un único registro seleccionado coordina mapa, vecinos y detalle."""
    ss = st.session_state
    ss.seleccionado = None if indice is None else int(indice)
    if indice is not None and origen == "vecinos":
        sentimiento = ETIQUETAS[corpus.tabla.iloc[int(indice)]["sentimiento"]]
        if ss.filtro not in ["Todos", sentimiento]:
            ss.filtro = "Todos"
            ss.aviso = "El mapa volvió a Todos para mostrar la reseña seleccionada."
    visibles = corpus.tabla if ss.filtro == "Todos" else corpus.tabla.loc[
        corpus.tabla["sentimiento"].map(ETIQUETAS).eq(ss.filtro)
    ]
    ss.mapa_selector = int(indice) if indice is not None and indice in visibles.index else None
    vecinos_actuales = [] if ss.resultado is None else ss.resultado["indices"]
    ss.vecino_selector = int(indice) if indice is not None and indice in vecinos_actuales else None
    if origen != "mapa_grafico":
        ss.mapa_version += 1
    # El dibujo manual conserva el borde, sin depender del estado nativo de Plotly.
    ss.vecinos_version += 1


def seleccionar_grafico(clave, origen):
    evento = st.session_state.get(clave, {})
    puntos = evento.get("selection", {}).get("points", [])
    if puntos:
        dato = puntos[-1].get("customdata", [])
        if dato and 0 <= int(dato[0]) < len(corpus.tabla):
            marcar(int(dato[0]), origen)


def seleccionar_lista(clave, origen):
    indice = st.session_state.get(clave)
    marcar(indice, origen)


def cambiar_filtro():
    ss = st.session_state
    seleccionado = ss.seleccionado
    visible = seleccionado is not None and (
        ss.filtro == "Todos" or ETIQUETAS[corpus.tabla.iloc[seleccionado]["sentimiento"]] == ss.filtro
    )
    ss.mapa_selector = seleccionado if visible else None
    ss.mapa_version += 1


with st.sidebar:
    st.subheader("Tres tareas")
    st.markdown("**1. Analizar:** escribe una reseña y pulsa Analizar.\n\n"
                "**2. Explorar:** filtra el mapa y selecciona un punto.\n\n"
                "**3. Comparar:** elige un vecino y lee ambos textos.")
    st.divider()
    st.metric("Reseñas de referencia", f"{len(corpus.tabla):,}")
    st.caption("Azul: positivo · Naranja: negativo. Los colores del corpus representan sus etiquetas originales.")
    with st.expander("Acerca del modelo"):
        st.write("MiniLM · 384 dimensiones · LinearSVC")
        st.write("PCA representa el espacio en dos dimensiones. La clasificación y la similitud usan el vector completo.")
        if corpus.info.get("ids_locales"):
            st.caption("Los códigos REF son identificadores locales de las referencias incluidas.")
        if "accuracy" in corpus.artefacto:
            st.caption(f"Exactitud guardada en este artefacto: {corpus.artefacto['accuracy']:.4f}. "
                       "No se vuelve a evaluar durante la consulta.")
        st.caption("El modelo incluido procede del proyecto anterior. Consulta LEEME.md para reproducir el entrenamiento.")

# V1. Entrada y resultado de la última consulta válida.
with st.container(border=True):
    st.subheader("1 · Analiza una reseña")
    with st.form("entrada", clear_on_submit=False):
        texto = st.text_area(
            "Reseña en inglés", key="texto_entrada", height=115, max_chars=30000,
            placeholder="The story was engaging and the performances were excellent.",
        )
        ejecutar = st.form_submit_button("Analizar reseña", type="primary",
                                         **ancho_completo(st.form_submit_button))
    if ejecutar:
        if not texto.strip():
            st.warning("Escribe una reseña antes de pulsar Analizar.")
        else:
            try:
                with st.spinner("Analizando… La primera consulta descarga MiniLM si todavía no está disponible."):
                    revision = corpus.info.get("encoder_revision", ENCODER_REVISION)
                    encoder = cargar_encoder(corpus.artefacto["embedding_model_name"], revision)
                    nuevo = analizar(texto, corpus, encoder)
                st.session_state.resultado = nuevo
                st.session_state.consulta_num += 1
                st.session_state.filtro = "Todos"
                st.session_state.aviso = None
                marcar(None, "nueva_consulta")
            except Exception as exc:
                st.error("No se pudo analizar la reseña. Comprueba la conexión de la primera descarga y las dependencias.")
                with st.expander("Detalle del error"):
                    st.code(str(exc), language=None)
    resultado = st.session_state.resultado
    if resultado is not None:
        a, b, c = st.columns(3)
        a.metric("Sentimiento predicho", ETIQUETAS[resultado["prediccion"]])
        b.metric("Margen SVM", f"{resultado['margen']:+.4f}")
        c.metric("Tokens de la reseña", resultado["tokens"])
        st.caption("El margen expresa la posición respecto a la frontera del clasificador; no es una probabilidad. "
                   f"El signo positivo corresponde a {ETIQUETAS[resultado['clase_margen_positivo']].lower()}.")
        if resultado["tokens"] > resultado["limite_tokens"]:
            st.warning(f"La reseña tiene {resultado['tokens']} tokens. El encoder utiliza solo los primeros "
                       f"{resultado['limite_tokens']} tokens, incluidos los especiales. Abajo puedes leer el texto completo.")
        if texto != resultado["texto"]:
            st.info("Las vistas muestran la última reseña analizada. Pulsa Analizar para actualizar los resultados.")
    else:
        st.caption("También puedes explorar el mapa antes de ingresar una reseña.")

izquierda, derecha = st.columns([1.45, 1], gap="large")

# V2. El filtro solo cambia los puntos del corpus visibles en el mapa.
with izquierda:
    st.subheader("2 · Explora el mapa")
    st.selectbox("Sentimiento original", ["Todos", "Positivo", "Negativo"],
                 key="filtro", on_change=cambiar_filtro)
    if st.session_state.aviso:
        st.caption(st.session_state.aviso)
        st.session_state.aviso = None
    clave_mapa = f"mapa_{st.session_state.consulta_num}_{st.session_state.mapa_version}"
    fig = grafico_mapa(corpus, st.session_state.filtro, resultado,
                       st.session_state.seleccionado, st.session_state.consulta_num)
    st.plotly_chart(fig, **ancho_completo(st.plotly_chart), key=clave_mapa,
                    on_select=partial(seleccionar_grafico, clave_mapa, "mapa_grafico"),
                    selection_mode="points", config={"scrollZoom": False, "displaylogo": False})
    visibles = corpus.tabla if st.session_state.filtro == "Todos" else corpus.tabla.loc[
        corpus.tabla["sentimiento"].map(ETIQUETAS).eq(st.session_state.filtro)
    ]
    st.selectbox(
        "Abrir una reseña por su código", options=visibles.index.tolist(), index=None,
        key="mapa_selector", placeholder="Selecciona un punto o busca un código…",
        format_func=lambda i: f"{corpus.tabla.iloc[i]['id']} · {ETIQUETAS[corpus.tabla.iloc[i]['sentimiento']]}",
        on_change=partial(seleccionar_lista, "mapa_selector", "mapa_lista"),
    )
    varianza = float(corpus.artefacto["pca"].explained_variance_ratio_.sum())
    st.caption(f"{len(visibles):,} referencias visibles. Estos dos ejes conservan {varianza:.2%} de la varianza. "
               "Usa zoom, desplaza el mapa o pulsa un punto. La cercanía en 2D no equivale exactamente a la similitud semántica.")

# V3. El ranking no cambia con el filtro del mapa.
with derecha:
    st.subheader("3 · Compara con sus vecinos")
    if resultado is None:
        st.info("Analiza una reseña para ver las ocho referencias más similares.")
    else:
        st.caption(f"Las ocho más similares entre {len(corpus.tabla):,} referencias, por coseno en "
                   f"{corpus.artefacto['dimension_embedding']} dimensiones. Se excluyen copias del texto consultado.")
        clave_vecinos = f"vecinos_{st.session_state.consulta_num}_{st.session_state.vecinos_version}"
        st.plotly_chart(
            grafico_vecinos(corpus, resultado, st.session_state.seleccionado),
            **ancho_completo(st.plotly_chart),
            key=clave_vecinos, on_select=partial(seleccionar_grafico, clave_vecinos, "vecinos"),
            selection_mode="points", config={"displayModeBar": False},
        )
        indices = [int(i) for i in resultado["indices"]]
        st.selectbox(
            "Selecciona una reseña similar", indices, index=None, key="vecino_selector",
            placeholder="Elige un vecino o pulsa su barra…",
            format_func=lambda i: (f"{indices.index(i) + 1}. {corpus.tabla.iloc[i]['id']} · "
                                   f"{ETIQUETAS[corpus.tabla.iloc[i]['sentimiento']]} · {resultado['similitudes'][i]:.4f}"),
            on_change=partial(seleccionar_lista, "vecino_selector", "vecinos"),
        )
        st.caption("Este ranking se mantiene al filtrar el mapa. Una reseña similar puede expresar un sentimiento diferente.")

# V4. El detalle siempre utiliza el texto de la última consulta analizada.
st.divider()
st.subheader("4 · Lee y contrasta los textos")
if st.session_state.seleccionado is not None:
    st.button("Quitar selección", on_click=partial(marcar, None, "limpiar"))
consulta, referencia = st.columns(2, gap="large")
with consulta:
    st.markdown("**Tu reseña**")
    if resultado is None:
        st.caption("Aquí aparecerá el texto que analices.")
    else:
        st.caption(f"Predicción: {ETIQUETAS[resultado['prediccion']]} · Margen: {resultado['margen']:+.4f}")
        with st.container(height=270, border=True):
            st.text(texto_visible(resultado["texto"]))
with referencia:
    st.markdown("**Reseña seleccionada**")
    indice = st.session_state.seleccionado
    if indice is None:
        st.caption("Selecciona un punto del mapa o una de las ocho reseñas similares.")
    else:
        fila = corpus.tabla.iloc[indice]
        detalle = f"{fila['id']} · Etiqueta original: {ETIQUETAS[fila['sentimiento']]} · {fila['palabras']} palabras"
        if resultado is not None:
            detalle += f" · Coseno: {resultado['similitudes'][indice]:.4f}"
        st.caption(detalle)
        with st.container(height=270, border=True):
            st.text(fila["texto"])
        if st.session_state.filtro not in ["Todos", ETIQUETAS[fila["sentimiento"]]]:
            st.caption("Esta reseña está fuera del filtro actual del mapa; su detalle sigue abierto.")

if resultado is not None:
    exportacion = {
        "consulta": resultado["texto"], "prediccion": resultado["prediccion"],
        "margen_svm": resultado["margen"], "tokens": resultado["tokens"],
        "limite_tokens": resultado["limite_tokens"], "referencias_disponibles": len(corpus.tabla),
        "modelo": corpus.info,
        "vecinos": [{"id": corpus.tabla.iloc[int(i)]["id"],
                     "sentimiento_original": corpus.tabla.iloc[int(i)]["sentimiento"],
                     "similitud_coseno": float(resultado["similitudes"][i])}
                    for i in resultado["indices"]],
    }
    st.download_button("Descargar este análisis", json.dumps(exportacion, ensure_ascii=False, indent=2),
                       file_name="analisis_resena.json", mime="application/json")
st.caption("Visual Analytics · Proyecto IMDb · Los vecinos aportan contexto; no explican de forma causal la decisión del clasificador.")
