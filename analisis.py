"""Carga del modelo y operaciones analíticas, independientes de Streamlit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import html
import json
import re

import joblib
import numpy as np
import pandas as pd


ETIQUETAS = {"positive": "Positivo", "negative": "Negativo"}
COLORES = {"positive": "#2457A6", "negative": "#D96B18"}
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def texto_visible(texto: str) -> str:
    """Limpieza SOLO para lectura, conteo y detección de duplicados."""
    texto = html.unescape(str(texto))
    return re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", texto)).strip()


def clave_texto(texto: str) -> str:
    return texto_visible(texto).casefold()


def contar_palabras(texto: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)*", texto_visible(texto)))


def preparar_datos(datos: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Deduplica antes del split; conserva review original para el encoder."""
    if not {"review", "sentiment"}.issubset(datos.columns):
        raise ValueError("El CSV debe tener las columnas review y sentiment.")
    df = datos[["review", "sentiment"]].copy()
    df["id"] = [f"R{i:07d}" for i in range(len(df))]
    resumen = {"filas_originales": len(df)}
    df = df.dropna(subset=["review", "sentiment"])
    df["review"] = df["review"].astype(str)
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df["clave"] = df["review"].map(clave_texto)
    df = df.loc[df["clave"].ne("")].copy()
    if not set(df["sentiment"]).issubset(ETIQUETAS):
        raise ValueError("sentiment solo admite positive y negative.")
    conflictos = df.groupby("clave")["sentiment"].nunique().gt(1).sum()
    if conflictos:
        raise ValueError(f"Hay {conflictos} textos con etiquetas contradictorias. Revisa el CSV.")
    resumen["filas_nulas_o_vacias"] = len(datos) - len(df)
    resumen["duplicados_retirados"] = int(df.duplicated("clave").sum())
    df = df.drop_duplicates("clave", keep="first").reset_index(drop=True)
    if df["sentiment"].nunique() != 2 or df["sentiment"].value_counts().min() < 4:
        raise ValueError("Se necesitan al menos cuatro reseñas distintas de cada sentimiento.")
    resumen["filas_preparadas"] = len(df)
    return df, resumen


@dataclass
class Corpus:
    artefacto: dict
    tabla: pd.DataFrame
    embeddings: np.ndarray
    info: dict


def cargar_corpus(ruta: str | Path) -> Corpus:
    ruta = Path(ruta)
    # El archivo incluido procede del repositorio del proyecto.
    modelo = joblib.load(ruta)
    requeridas = {
        "clasificador", "pca", "embedding_model_name", "dimension_embedding",
        "embeddings_referencia", "proyeccion_referencia", "textos_referencia",
        "sentimientos_referencia",
    }
    if not isinstance(modelo, dict) or not requeridas.issubset(modelo):
        raise ValueError("El archivo no tiene la estructura del modelo de este proyecto.")
    emb = np.asarray(modelo["embeddings_referencia"], dtype=np.float32)
    xy = np.asarray(modelo["proyeccion_referencia"], dtype=np.float32)
    textos = modelo["textos_referencia"]
    etiquetas = np.asarray(modelo["sentimientos_referencia"], dtype=str)
    n = len(textos)
    dim = int(modelo["dimension_embedding"])
    if n < 8 or emb.shape != (n, dim) or xy.shape != (n, 2) or etiquetas.shape != (n,):
        raise ValueError("Las dimensiones del corpus de referencia no coinciden.")
    if not np.isfinite(emb).all() or not np.isfinite(xy).all():
        raise ValueError("El modelo contiene valores no finitos.")
    if not np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4):
        raise ValueError("Los embeddings de referencia deben estar normalizados L2.")
    if set(modelo["clasificador"].classes_) != set(ETIQUETAS):
        raise ValueError("El clasificador debe tener las clases positive y negative.")
    if not set(etiquetas).issubset(ETIQUETAS):
        raise ValueError("El corpus tiene etiquetas desconocidas.")
    ids = modelo.get("ids_referencia", [f"REF{i + 1:04d}" for i in range(n)])
    if len(ids) != n or len(set(ids)) != n:
        raise ValueError("Los identificadores del corpus deben ser únicos.")
    tabla = pd.DataFrame({
        "id": ids, "original": textos, "sentimiento": etiquetas,
        "pc1": xy[:, 0], "pc2": xy[:, 1],
    })
    tabla["texto"] = tabla["original"].map(texto_visible)
    tabla["clave"] = tabla["original"].map(clave_texto)
    tabla["palabras"] = tabla["original"].map(contar_palabras)
    tabla["fragmento"] = tabla["texto"].map(lambda t: t[:150] + ("…" if len(t) > 150 else ""))
    info = dict(modelo.get("metadata", {}))
    # Metadatos externos solo se usan si corresponden exactamente al binario.
    info_path = ruta.with_name("modelo_info.json")
    if not info and info_path.exists():
        externo = json.loads(info_path.read_text(encoding="utf-8"))
        if externo.get("sha256") == hashlib.sha256(ruta.read_bytes()).hexdigest():
            info = externo
    info.setdefault("ids_locales", "ids_referencia" not in modelo)
    return Corpus(modelo, tabla, emb, info)


def vecinos(corpus: Corpus, vector: np.ndarray, texto: str, cantidad: int = 8):
    """Coseno en todas las dimensiones; excluye copias de la propia consulta."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norma = np.linalg.norm(vector)
    if not np.isfinite(norma) or norma == 0:
        raise ValueError("El embedding de la consulta no es válido.")
    similitudes = np.clip(corpus.embeddings @ (vector / norma), -1.0, 1.0)
    distintos = corpus.tabla["clave"].ne(clave_texto(texto)).to_numpy()
    indices = np.flatnonzero(distintos)
    indices = indices[np.argsort(-similitudes[indices], kind="stable")[:cantidad]]
    return indices, similitudes


def analizar(texto: str, corpus: Corpus, encoder) -> dict:
    if not texto or not texto.strip():
        raise ValueError("Escribe una reseña antes de analizar.")
    tokens = encoder.tokenizer(
        texto, truncation=False, add_special_tokens=True,
        return_attention_mask=False, verbose=False,
    )["input_ids"]
    # No limpiar el texto: así fue entrenado el clasificador existente.
    emb = encoder.encode(
        [texto], normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
    modelo = corpus.artefacto
    if emb.shape != (1, modelo["dimension_embedding"]):
        raise ValueError("El encoder y el clasificador tienen dimensiones incompatibles.")
    indices, similitudes = vecinos(corpus, emb[0], texto)
    return {
        "texto": texto,
        "prediccion": str(modelo["clasificador"].predict(emb)[0]),
        "margen": float(modelo["clasificador"].decision_function(emb)[0]),
        "clase_margen_positivo": str(modelo["clasificador"].classes_[1]),
        "xy": modelo["pca"].transform(emb)[0],
        "embedding": emb[0], "indices": indices,
        "similitudes": similitudes, "tokens": len(tokens),
        "limite_tokens": int(encoder.max_seq_length),
    }
