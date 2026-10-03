"""Gráficos de las vistas V2 y V3."""
import html
import numpy as np
import plotly.graph_objects as go

from analisis import COLORES, ETIQUETAS


def grafico_mapa(corpus, filtro, resultado, seleccionado, revision):
    tabla = corpus.tabla
    fig = go.Figure()
    for clase in ["positive", "negative"]:
        if filtro != "Todos" and ETIQUETAS[clase] != filtro:
            continue
        parte = tabla.loc[tabla["sentimiento"].eq(clase)]
        custom = [
            [int(i), fila["id"], html.escape(fila["fragmento"]), ETIQUETAS[clase]]
            for i, fila in parte.iterrows()
        ]
        fig.add_trace(go.Scattergl(
            x=parte["pc1"], y=parte["pc2"], mode="markers",
            name=ETIQUETAS[clase], customdata=custom,
            marker={"color": COLORES[clase], "size": 7, "opacity": 0.52},
            hovertemplate="<b>%{customdata[1]} · %{customdata[3]}</b><br>"
                          "%{customdata[2]}<extra></extra>",
        ))
    if resultado is not None:
        # Los ocho vecinos se distinguen, respetando el filtro del mapa.
        indices = [int(i) for i in resultado["indices"] if filtro == "Todos"
                   or ETIQUETAS[tabla.iloc[i]["sentimiento"]] == filtro]
        if indices:
            parte = tabla.iloc[indices]
            fig.add_trace(go.Scattergl(
                x=parte["pc1"], y=parte["pc2"], mode="markers", name="Vecinos",
                customdata=[[i, tabla.iloc[i]["id"]] for i in indices],
                marker={"size": 12, "symbol": "circle-open", "color": "#677589", "line": {"width": 1}},
                hovertemplate="<b>%{customdata[1]} · vecino</b><extra></extra>",
            ))
        fig.add_trace(go.Scattergl(
            x=[float(resultado["xy"][0])], y=[float(resultado["xy"][1])],
            mode="markers", name="Tu reseña", customdata=[[-1, "CONSULTA"]],
            marker={"size": 18, "symbol": "star", "color": "#172637", "line": {"width": 1, "color": "white"}},
            hovertemplate="<b>Tu reseña</b><extra></extra>",
        ))
    if seleccionado is not None:
        fila = tabla.iloc[seleccionado]
        if filtro == "Todos" or ETIQUETAS[fila["sentimiento"]] == filtro:
            fig.add_trace(go.Scattergl(
                x=[fila["pc1"]], y=[fila["pc2"]], mode="markers", name="Seleccionada",
                customdata=[[int(seleccionado), fila["id"]]],
                marker={"size": 17, "symbol": "circle-open", "color": "#101820", "line": {"width": 3}},
                hovertemplate="<b>%{customdata[1]} · seleccionada</b><extra></extra>",
            ))
    fig.update_layout(
        height=470, margin={"l": 10, "r": 10, "t": 15, "b": 10},
        xaxis_title="Componente principal 1", yaxis_title="Componente principal 2",
        paper_bgcolor="white", plot_bgcolor="#F7F9FC", dragmode="pan",
        clickmode="event+select", uirevision=str(revision),
        legend={"orientation": "h", "y": 1.1, "itemclick": False, "itemdoubleclick": False},
        font={"family": "Arial, sans-serif", "color": "#182638"},
    )
    return fig


def grafico_vecinos(corpus, resultado, seleccionado):
    indices = [int(i) for i in resultado["indices"]]
    tabla = corpus.tabla.iloc[indices]
    valores = resultado["similitudes"][indices]
    labels = [f"{j}. {id_}" for j, id_ in enumerate(tabla["id"], start=1)]
    fig = go.Figure(go.Bar(
        x=valores, y=labels, orientation="h",
        customdata=[[i, ETIQUETAS[corpus.tabla.iloc[i]["sentimiento"]]] for i in indices],
        marker={
            "color": [COLORES[c] for c in tabla["sentimiento"]],
            "line": {"color": "#172637", "width": [3 if i == seleccionado else 0 for i in indices]},
        },
        text=[f"{v:.3f}" for v in valores], textposition="auto",
        hovertemplate="<b>%{y}</b><br>%{customdata[1]}<br>Coseno: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=300, margin={"l": 5, "r": 10, "t": 5, "b": 5},
        xaxis={"title": "Similitud coseno", "range": [min(0.0, float(np.min(valores))), 1.0]},
        yaxis={"autorange": "reversed"}, showlegend=False,
        paper_bgcolor="white", plot_bgcolor="white", clickmode="event+select",
        font={"family": "Arial, sans-serif", "color": "#182638"},
    )
    return fig
