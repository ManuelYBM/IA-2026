"""Q-learning para Cliff Walking 4x12. 
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ACCIONES = ("N", "E", "S", "W")  # El primer máximo gana los empates.
CAMBIOS = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
INICIO, META = (3, 0), (3, 11)
FILAS, COLUMNAS = 4, 12


def paso(estado, accion):
    """Regresa (siguiente estado, recompensa, terminó, cayó)."""
    if estado == META:
        raise ValueError("La meta es terminal: no se ejecutan acciones desde ella.")
    df, dc = CAMBIOS[accion]
    fila = max(0, min(FILAS - 1, estado[0] + df))
    columna = max(0, min(COLUMNAS - 1, estado[1] + dc))
    siguiente = fila, columna
    if fila == 3 and 1 <= columna <= 10:
        return INICIO, -100, False, True
    return siguiente, -1, siguiente == META, False


def tabla_vacia():
    return {
        (f, c): {a: 0.0 for a in ACCIONES}
        for f in range(FILAS)
        for c in range(COLUMNAS)
        if (f, c) != META
    }


def mejor_accion(q, estado):
    return max(ACCIONES, key=lambda a: q[estado][a])


def epsilon_greedy(q, estado, epsilon, generador):
    if generador.random() < epsilon:
        return generador.choice(ACCIONES)
    return mejor_accion(q, estado)


def actualizar(q, estado, accion, recompensa, siguiente, terminal, alfa, gamma):
    # Caer al acantilado NO es terminal: el futuro usa Q(INICIO).
    futuro = 0.0 if terminal else max(q[siguiente].values())
    objetivo = recompensa + gamma * futuro
    anterior = q[estado][accion]
    q[estado][accion] = anterior + alfa * (objetivo - anterior)
    return q[estado][accion]


def entrenar(semilla=42, episodios=500, alfa=0.5, gamma=1.0, epsilon=0.1, tope=200):
    generador = random.Random(semilla)
    q = tabla_vacia()
    registros = []
    for episodio in range(1, episodios + 1):
        estado, retorno, caidas = INICIO, 0, 0
        for pasos in range(1, tope + 1):
            accion = epsilon_greedy(q, estado, epsilon, generador)
            siguiente, recompensa, terminal, cayo = paso(estado, accion)
            actualizar(q, estado, accion, recompensa, siguiente, terminal, alfa, gamma)
            retorno += recompensa
            caidas += int(cayo)
            estado = siguiente
            if terminal:
                break
        registros.append({"episodio": episodio, "retorno": retorno, "pasos": pasos,
                          "caidas": caidas, "llego_meta": terminal})
    return q, registros


def evaluar(q, tope=200):
    estado, retorno = INICIO, 0
    camino, acciones = [estado], []
    for _ in range(tope):
        accion = mejor_accion(q, estado)
        estado, recompensa, terminal, _ = paso(estado, accion)
        acciones.append(accion)
        camino.append(estado)
        retorno += recompensa
        if terminal:
            break
    return {"camino": camino, "acciones": acciones, "pasos": len(acciones),
            "retorno": retorno, "llego_meta": terminal, "epsilon": 0}


def politica_texto(q):
    filas = ["       " + " ".join(f"{c:>2}" for c in range(COLUMNAS))]
    for f in range(FILAS):
        casillas = []
        for c in range(COLUMNAS):
            if (f, c) == META:
                texto = "G"
            elif f == 3 and 1 <= c <= 10:
                texto = "C"
            elif (f, c) == INICIO:
                texto = "I"
            else:
                texto = mejor_accion(q, (f, c))
            casillas.append(f"{texto:>2}")
        filas.append(f"fila {f} " + " ".join(casillas))
    return "\n".join(filas)


def graficar(registros, destino):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valores = [r["retorno"] for r in registros]
    media = [sum(valores[max(0, i - 9):i + 1]) / min(i + 1, 10)
             for i in range(len(valores))]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(valores) + 1), valores, color="0.65", linewidth=0.6,
            label="Retorno por episodio (exploración activa)")
    ax.plot(range(1, len(media) + 1), media, color="black", linewidth=1.3,
            label="Media móvil de 10 episodios")
    ax.axhline(-13, color="black", linestyle="--", linewidth=1,
               label="Referencia greedy óptima: -13")
    ax.set(xlabel="Episodio", ylabel="Recompensa acumulada",
           title="Q-learning: 500 episodios, semilla 42")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(destino, dpi=160)
    plt.close(fig)


def graficar_politica(evaluacion, destino):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(11, 4.2))
    for f in range(FILAS):
        for c in range(COLUMNAS):
            acantilado = f == 3 and 1 <= c <= 10
            ax.add_patch(Rectangle((c, f), 1, 1, facecolor="0.85" if acantilado else "white",
                                   edgecolor="black", linewidth=0.8))
            if acantilado:
                ax.text(c + 0.5, f + 0.5, "C", ha="center", va="center")
    for origen, destino_paso in zip(evaluacion["camino"], evaluacion["camino"][1:]):
        ax.annotate("", (destino_paso[1] + 0.5, destino_paso[0] + 0.5),
                    (origen[1] + 0.5, origen[0] + 0.5),
                    arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.8})
    ax.text(0.5, 3.7, "S", ha="center", va="center")
    ax.text(11.5, 3.7, "G", ha="center", va="center")
    ax.set(xlim=(0, 12), ylim=(4, 0), aspect="equal",
           title="Política greedy desde S: N, 11 E, S | 13 pasos | retorno -13")
    ax.set_xticks([c + 0.5 for c in range(12)], labels=range(12))
    ax.set_yticks([f + 0.5 for f in range(4)], labels=range(4))
    ax.set_xlabel("Columna")
    ax.set_ylabel("Fila")
    fig.tight_layout()
    fig.savefig(destino, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sin-grafica", action="store_true", help="No requiere matplotlib.")
    parser.add_argument("--guardar", type=Path, metavar="CARPETA",
                        help="Guardar los datos y las figuras en esta carpeta, si se desea.")
    args = parser.parse_args()
    q, registros = entrenar()
    evaluacion = evaluar(q)
    parametros = {"semilla": 42, "episodios": 500, "alfa": 0.5, "gamma": 1,
                  "epsilon_entrenamiento": 0.1, "q_inicial": 0, "tope_pasos": 200,
                  "desempate": list(ACCIONES)}
    resumen = {"parametros": parametros, "evaluacion_greedy": evaluacion,
               "episodios_cortados": sum(not r["llego_meta"] for r in registros),
               "media_primeros_50": sum(r["retorno"] for r in registros[:50]) / 50,
               "media_ultimos_50": sum(r["retorno"] for r in registros[-50:]) / 50,
               "caidas_totales": sum(r["caidas"] for r in registros)}
    if args.guardar:
        resultados, evidencias = args.guardar / "resultados", args.guardar / "evidencias"
        resultados.mkdir(parents=True, exist_ok=True)
        evidencias.mkdir(parents=True, exist_ok=True)
        with (resultados / "episodios.csv").open("w", newline="", encoding="utf-8") as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=list(registros[0]))
            escritor.writeheader()
            escritor.writerows(registros)
        (resultados / "metricas.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")
        serial_q = {f"{s[0]},{s[1]}": valores for s, valores in q.items()}
        (resultados / "tabla_q.json").write_text(json.dumps(serial_q, indent=2), encoding="utf-8")
        (resultados / "politica.txt").write_text(politica_texto(q) + "\n", encoding="utf-8")
    lineas = ["EJERCICIO 8 - Q-LEARNING EN CLIFF WALKING",
              "Ejecución local.",
              f"Comando: {sys.executable} {__file__}" + (" --sin-grafica" if args.sin_grafica else ""),
              f"Python: {sys.version.split()[0]}",
              "Parámetros: semilla=42, alpha=0.5, gamma=1, epsilon=0.1, episodios=500.",
              "Desempate: N, E, S, W. Tope: 200 pasos por episodio", "",
              "TRANSICIONES COMPROBADAS:",
              f"S, N -> {paso(INICIO, 'N')}",
              f"S, E -> {paso(INICIO, 'E')}",
              f"(2,11), S -> {paso((2, 11), 'S')}",
              "Tuplas: (siguiente, recompensa, terminal, cayó).", "",
              "POLÍTICA GREEDY (N=norte, E=este, S=sur, W=oeste; I=inicio, C=acantilado):",
              politica_texto(q), "", "EVALUACIÓN SIN EXPLORACIÓN (epsilon=0):",
              "Acciones: " + " -> ".join(evaluacion["acciones"]),
              "Camino: " + " -> ".join(map(str, evaluacion["camino"])),
              f"Pasos: {evaluacion['pasos']}; retorno: {evaluacion['retorno']}; llegó: {evaluacion['llego_meta']}",
              f"Episodios cortados por el tope: {resumen['episodios_cortados']}",
              f"Media primeros 50 episodios: {resumen['media_primeros_50']:.2f}",
              f"Media últimos 50 episodios: {resumen['media_ultimos_50']:.2f}",
              f"Caídas en todo el entrenamiento: {resumen['caidas_totales']}"]
    if args.guardar and not args.sin_grafica:
        graficar(registros, resultados / "recompensas.png")
        graficar_politica(evaluacion, resultados / "politica.png")
    if args.guardar:
        lineas.append(f"Datos guardados en: {args.guardar}")
    texto = "\n".join(lineas) + "\n"
    print(texto, end="")
    if args.guardar:
        (evidencias / "ejecucion.txt").write_text(texto, encoding="utf-8")
    if not (evaluacion["pasos"] == 13 and evaluacion["retorno"] == -13
            and evaluacion["llego_meta"]):
        raise RuntimeError("La evaluación no cumple el objetivo de la consigna")


if __name__ == "__main__":
    main()
