"""Inferencia con cláusulas definidas, variables y términos sin funciones.

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys


@dataclass(frozen=True)
class Atomo:
    predicado: str
    argumentos: tuple

    def __str__(self):
        return self.predicado + "(" + ",".join(self.argumentos) + ")"


@dataclass(frozen=True)
class Regla:
    nombre: str
    premisas: tuple
    cabeza: Atomo


def atomo(texto):
    coincidencia = re.fullmatch(r"\s*([A-Za-z_ÁÉÍÓÚáéíóúÑñ]\w*)\s*\(([^()]*)\)\s*", texto)
    if not coincidencia:
        raise ValueError("Átomo no válido: " + texto)
    terminos = tuple(t.strip() for t in coincidencia[2].split(","))
    if any(not re.fullmatch(r"[\wÁÉÍÓÚáéíóúÑñ]+", t) for t in terminos):
        raise ValueError("Se admiten variables y constantes sin funciones: " + texto)
    return Atomo(coincidencia[1], terminos)


def variable(termino):
    return termino[0].islower()


def resolver(termino, sustitucion):
    while variable(termino) and termino in sustitucion:
        termino = sustitucion[termino]
    return termino


def aplicar(hecho, sustitucion):
    return Atomo(hecho.predicado, tuple(resolver(t, sustitucion) for t in hecho.argumentos))


def unificar(izquierdo, derecho, sustitucion=None):
    if izquierdo.predicado != derecho.predicado or len(izquierdo.argumentos) != len(derecho.argumentos):
        return None
    nueva = dict(sustitucion or {})
    for a, b in zip(izquierdo.argumentos, derecho.argumentos):
        a, b = resolver(a, nueva), resolver(b, nueva)
        if a == b:
            continue
        if variable(a):
            nueva[a] = b
        elif variable(b):
            nueva[b] = a
        else:
            return None
    return nueva


def renombrar(regla, numero):
    """Cada uso tiene variables frescas para evitar mezclar sustituciones."""
    nombres = {}

    def nuevo(a):
        argumentos = []
        for t in a.argumentos:
            if variable(t):
                nombres.setdefault(t, f"{t}_{numero}")
                t = nombres[t]
            argumentos.append(t)
        return Atomo(a.predicado, tuple(argumentos))

    return Regla(regla.nombre, tuple(nuevo(p) for p in regla.premisas), nuevo(regla.cabeza))


def cargar(ruta):
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    hechos = [atomo(f) for f in datos["hechos"]]
    if any(variable(t) for f in hechos for t in f.argumentos):
        raise ValueError("Los hechos deben contener constantes, no variables.")
    reglas = [Regla(r["id"], tuple(atomo(p) for p in r["si"]), atomo(r["entonces"]))
              for r in datos["reglas"]]
    consultas = [atomo(c) for c in datos["consultas"]]
    return hechos, reglas, consultas


def satisfacer(premisas, hechos, sustitucion):
    if not premisas:
        yield sustitucion
        return
    primera = aplicar(premisas[0], sustitucion)
    for hecho in hechos:
        nueva = unificar(primera, hecho, sustitucion)
        if nueva is not None:
            yield from satisfacer(premisas[1:], hechos, nueva)


def adelante(hechos_iniciales, reglas, consulta):
    hechos, nuevos, traza = list(hechos_iniciales), [], [f"Consulta: {consulta}"]
    contador, ronda = 0, 0
    while True:
        ronda += 1
        pendientes = []
        # La instantánea impide usar un hecho nuevo en esta misma ronda.
        instantanea = tuple(hechos)
        traza.append(f"Ronda {ronda}:")
        for regla in reglas:
            contador += 1
            fresca = renombrar(regla, contador)
            for sustitucion in satisfacer(fresca.premisas, instantanea, {}):
                cabeza = aplicar(fresca.cabeza, sustitucion)
                if any(variable(t) for t in cabeza.argumentos):
                    raise ValueError("La cabeza de una regla debe estar ligada por sus premisas.")
                if cabeza not in hechos and cabeza not in pendientes:
                    pendientes.append(cabeza)
                    traza.append(f"  {regla.nombre} -> {cabeza}")
        if not pendientes:
            traza.append("  Sin hechos nuevos: terminó el encadenamiento.")
            break
        hechos.extend(pendientes)
        nuevos.extend(pendientes)
    respuesta = next((s for h in hechos if (s := unificar(consulta, h)) is not None), None)
    probada = respuesta is not None
    traza.append("Resultado: " + ("SÍ, probada." if probada else "NO, no probada con esta base."))
    return {"probada": probada, "nuevos": list(map(str, nuevos)),
            "hechos_finales": list(map(str, hechos)), "sustitucion": respuesta, "traza": traza}


def clave_meta(meta):
    # Dos metas que solo cambian el nombre de sus variables son el mismo ciclo.
    nombres = {}
    args = []
    for t in meta.argumentos:
        if variable(t):
            nombres.setdefault(t, len(nombres))
            args.append(("variable", nombres[t]))
        else:
            args.append(("constante", t))
    return meta.predicado, tuple(args)


def atras(hechos, reglas, consulta):
    traza, contador = [f"Consulta: {consulta}"], [0]

    def probar_meta(meta, sustitucion, pila, nivel):
        meta = aplicar(meta, sustitucion)
        espacio = "  " * nivel
        clave = clave_meta(meta)
        traza.append(f"{espacio}Meta: {meta}")
        if clave in pila:
            traza.append(f"{espacio}Ciclo: esta meta ya está en la pila; se corta este camino.")
            return None
        for hecho in hechos:
            nueva = unificar(meta, hecho, sustitucion)
            if nueva is not None:
                traza.append(f"{espacio}Hecho encontrado: {hecho}")
                return nueva
        for regla in reglas:
            contador[0] += 1
            fresca = renombrar(regla, contador[0])
            nueva = unificar(meta, fresca.cabeza, sustitucion)
            if nueva is None:
                continue
            traza.append(f"{espacio}Intentar {regla.nombre} para {meta}")
            resultado = probar_premisas(fresca.premisas, nueva, pila + (clave,), nivel + 1)
            if resultado is not None:
                traza.append(f"{espacio}Éxito: {meta} por {regla.nombre}")
                return resultado
            traza.append(f"{espacio}Falló {regla.nombre} para {meta}")
        traza.append(f"{espacio}Fallo: no se pudo probar {meta}")
        return None

    def probar_premisas(premisas, sustitucion, pila, nivel):
        if not premisas:
            return sustitucion
        nueva = probar_meta(premisas[0], sustitucion, pila, nivel)
        if nueva is None:
            return None
        return probar_premisas(premisas[1:], nueva, pila, nivel)

    respuesta = probar_meta(consulta, {}, (), 0)
    probada = respuesta is not None
    traza.append("Resultado: " + ("SÍ, probada." if probada else "NO, no probada con esta base."))
    return {"probada": probada, "sustitucion": respuesta, "traza": traza,
            "hechos_agregados": []}


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=base / "biblioteca.json")
    parser.add_argument("--guardar", type=Path, metavar="CARPETA",
                        help="Guardar las trazas y las respuestas en esta carpeta, si se desea.")
    args = parser.parse_args()
    hechos, reglas, consultas = cargar(args.base)
    if args.guardar:
        resultados, evidencias = args.guardar / "resultados", args.guardar / "evidencias"
        resultados.mkdir(parents=True, exist_ok=True)
        evidencias.mkdir(parents=True, exist_ok=True)
    salida = ["EJERCICIO 9 - ENCADENAMIENTO EN LA BIBLIOTECA",
              "Ejecución real local de código propio.",
              f"Comando: {sys.executable} {__file__}", f"Python: {sys.version.split()[0]}",
              "Base original sin agregar hechos. Archivo: " + args.base.name, ""]
    registros = {}
    for consulta in consultas:
        nombre = str(consulta)
        registros[nombre] = {}
        for etiqueta, funcion in (("adelante", adelante), ("atras", atras)):
            resultado = funcion(hechos, reglas, consulta)
            registros[nombre][etiqueta] = resultado
            texto = "\n".join([f"ENCADENAMIENTO HACIA {etiqueta.upper()}"] + resultado["traza"]) + "\n"
            persona = consulta.argumentos[0]
            if args.guardar:
                (evidencias / f"{persona}_{etiqueta}.txt").write_text(texto, encoding="utf-8")
            salida += [texto]

    consulta_luis = atomo("AccesoNoche(Luis)")
    hechos_extra = hechos + [atomo("TieneCredencial(Luis)")]
    contrafactual = {"hecho_agregado": "TieneCredencial(Luis)",
                    "adelante": adelante(hechos_extra, reglas, consulta_luis),
                    "atras": atras(hechos_extra, reglas, consulta_luis)}
    antes = set(registros["AccesoNoche(Luis)"]["adelante"]["hechos_finales"])
    contrafactual["consecuencias_adicionales"] = [
        h for h in contrafactual["adelante"]["nuevos"] if h not in antes]
    salida += ["CASO SEPARADO: AGREGAR TieneCredencial(Luis)",
               "No cambia biblioteca.json ni las tres consultas anteriores."]
    for etiqueta in ("adelante", "atras"):
        texto = "\n".join([f"ENCADENAMIENTO HACIA {etiqueta.upper()}"] + contrafactual[etiqueta]["traza"]) + "\n"
        if args.guardar:
            (evidencias / f"Luis_con_credencial_{etiqueta}.txt").write_text(texto, encoding="utf-8")
        salida.append(texto)
    salida += ["Consecuencias adicionales respecto a la base original: " +
               ", ".join(contrafactual["consecuencias_adicionales"])]
    if args.guardar:
        (resultados / "metricas.json").write_text(json.dumps({"consultas": registros,
            "caso_luis_con_credencial": contrafactual}, ensure_ascii=False, indent=2), encoding="utf-8")
    texto_total = "\n".join(salida) + "\n"
    if args.guardar:
        (evidencias / "ejecucion.txt").write_text(texto_total, encoding="utf-8")
    print(texto_total, end="")


if __name__ == "__main__":
    main()
