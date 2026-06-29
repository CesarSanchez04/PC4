# Proyecto 2: Embeddings y Búsqueda Semántica
**Curso:** CC0C2 - Procesamiento del Lenguaje Natural  
**Estudiante:** César (CC0C2 Estudiante)  
**Semestre:** 2026-I  
**Sustentación:** Práctica Calificada 4

---

## 🎯 Objetivo
El objetivo de este proyecto es implementar, evaluar y comparar un sistema de recuperación de información híbrido que combina la búsqueda léxica basada en coincidencia exacta de términos (BM25) con la búsqueda semántica densa basada en representaciones vectoriales continuas (Embeddings).

El sistema se enfoca en un corpus especializado de **15 documentos** sobre la película *Interstellar* (física de agujeros negros, relatividad general, relaciones familiares y banda sonora). Se analiza cuantitativa y cualitativamente el impacto del modelo de representación (`BAAI/bge-small-en-v1.5` vs. `sentence-transformers/all-MiniLM-L6-v2`) y la métrica de distancia espacial (Similitud Coseno vs. Distancia Euclidiana $L_2$).

---

## 📘 Cuaderno Base
Este desarrollo toma como base el **[Cuaderno 21 - CC0C2](file:///Users/cesar/Desktop/CC-0C2/Semana10/Cuaderno21-CC0C2.ipynb)** del curso, correspondiente a la introducción a Embeddings, Búsqueda Semántica, Chunking y Vector Stores.

---

## ⚙️ Resumen de la Línea Base
La línea base implementa:
1. Un motor de búsqueda léxico usando **BM25Okapi** con tokenización básica en minúsculas y eliminación de puntuación.
2. Un indexador denso utilizando el modelo **`BAAI/bge-small-en-v1.5`** (dimensión $d = 384$) con dos índices de la librería **FAISS**:
   * `IndexFlatIP`: Para búsqueda por producto punto (equivalente a Similitud Coseno al normalizar los vectores).
   * `IndexFlatL2`: Para búsqueda mediante distancia Euclidiana $L_2$ con embeddings crudos.

---

## 🛠️ Modificación Realizada
Se extendió y robusteció la línea base mediante las siguientes adiciones:
1. **Normalización y Fusión Híbrida:** Implementación de un fusionador lineal de puntajes normalizados vía Min-Max (`search_hybrid` con parámetro $\alpha$ para balancear peso léxico y denso).
2. **Evaluación Sistemática:** Creación de un dataset de prueba (*gold standard*) de 5 consultas específicas con sus respectivos documentos relevantes etiquetados manualmente.
3. **Cálculo de Métricas:** Medición de calidad usando **Mean Recall@3** y **Mean MRR@3**.
4. **Codificación en Vivo A (Similitud Coseno Manual):** Creación de una función NumPy (`manual_cosine_similarity`) para comprobar algebraicamente cómo la normalización unitaria previa reduce la similitud coseno a un simple producto interno en FAISS.
5. **Codificación en Vivo B (Comparación Semántica):** Celda interactiva de selección de modelo (`chosen_model = "A" | "B"`) para comparar de forma directa la representación de consultas abstractas sin coincidencia léxica.

---

## 📊 Principales Resultados (Métricas)

La evaluación cuantitativa sobre el benchmark de 5 consultas arroja los siguientes resultados:

| Configuración de Recuperación | Mean Recall@3 | Mean MRR@3 |
|---|:---:|:---:|
| **Búsqueda Léxica (BM25)** | `1.0000` | `1.0000` |
| **Dense A (BGE-small) - Coseno** | `0.7333` | `1.0000` |
| **Dense A (BGE-small) - L2** | `0.7333` | `1.0000` |
| **Dense B (all-MiniLM) - Coseno** | `0.7333` | `1.0000` |
| **Híbrido A (BGE-small + BM25, $\alpha=0.5$)** | `0.9333` | `1.0000` |
| **Híbrido B (all-MiniLM + BM25, $\alpha=0.5$)** | `1.0000` | `1.0000` |

> [!NOTE]
> Debido al tamaño acotado del corpus (15 documentos) y a la alta especificidad léxica de las consultas de prueba, BM25 obtiene un Recall perfecto. Sin embargo, en el análisis cualitativo ante consultas de alta abstracción semántica (como *parent-child emotional bond*), los modelos de embeddings densos demuestran recuperar información relevante donde BM25 falla por completo.

---

## ⚠️ Limitaciones
1. **Volumen del Corpus:** El corpus de 15 documentos facilita la interpretabilidad de los rankings, pero no permite apreciar la degradación de latencia o las colisiones espaciales en una base vectorial de millones de chunks.
2. **Consultas Sintéticas:** Las consultas del benchmark están optimizadas para describir con precisión temática las secciones del corpus.
3. **Hardware y Red:** Las ejecuciones y cargas de modelos se realizan de forma local en CPU, requiriendo el uso de modelos compactos de parámetros reducidos (384 dimensiones) en lugar de modelos masivos (ej. 1536 dimensiones) o APIs propietarias.

---

## 🚀 Cómo Ejecutar el Notebook
1. **Preparación del Entorno:**
   Asegúrate de contar con Python 3.10+ y el entorno virtual activo:
   ```bash
   source .venv/bin/activate
   ```
2. **Instalación de Dependencias:**
   Instala los paquetes necesarios definidos en `requirements-base.txt` y `requirements-opcional.txt`:
   ```bash
   pip install numpy pandas sentence-transformers rank_bm25 faiss-cpu
   ```
3. **Lanzar Jupyter:**
   Abre el entorno gráfico de Jupyter:
   ```bash
   jupyter notebook
   ```
4. **Ejecución:**
   Abre el notebook [Proyecto2_Embeddings_BusquedaSemantica.ipynb](file:///Users/cesar/Desktop/CC-0C2/Semana11/Proyecto2_Embeddings_BusquedaSemantica.ipynb) y ejecuta todas las celdas en orden. Asegúrate de que la **Celda de Verificación Personal** muestre correctamente la semilla `42`.

---

## 🎥 Qué se muestra en el Video Técnico
La defensa audiovisual tiene una duración superior a los 12 minutos y sigue la estructura obligatoria:
* **0:00 - 2:30 | Introducción:** Presentación personal, marco experimental, semilla (42) y aislamiento de variables (dimensión vectorial constante $d=384$ para ambos modelos).
* **2:30 - 6:30 | Marco Teórico:** Discusión de ecuaciones de BM25, pooling tokens (Mean vs. CLS), y métricas espaciales (Coseno vs. $L_2$).
* **6:30 - 8:30 | Ingesta de Datos:** Carga del corpus de Interstellar y configuración de índices FAISS.
* **8:30 - 13:30 | Codificación en Vivo A:** Programación manual de similitud coseno en NumPy (`manual_cosine_similarity`) y verificación de equivalencia contra el índice `IndexFlatIP`.
* **13:30 - 18:30 | Codificación en Vivo B:** Modificación del parámetro `chosen_model` (de `"A"` a `"B"`) en vivo, prediciendo cómo cambia el ranking semántico para la consulta abstracta `'emotional bond of parent and child'` y por qué BGE-small es superior.
* **18:30 - 20:00 | Cierre:** Análisis de métricas, análisis de limitaciones y puente conceptual al curso.

---

## ⚖️ Declaración de Autoría y Uso de IA
```
Declaro que comprendo el código, los resultados y las explicaciones entregadas en esta Práctica.
Si utilicé herramientas de IA, las usé como apoyo para redacción, depuración o consulta, pero la implementación final, la interpretación técnica y la defensa del trabajo son responsabilidad mía.
```
* **Detalle del apoyo de IA:** Se utilizó un LLM para la estructuración y formateo del archivo README.md, así como para la depuración de la alineación de índices de FAISS en la búsqueda híbrida.

---

## 📝 Conclusión Final (Qué hice, por qué lo hice y qué significan mis resultados)
* **Qué hice:** Construí un framework de recuperación híbrido (léxico-semántico) sobre un corpus especializado de *Interstellar*, comparando modelos de embeddings densos con dimensiones aisladas y métricas de distancia geométricas.
* **Por qué lo hice:** Para estudiar cómo afecta la capacidad inherente de los modelos de representación vectorial a la recuperación semántica pura frente a la búsqueda exacta por palabras clave, garantizando condiciones de comparación homogéneas.
* **Qué significan mis resultados:** Que los enfoques híbridos son indispensables: BM25 garantiza precisión léxica ante consultas con palabras exactas, pero la búsqueda densa con modelos optimizados mediante aprendizaje contrastivo (como `BGE-small`) es la única capaz de resolver la polisemia y capturar conceptos abstractos complejos.

---

## 🌉 Puente al Curso
Este proyecto se integra de forma directa con los siguientes conceptos del curso:
1. **Búsqueda Semántica:** Demostración práctica de cómo se supera la limitación léxica de coincidencias exactas mediante la representación densa de oraciones en un espacio latente continuo.
2. **Evaluación de Retrieval:** Uso pragmático de Recall y MRR para auditar y sintonizar la calidad del componente de búsqueda, que constituye la primera etapa (y el cuello de botella) de cualquier arquitectura RAG productiva.
