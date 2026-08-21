# RAG

1- Descargar documentos aixiv del area physics.ao-ph total 8440 documentos
2- Generar keywords para la base de conocimientos de documentos usando un modelos de embedding (Qwen-0.6B) y la libreria KeyBert
3- Generar una estadistica inicial de los keywords para ver la distribución de los mismos
4- Calcular el IDF (Invert Document Frecuency) de los keywords sobre los documentos
5- Aglomerar los keywords en grupos que abarquen mas documentos y obtener los keywords que por lo menos tenga un df (frecuencia de aparicion del keywords en el corpus) mayor a 30 documentos
6- Utilizar esos 500 keywords y limipiarlos usando lematizacion y limpieza basica
7- Crear un vocabulario con esas 500 keywords
8- Generar nuevas keywords mediante KeyBert y Qwen-0.6B utilizando como vocabulario los 500 keywords filtradas
9- Filtrar las nuevas keywords descartando los que tenga un df menor a 30
10- Resultado final de las keywords filtradas es:
    "Keywords antes": 33623,
    "Keywords después": 30900,
    "Documentos sin keywords": 10
11- Asegurando que a cada keyword le corresponda como minimo 30 documentos/abstracts
12 - Cargar los datos en faiss usando el modelos de embeddings Qwen-Embedding-8B

#### Resumen de la IA
# KernelRAG

## Descripción

KernelRAG es un proyecto de **Retrieval-Augmented Generation (RAG)** orientado a artículos científicos de arXiv. El objetivo es construir una base de conocimiento que permita recuperar documentos relevantes utilizando tanto **embeddings vectoriales** como un **vocabulario controlado de palabras clave (keywords)**.

En lugar de utilizar directamente las keywords generadas por un modelo, el proyecto desarrolla un proceso de filtrado y normalización para obtener un conjunto de términos representativos que cubran una gran parte del corpus y reduzcan el ruido producido por keywords demasiado específicas o poco frecuentes.

El corpus utilizado está compuesto por **8440 abstracts** de la categoría **physics.ao-ph** de arXiv.

---

# Pipeline

## 1. Obtención del corpus

Se descargan **8440 documentos** de arXiv pertenecientes a la categoría **physics.ao-ph**, utilizando únicamente el resumen (*abstract*) de cada artículo como fuente de información.

---

## 2. Generación inicial de keywords

Para cada abstract se generan keywords mediante:

* **KeyBERT**
* Modelo de embeddings **Qwen-0.6B**

En esta etapa se obtiene un conjunto amplio de keywords candidatas, muchas de ellas muy específicas o presentes en muy pocos documentos.

---

## 3. Análisis estadístico de las keywords

Se calcula la distribución de las keywords obtenidas para analizar características como:

* cantidad total de keywords
* frecuencia de aparición
* distribución de Document Frequency (DF)
* distribución tipo Zipf
* histogramas y gráficos descriptivos

Este análisis permite comprender el comportamiento del vocabulario antes de comenzar el filtrado.

---

## 4. Cálculo del Document Frequency (DF)

Se construye un índice invertido que relaciona cada keyword con los documentos donde aparece.

A partir de este índice se calcula el **Document Frequency (DF)**, es decir, la cantidad de documentos en los que aparece cada keyword.

---

## 5. Cálculo del IDF

Con el DF se calcula el **Inverse Document Frequency (IDF)** de todas las keywords.

Este valor permite distinguir entre:

* keywords demasiado generales
* keywords excesivamente específicas

El IDF sirve como una métrica para analizar la utilidad de cada término dentro del corpus.

---

## 6. Construcción de un vocabulario controlado

El objetivo del proyecto es disponer de un conjunto reducido de keywords capaces de representar una gran cantidad de documentos.

Para ello se:

* eliminan keywords con baja frecuencia
* seleccionan únicamente aquellas con **DF ≥ 30**
* agrupan términos similares
* eliminan duplicados
* aplican normalización (minúsculas, lematización y limpieza básica)
* corrigen variantes triviales del mismo término

Como resultado se obtiene un vocabulario de aproximadamente **500 keywords** que representan una parte importante del corpus.

---

## 7. Segunda extracción de keywords

Se vuelve a ejecutar KeyBERT utilizando el vocabulario construido como restricción (*candidate vocabulary*).

De esta manera, las nuevas keywords sólo pueden pertenecer al vocabulario previamente seleccionado, evitando generar términos nuevos poco útiles.

---

## 8. Filtrado final

Las keywords generadas vuelven a filtrarse eliminando aquellas cuyo **DF sea menor a 30 documentos**.

Resultados obtenidos:

* Keywords antes del filtrado: **33 623**
* Keywords después del filtrado: **30 900**
* Documentos sin keywords: **10**

Esto garantiza que prácticamente todos los documentos posean al menos una keyword representativa y que cada keyword esté respaldada por un número suficiente de documentos.

---

## 9. Indexación vectorial

Finalmente, los abstracts son embebidos utilizando el modelo:

* **Qwen-Embedding-8B**

Los vectores resultantes se almacenan en **FAISS**, permitiendo realizar búsquedas semánticas eficientes sobre el corpus.

---

# Resultado

El proyecto produce:

* un corpus limpio de abstracts científicos;
* un vocabulario controlado de aproximadamente **500 keywords**;
* un índice invertido keyword → documentos;
* métricas DF e IDF para todas las keywords;
* una colección de keywords normalizadas y representativas;
* un índice vectorial en FAISS para recuperación semántica.

Este pipeline reduce significativamente el ruido generado por la extracción automática de keywords y mejora la capacidad de recuperación de documentos dentro de un sistema RAG. 
