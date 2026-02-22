import json

# Notebook 1 mejorado: spaCy NLP
notebook1 = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# Procesamiento de Lenguaje Natural con spaCy\n',
                '## Análisis del texto: "El Corazón Delator" de Edgar Allan Poe\n',
                '\n',
                '[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WillianReinaG/PNL_unidad1/blob/main/1_NLP_spacy_ElCorazonDelator.ipynb)\n',
                '\n',
                'Este notebook realiza un **análisis completo de Procesamiento de Lenguaje Natural (NLP)** en español utilizando **spaCy**.\n',
                '\n',
                '**Contenidos:**\n',
                '- Tokenización y análisis de tokens\n',
                '- Etiquetado de partes del lenguaje (POS tagging)\n',
                '- Análisis de dependencias (Dependency parsing)\n',
                '- Extracción de sintagmas nominales\n',
                '- Identificación de verbos\n',
                '- Reconocimiento de entidades nombradas (NER)\n',
                '- Búsqueda de patrones con Matcher\n',
                '- Análisis de frecuencia de palabras'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 📦 Paso 1: Instalación de dependencias\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Instala spaCy (librería de NLP) y descarga el modelo pre-entrenado para español `es_core_news_sm`. Este modelo incluye:\n',
                '- Tokenizador\n',
                '- Etiquetador de POS (Part-of-Speech)\n',
                '- Analizador de dependencias\n',
                '- Reconocedor de entidades\n',
                '\n',
                '**Nota:** En Google Colab, esto puede tomar 2-3 minutos.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '!pip install spacy -q\n',
                '!python -m spacy download es_core_news_sm -q\n',
                'print("✓ spaCy instalado correctamente")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 📚 Paso 2: Importar librerías necesarias\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Carga las librerías que utilizaremos:\n',
                '- **spacy**: Para procesamiento NLP\n',
                '- **Matcher**: Para búsqueda de patrones en texto\n',
                '- **pandas**: Para manipulación de datos\n',
                '- **Counter**: Para contar frecuencias\n',
                '\n',
                'También carga el modelo de español que descargamos.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'import spacy\n',
                'from spacy.matcher import Matcher\n',
                'import pandas as pd\n',
                'from collections import Counter\n',
                '\n',
                '# Cargar el modelo de español\n',
                'nlp = spacy.load("es_core_news_sm")\n',
                'print("✓ Modelo de español cargado")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 📄 Paso 3: Cargar el texto a analizar\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Lee el archivo de texto "El Corazón Delator.txt" que contiene el relato completo.\n',
                'En Google Colab, puedes subir archivos locales o conectarte a Google Drive.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'with open("El_corazón_delator.txt", "r", encoding="utf-8") as f:\n',
                '    texto = f.read()\n',
                '\n',
                'print(f"Texto cargado correctamente")\n',
                'print(f"Longitud: {len(texto)} caracteres")\n',
                'print(f"\\nPrimer párrafo (primeros 200 caracteres):")\n',
                'print(texto[:200] + "...")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## ⚙️ Paso 4: Procesar el texto con spaCy\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Aplica el pipeline de spaCy al texto. Esto significa:\n',
                '1. **Tokenización**: Divide el texto en tokens (palabras, puntuación, etc.)\n',
                '2. **Análisis morfológico**: Analiza partes del lenguaje y lemas\n',
                '3. **Análisis sintáctico**: Detecta dependencias entre palabras\n',
                '4. **NER**: Identifica entidades nombradas (personas, lugares, etc.)\n',
                '\n',
                '⏱️ Esto puede tomar algunos segundos.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# Procesar el texto\n',
                'doc = nlp(texto)\n',
                'print("✓ Texto procesado con spaCy")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 🔢 Paso 5: Análisis de TOKENS\n',
                '\n',
                '### ¿Qué es un token?\n',
                'Un token es la unidad más pequeña de análisis: una palabra, número, signo de puntuación, etc.\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Cuenta el número total de tokens en el documento.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'total_tokens = len(doc)\n',
                'print(f"📊 Total de tokens en el archivo: {total_tokens}")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 📝 Paso 6: Análisis de ORACIONES\n',
                '\n',
                '### ¿Qué es una oración?\n',
                'Una oración es una secuencia de tokens que forma una unidad gramatical completa.\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Cuenta el número total de oraciones y muestra las primeras 3 como ejemplo.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'sentences = list(doc.sents)\n',
                'total_sentences = len(sentences)\n',
                '\n',
                'print(f"📊 Total de oraciones: {total_sentences}")\n',
                'print(f"\\nPrimeras 3 oraciones:")\n',
                'for i, sent in enumerate(sentences[:3], 1):\n',
                '    print(f"{i}. {sent.text[:100]}...")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 🎯 Paso 7: Extracción de la TERCERA ORACIÓN\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Extrae y muestra la tercera oración del documento completo.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'if len(sentences) >= 3:\n',
                '    tercera_oracion = sentences[2]\n',
                '    print("📌 Tercera oración del documento:")\n',
                '    print(f"\\n{tercera_oracion.text}")\n',
                'else:\n',
                '    print(f"El documento tiene solo {len(sentences)} oraciones")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 🔍 Paso 8: Análisis DETALLADO de tokens de la tercera oración\n',
                '\n',
                '### ¿Qué son POS tags y DEP tags?\n',
                '- **POS tag (Part-Of-Speech)**: La categoría gramatical (NOUN=sustantivo, VERB=verbo, ADJ=adjetivo, etc.)\n',
                '- **DEP tag (Dependency tag)**: La relación sintáctica de la palabra en la oración (sujeto, verbo, objeto, etc.)\n',
                '- **LEMMA**: La forma canónica de la palabra (ej: "corriendo" → "correr")\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Para cada token de la tercera oración, muestra:\n',
                '1. El texto del token\n',
                '2. Su POS tag\n',
                '3. Su DEP tag\n',
                '4. Su lema'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'if len(sentences) >= 3:\n',
                '    tercera_oracion = sentences[2]\n',
                '    print("Análisis detallado de tokens:")\n',
                '    print("-" * 80)\n',
                '    print(f"{\\"Token\\":<15} {\\"POS Tag\\":<12} {\\"DEP Tag\\":<12} {\\"Lemma\\":<15}")\n',
                '    print("-" * 80)\n',
                '    for token in tercera_oracion:\n',
                '        print(f"{token.text:<15} {token.pos_:<12} {token.dep_:<12} {token.lemma_:<15}")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 🏷️ Paso 9: Extracción de SINTAGMAS NOMINALES (Noun Chunks)\n',
                '\n',
                '### ¿Qué es un sintagma nominal?\n',
                'Un sintagma nominal es un grupo de palabras que funciona como sustantivo.\n',
                'Ejemplo: "El viejo corazón rojo" es un sintagma nominal.\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Extrae todos los sintagmas nominales del documento y muestra los primeros 15.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'noun_chunks = list(doc.noun_chunks)\n',
                'print(f"📊 Total de sintagmas nominales: {len(noun_chunks)}")\n',
                'print(f"\\nPrimeros 15 sintagmas nominales:")\n',
                'for i, chunk in enumerate(noun_chunks[:15], 1):\n',
                '    print(f"{i}. {chunk.text}")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 🔤 Paso 10: Extracción de VERBOS\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Busca todos los verbos (tokens con POS tag = VERB) en el documento.\n',
                'Muestra los verbos únicos (por su lema) para evitar duplicados como "habla", "hablaba", "hablaban".'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'verbos = [token for token in doc if token.pos_ == "VERB"]\n',
                'verbos_unicos = sorted(set([v.lemma_ for v in verbos]))\n',
                '\n',
                'print(f"📊 Total de verbos: {len(verbos)}")\n',
                'print(f"Verbos únicos: {len(verbos_unicos)}")\n',
                'print(f"\\nPrimeros 20 verbos encontrados:")\n',
                'for i, verbo in enumerate(verbos_unicos[:20], 1):\n',
                '    print(f"{i}. {verbo}")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 🏢 Paso 11: RECONOCIMIENTO DE ENTIDADES NOMBRADAS (NER)\n',
                '\n',
                '### ¿Qué es una entidad nombrada?\n',
                'Una entidad nombrada es un nombre específico de:\n',
                '- **PERSON**: Personas\n',
                '- **ORG**: Organizaciones\n',
                '- **GPE**: Lugares geográficos/políticos\n',
                '- **DATE**: Fechas\n',
                '- Y otras categorías...\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Identifica todas las entidades nombradas en el texto y las agrupa por tipo.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'print("🏷️ Entidades Nombradas encontradas:")\n',
                'entidades_dict = {}\n',
                'for ent in doc.ents:\n',
                '    if ent.label_ not in entidades_dict:\n',
                '        entidades_dict[ent.label_] = []\n',
                '    if ent.text not in entidades_dict[ent.label_]:\n',
                '        entidades_dict[ent.label_].append(ent.text)\n',
                '\n',
                'for label, entities in sorted(entidades_dict.items()):\n',
                '    print(f"\\n{label}:")\n',
                '    for ent in entities[:5]:\n',
                '        print(f"  - {ent}")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 🔎 Paso 12: BÚSQUEDA DE PATRONES con Matcher\n',
                '\n',
                '### ¿Qué es un Matcher?\n',
                'El Matcher es una herramienta que busca patrones específicos de tokens en el texto.\n',
                'Por ejemplo, puedes buscar: "Verbo seguido de Adverbio" para encontrar acciones con adverbios modificadores.\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Busca todos los patrones donde un VERBO es seguido por un ADVERBIO (actividades con intensidad).\n',
                'Ejemplo: "habla rápidamente", "actúa cautelosamente", etc.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'matcher = Matcher(nlp.vocab)\n',
                'pattern = [{\"POS\": \"VERB\"}, {\"POS\": \"ADV\"}]\n',
                'matcher.add("Vigorous_Activities", [pattern])\n',
                'matches = matcher(doc)\n',
                '\n',
                'print(f"🔎 Patrones encontrados (VERBO + ADVERBIO): {len(matches)}")\n',
                'print(f"\\nPrimeros 5 matches:")\n',
                'for i, (match_id, start, end) in enumerate(matches[:5], 1):\n',
                '    print(f"{i}. {doc[start:end].text}")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 📊 Paso 13: ESTADÍSTICAS DE PALABRAS\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Identifica las palabras más frecuentes en el documento.\n',
                'Excluye "stop words" (palabras comunes como "el", "la", "de", etc.) para mostrar palabras significativas.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'palabra_freq = Counter()\n',
                'for token in doc:\n',
                '    if not token.is_stop and token.is_alpha:\n',
                '        palabra_freq[token.lemma_] += 1\n',
                '\n',
                'print("📊 20 palabras más frecuentes (sin stop words):")\n',
                'print("-" * 40)\n',
                'for palabra, freq in palabra_freq.most_common(20):\n',
                '    print(f"{palabra:<20} {freq:>5} veces")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 📈 Paso 14: RESUMEN EJECUTIVO\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Muestra un resumen de todos los análisis realizados en un formato fácil de leer.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'print("=" * 80)\n',
                'print("RESUMEN EJECUTIVO DEL ANÁLISIS NLP")\n',
                'print("=" * 80)\n',
                'print(f"\\n📊 ESTADÍSTICAS GENERALES:")\n',
                'print(f"   • Total de tokens: {total_tokens}")\n',
                'print(f"   • Total de oraciones: {total_sentences}")\n',
                'print(f"   • Sintagmas nominales: {len(noun_chunks)}")\n',
                'print(f"   • Verbos: {len(verbos)}")\n',
                'print(f"   • Verbos únicos: {len(verbos_unicos)}")\n',
                'print(f"   • Entidades nombradas: {len(doc.ents)}")\n',
                'print(f"   • Patrones VERBO+ADVERBIO: {len(matches)}")\n',
                'print(f"\\n📝 LONGITUD PROMEDIO:")\n',
                'print(f"   • Tokens por oración: {total_tokens / total_sentences:.2f}")\n',
                'print(f"   • Caracteres: {len(texto)}")\n',
                'print("\\n✓ Análisis completado exitosamente")'
            ]
        }
    ],
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.8.0'}
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

# Guardar notebook
import os
os.chdir(r'C:\Users\bebes\Documents\MIAA\3. SEMESTRE\2. NLP TRANSFORMES\icesi_NLP_unidad1')

with open('1_NLP_spacy_ElCorazonDelator.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook1, f, ensure_ascii=False, indent=1)

print('✓ Notebook 1 mejorado y guardado')
