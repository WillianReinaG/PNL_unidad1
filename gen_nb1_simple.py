import json

# Notebook 1 SIMPLIFICADO: spaCy NLP
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
                'Este notebook realiza un análisis completo de procesamiento de lenguaje natural (NLP) en español utilizando spaCy.'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '**Instalación de dependencias:** Instala spaCy y descarga el modelo pre-entrenado para español que incluye tokenizador, analizador de dependencias y reconocedor de entidades.'
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
                'print("✓ spaCy instalado")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '**Importar librerías:** Cargamos spaCy, el Matcher para búsqueda de patrones, pandas y Counter para análisis de frecuencias.'
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
                'nlp = spacy.load("es_core_news_sm")\n',
                'print("✓ Modelo cargado")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '**Cargar el texto:** Lee el archivo de texto "El Corazón Delator" que será analizado.'
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
                'print(f"Texto cargado: {len(texto)} caracteres")\n',
                'print(f"\\nPrimer párrafo:\\n{texto[:200]}...")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '**Procesar el texto:** spaCy analiza el texto completo aplicando tokenización, etiquetado de partes del lenguaje, análisis de dependencias y reconocimiento de entidades.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'doc = nlp(texto)\n',
                'print("✓ Texto procesado")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '**Contar tokens:** Un token es cada palabra, número o símbolo de puntuación. Aquí contamos el total de tokens.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'total_tokens = len(doc)\n',
                'print(f"Total de tokens: {total_tokens}")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '**Análisis de oraciones:** Identificamos todas las oraciones en el documento y mostramos las primeras como ejemplo.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'sentences = list(doc.sents)\n',
                'print(f"Total de oraciones: {len(sentences)}")\n',
                'print(f"\\nPrimeras 3 oraciones:")\n',
                'for i, sent in enumerate(sentences[:3], 1):\n',
                '    print(f"{i}. {sent.text[:80]}...")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '**Tercera oración:** Extraemos la tercera oración del documento.'
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
                '    print(f"Tercera oración:\\n{tercera_oracion.text}")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '**Análisis detallado de tokens:** Para cada token en la tercera oración, mostramos su texto, POS tag (parte del lenguaje), DEP tag (relación sintáctica) y lemma (forma canónica).'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'if len(sentences) >= 3:\n',
                '    print("Token\\tPOS\\tDEP\\tLemma")\n',
                '    print("-" * 60)\n',
                '    for token in tercera_oracion:\n',
                '        print(f"{token.text}\\t{token.pos_}\\t{token.dep_}\\t{token.lemma_}")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '**Sintagmas nominales:** Extrae grupos de palabras que funcionan como sustantivos (ej: "El viejo corazón").'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'noun_chunks = list(doc.noun_chunks)\n',
                'print(f"Total de sintagmas nominales: {len(noun_chunks)}")\n',
                'print(f"\\nPrimeros 15:")\n',
                'for i, chunk in enumerate(noun_chunks[:15], 1):\n',
                '    print(f"{i}. {chunk.text}")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '**Verbos:** Busca todos los verbos en el documento y muestra los únicos (agrupados por lema para evitar duplicados).'
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
                'print(f"Total de verbos: {len(verbos)}")\n',
                'print(f"Verbos únicos: {len(verbos_unicos)}")\n',
                'print(f"\\nPrimeros 20 verbos:")\n',
                'for i, verbo in enumerate(verbos_unicos[:20], 1):\n',
                '    print(f"{i}. {verbo}")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '**Reconocimiento de entidades:** Identifica nombres específicos como personas, lugares, eventos, fechas, etc.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'print("Entidades encontradas:")\n',
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
                '**Búsqueda de patrones:** Usa el Matcher para encontrar patrones específicos: un VERBO seguido de un ADVERBIO (actividades con intensidad).'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'matcher = Matcher(nlp.vocab)\n',
                'pattern = [{"POS": "VERB"}, {"POS": "ADV"}]\n',
                'matcher.add("Vigorous_Activities", [pattern])\n',
                'matches = matcher(doc)\n',
                '\n',
                'print(f"Patrones encontrados: {len(matches)}")\n',
                'print(f"\\nPrimeros 5:")\n',
                'for i, (match_id, start, end) in enumerate(matches[:5], 1):\n',
                '    print(f"{i}. {doc[start:end].text}")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '**Palabras más frecuentes:** Identifica las palabras con mayor frecuencia, excluyendo stop words (palabras comunes como "el", "de", etc.).'
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
                'print("20 palabras más frecuentes:")\n',
                'print("-" * 40)\n',
                'for palabra, freq in palabra_freq.most_common(20):\n',
                '    print(f"{palabra:<20} {freq:>5} veces")'
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

import os
os.chdir(r'C:\Users\bebes\Documents\MIAA\3. SEMESTRE\2. NLP TRANSFORMES\icesi_NLP_unidad1')

with open('1_NLP_spacy_ElCorazonDelator.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook1, f, ensure_ascii=False, indent=1)

print('✓ Notebook 1 simplificado')
