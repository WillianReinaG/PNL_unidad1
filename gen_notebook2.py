import json

# Notebook 2 mejorado: VADER Sentiment Analysis
notebook2 = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# Análisis de Sentimientos con NLTK VADER\n',
                '## Análisis de reseñas de productos de Amazon\n',
                '\n',
                '[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WillianReinaG/PNL_unidad1/blob/main/2_Analisis_Sentimientos_VADER.ipynb)\n',
                '\n',
                'Este notebook realiza un **análisis completo de sentimientos** utilizando **NLTK con VADER (Valence Aware Dictionary and sEntiment Reasoner)**.\n',
                '\n',
                '**Contenidos:**\n',
                '- Carga y exploración de datos\n',
                '- Limpieza y preprocesamiento de texto\n',
                '- Análisis de sentimientos con VADER\n',
                '- Conversión de scores a predicciones binarias\n',
                '- Cálculo de métricas de evaluación\n',
                '- Visualización de resultados\n',
                '- Análisis de casos mal clasificados\n',
                '- Palabras más influyentes por sentimiento'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 📦 Paso 1: Instalación de dependencias\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Instala las librerías necesarias:\n',
                '- **nltk**: Natural Language Toolkit con VADER\n',
                '- **scikit-learn**: Para métricas de evaluación\n',
                '- **matplotlib/seaborn**: Para visualizaciones\n',
                '- **pandas/numpy**: Para manipulación de datos\n',
                '\n',
                'También descarga recursos de NLTK como el lexicón VADER.\n',
                '\n',
                '**Nota:** Esto puede tomar 1-2 minutos en la primera ejecución.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '!pip install nltk scikit-learn matplotlib seaborn pandas numpy -q\n',
                '\n',
                'import nltk\n',
                'nltk.download("vader_lexicon", quiet=True)\n',
                'nltk.download("punkt", quiet=True)\n',
                'nltk.download("averaged_perceptron_tagger", quiet=True)\n',
                '\n',
                'print("✓ Todas las dependencias instaladas")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 📚 Paso 2: Importar librerías necesarias\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Carga e inicializa las librerías:\n',
                '- **SentimentIntensityAnalyzer**: Motor de análisis VADER\n',
                '- **word_tokenize**: Para dividir texto en palabras\n',
                '- **Métricas de sklearn**: Para evaluar el modelo\n',
                '- **matplotlib/seaborn**: Para crear gráficos\n',
                '- **Counter**: Para contar frecuencias'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'import pandas as pd\n',
                'import numpy as np\n',
                'from nltk.sentiment.vader import SentimentIntensityAnalyzer\n',
                'from nltk.tokenize import word_tokenize\n',
                'from sklearn.metrics import (\n',
                '    confusion_matrix, classification_report, accuracy_score,\n',
                '    precision_score, recall_score, f1_score\n',
                ')\n',
                'import matplotlib.pyplot as plt\n',
                'import seaborn as sns\n',
                'import re\n',
                'from collections import Counter\n',
                '\n',
                '# Inicializar VADER\n',
                'sia = SentimentIntensityAnalyzer()\n',
                '\n',
                'print("✓ Librerías importadas correctamente")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 📊 Paso 3: Cargar y explorar los datos\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Lee el archivo `amazon_cells_labelled.txt` que contiene ~1,000 reseñas de productos de Amazon.\n',
                'Cada línea tiene:\n',
                '- La reseña (texto)\n',
                '- La etiqueta de sentimiento (0=negativo, 1=positivo)\n',
                '\n',
                '### Formato del archivo:\n',
                '```\n',
                'Good case, Excellent value.\\t1\n',
                'Tied to charger for conversations.\\t0\n',
                '```'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'df = pd.read_csv("amazon_cells_labelled.txt", sep="\\t", header=None, names=["review", "sentiment"])\n',
                '\n',
                'print(f"✓ Datos cargados: {len(df)} reseñas")\n',
                'print(f"\\n📊 Distribución de sentimientos:")\n',
                'print(df["sentiment"].value_counts())\n',
                'print(f"\\nPorcentaje: {(df[\\"sentiment\\"] == 1).sum() / len(df) * 100:.2f}% positivas")\n',
                'print(f"\\nPrimeras 3 reseñas:")\n',
                'for i in range(3):\n',
                '    print(f"{i+1}. [{df[\\"sentiment\\"].iloc[i]}] {df[\\"review\\"].iloc[i][:70]}...")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 🧹 Paso 4: Limpieza y preprocesamiento de texto\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Aplica transformaciones al texto para mejorar el análisis:\n',
                '- Convertir a minúsculas (normalización)\n',
                '- Remover URLs\n',
                '- Remover caracteres especiales\n',
                '- Remover espacios en blanco múltiples\n',
                '\n',
                '### ¿Por qué es importante?\n',
                'El texto limpio mejora la precisión del análisis de sentimientos.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'def clean_text(text):\n',
                '    """\n',
                '    Limpia el texto para análisis de sentimientos\n',
                '    """\n',
                '    # Convertir a minúsculas\n',
                '    text = text.lower()\n',
                '    # Remover URLs\n',
                '    text = re.sub(r"http\\\\S+|www\\\\S+", "", text)\n',
                '    # Remover caracteres especiales pero mantener puntuación importante\n',
                '    text = re.sub(r"[^a-zA-Z0-9\\\\s.!?,-]", "", text)\n',
                '    # Remover espacios en blanco múltiples\n',
                '    text = re.sub(r"\\\\s+", " ", text).strip()\n',
                '    return text\n',
                '\n',
                'df["review_cleaned"] = df["review"].apply(clean_text)\n',
                'df = df[df["review_cleaned"].str.len() > 0]\n',
                '\n',
                'print(f"✓ Limpieza completada")\n',
                'print(f"Reseñas después de limpieza: {len(df)}")\n',
                'print(f"\\nEjemplo de limpieza:")\n',
                'print(f"Original: {df[\\"review\\"].iloc[0]}")\n',
                'print(f"Limpia:   {df[\\"review_cleaned\\"].iloc[0]}")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 😊😞 Paso 5: Análisis de sentimientos con VADER\n',
                '\n',
                '### ¿Qué es VADER?\n',
                '**VADER** (Valence Aware Dictionary and sEntiment Reasoner) es un analizador de sentimientos:\n',
                '- Basado en **léxicon** (diccionario de palabras con sentimientos)\n',
                '- Optimizado para redes sociales y textos cortos\n',
                '- Proporciona **4 scores**:\n',
                '  - **pos**: Proporción de sentimiento positivo (0-1)\n',
                '  - **neg**: Proporción de sentimiento negativo (0-1)\n',
                '  - **neu**: Proporción de sentimiento neutral (0-1)\n',
                '  - **compound**: Score normalizado (-1 a +1) → **Este es el que usamos**\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Aplica VADER a todas las reseñas y calcula los scores.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'print("Analizando sentimientos con VADER...")\n',
                'sentiment_scores = df["review_cleaned"].apply(lambda x: sia.polarity_scores(x))\n',
                '\n',
                '# Expandir los scores en columnas\n',
                'sentiment_df = pd.DataFrame(sentiment_scores.tolist())\n',
                'df = pd.concat([df, sentiment_df], axis=1)\n',
                '\n',
                '# Renombrar columnas\n',
                'df = df.rename(columns={\n',
                '    "neg": "vader_negative",\n',
                '    "neu": "vader_neutral",\n',
                '    "pos": "vader_positive",\n',
                '    "compound": "vader_compound"\n',
                '})\n',
                '\n',
                'print("✓ Análisis completado")\n',
                'print(f"\\n📊 Estadísticas del compound score:")\n',
                'print(f"  Mínimo: {df[\\"vader_compound\\"].min():.4f}")\n',
                'print(f"  Máximo: {df[\\"vader_compound\\"].max():.4f}")\n',
                'print(f"  Promedio: {df[\\"vader_compound\\"].mean():.4f}")\n',
                '\n',
                'print(f"\\nPrimeras 5 reseñas con scores:")\n',
                'for i in range(5):\n',
                '    print(f"{df[\\"review\\"].iloc[i][:50]}... | Score: {df[\\"vader_compound\\"].iloc[i]:.3f}")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 🏷️ Paso 6: Conversión de scores a predicciones\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Convierte los **compound scores continuos** (-1 a +1) a **predicciones binarias** (0 o 1):\n',
                '- Si compound >= 0.05 → **Predicción: 1 (Positivo)**\n',
                '- Si compound < 0.05 → **Predicción: 0 (Negativo)**\n',
                '\n',
                '### ¿Por qué 0.05?\n',
                '0.05 es un threshold estándar que equilibra falsos positivos y falsos negativos.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'df["vader_prediction"] = df["vader_compound"].apply(lambda x: 1 if x >= 0.05 else 0)\n',
                '\n',
                'print("✓ Predicciones creadas")\n',
                'print(f"\\n📊 Distribución de predicciones:")\n',
                'print(df["vader_prediction"].value_counts())\n',
                'print(f"\\nPositivas predichas: {(df[\\"vader_prediction\\"] == 1).sum() / len(df) * 100:.2f}%")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 📈 Paso 7: Cálculo de MÉTRICAS DE CALIDAD\n',
                '\n',
                '### ¿Qué son estas métricas?\n',
                '- **Accuracy**: ¿Qué porcentaje de predicciones fue correcto?\n',
                '- **Precision**: De las predicciones positivas, ¿cuántas fueron correctas?\n',
                '- **Recall**: De los casos positivos reales, ¿cuántos detectamos?\n',
                '- **F1-Score**: Promedio armónico de Precision y Recall\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Calcula todas estas métricas comparando predicciones con valores reales.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'y_true = df["sentiment"].values\n',
                'y_pred = df["vader_prediction"].values\n',
                '\n',
                'accuracy = accuracy_score(y_true, y_pred)\n',
                'precision = precision_score(y_true, y_pred, zero_division=0)\n',
                'recall = recall_score(y_true, y_pred, zero_division=0)\n',
                'f1 = f1_score(y_true, y_pred, zero_division=0)\n',
                '\n',
                'print("=" * 60)\n',
                'print("MÉTRICAS DE CALIDAD DEL MODELO VADER")\n',
                'print("=" * 60)\n',
                'print(f"\\n✓ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")\n',
                'print(f"✓ Precision: {precision:.4f}")\n',
                'print(f"✓ Recall:    {recall:.4f}")\n',
                'print(f"✓ F1-Score:  {f1:.4f}")\n',
                'print("\\n" + "=" * 60)'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 🔲 Paso 8: MATRIZ DE CONFUSIÓN\n',
                '\n',
                '### ¿Qué es una matriz de confusión?\n',
                'Compara las predicciones con los valores reales:\n',
                '- **VP (Verdaderos Positivos)**: Predijo positivo y era positivo\n',
                '- **FN (Falsos Negativos)**: Predijo negativo pero era positivo\n',
                '- **FP (Falsos Positivos)**: Predijo positivo pero era negativo\n',
                '- **VN (Verdaderos Negativos)**: Predijo negativo y era negativo\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Crea y visualiza la matriz de confusión.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'cm = confusion_matrix(y_true, y_pred)\n',
                '\n',
                'print("🔲 Matriz de Confusión:")\n',
                'print(f"\\n{\\"\\\":<20} Predicción Neg  Predicción Pos")\n',
                'print(f"Real Negativo:      {cm[0,0]:>6}           {cm[0,1]:>6}")\n',
                'print(f"Real Positivo:      {cm[1,0]:>6}           {cm[1,1]:>6}")\n',
                '\n',
                'print("\\nReporte detallado:")\n',
                'print(classification_report(y_true, y_pred, target_names=["Negativo", "Positivo"]))'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 📊 Paso 9: VISUALIZACIÓN - Matriz de Confusión\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Crea un gráfico de calor (heatmap) para visualizar la matriz de confusión de forma gráfica.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'plt.figure(figsize=(8, 6))\n',
                'sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",\n',
                '            xticklabels=["Negativo", "Positivo"],\n',
                '            yticklabels=["Negativo", "Positivo"],\n',
                '            cbar_kws={"label": "Cantidad"})\n',
                'plt.title("Matriz de Confusión - Análisis de Sentimientos VADER", fontsize=14, fontweight="bold")\n',
                'plt.ylabel("Sentimiento Real")\n',
                'plt.xlabel("Sentimiento Predicho")\n',
                'plt.tight_layout()\n',
                'plt.show()'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 📉 Paso 10: VISUALIZACIÓN - Distribución de Scores\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Crea dos gráficos:\n',
                '1. **Histograma**: Distribución de todos los compound scores\n',
                '2. **Box plot**: Comparación de scores por sentimiento real'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n',
                '\n',
                '# Histograma\n',
                'axes[0].hist(df["vader_compound"], bins=50, edgecolor="black", alpha=0.7, color="skyblue")\n',
                'axes[0].axvline(x=0.05, color="green", linestyle="--", linewidth=2, label="Threshold")\n',
                'axes[0].set_xlabel("Compound Score VADER")\n',
                'axes[0].set_ylabel("Frecuencia")\n',
                'axes[0].set_title("Distribución de Compound Scores")\n',
                'axes[0].legend()\n',
                'axes[0].grid(True, alpha=0.3)\n',
                '\n',
                '# Box plot\n',
                'df.boxplot(column="vader_compound", by="sentiment", ax=axes[1])\n',
                'axes[1].set_xlabel("Sentimiento Real (0=Negativo, 1=Positivo)")\n',
                'axes[1].set_ylabel("Compound Score VADER")\n',
                'axes[1].set_title("Compound Score por Sentimiento Real")\n',
                'plt.suptitle("")\n',
                'plt.tight_layout()\n',
                'plt.show()'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 🔤 Paso 11: ANÁLISIS DE PALABRAS INFLUYENTES\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Identifica las palabras más frecuentes en reseñas positivas y negativas.\n',
                'Esto ayuda a entender qué palabras VADER asocia con cada sentimiento.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'positive_reviews = df[df["sentiment"] == 1]["review_cleaned"].str.cat(sep=" ")\n',
                'negative_reviews = df[df["sentiment"] == 0]["review_cleaned"].str.cat(sep=" ")\n',
                '\n',
                'positive_words = word_tokenize(positive_reviews)\n',
                'negative_words = word_tokenize(negative_reviews)\n',
                '\n',
                'positive_freq = Counter([w for w in positive_words if len(w) > 2 and w.isalpha()])\n',
                'negative_freq = Counter([w for w in negative_words if len(w) > 2 and w.isalpha()])\n',
                '\n',
                'print("🎯 PALABRAS EN RESEÑAS POSITIVAS:")\n',
                'for word, freq in positive_freq.most_common(10):\n',
                '    print(f"  {word:<15} - {freq:>4} veces")\n',
                '\n',
                'print("\\n😞 PALABRAS EN RESEÑAS NEGATIVAS:")\n',
                'for word, freq in negative_freq.most_common(10):\n',
                '    print(f"  {word:<15} - {freq:>4} veces")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## ❌ Paso 12: ANÁLISIS DE CASOS MAL CLASIFICADOS\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Identifica y analiza:\n',
                '- **Falsos Positivos**: Sentimiento negativo predicho como positivo\n',
                '- **Falsos Negativos**: Sentimiento positivo predicho como negativo\n',
                '\n',
                'Esto nos ayuda a entender las limitaciones del modelo.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'df["correct"] = df["sentiment"] == df["vader_prediction"]\n',
                'misclassified = df[~df["correct"]]\n',
                'correctly_classified = df[df["correct"]]\n',
                '\n',
                'fp = misclassified[(misclassified["sentiment"] == 0) & (misclassified["vader_prediction"] == 1)]\n',
                'fn = misclassified[(misclassified["sentiment"] == 1) & (misclassified["vader_prediction"] == 0)]\n',
                '\n',
                'print("=" * 70)\n',
                'print("ANÁLISIS DE ERRORES")\n',
                'print("=" * 70)\n',
                'print(f"\\n✓ Correctamente clasificadas: {len(correctly_classified)} ({len(correctly_classified)/len(df)*100:.2f}%)")\n',
                'print(f"✗ Mal clasificadas: {len(misclassified)} ({len(misclassified)/len(df)*100:.2f}%)")\n',
                'print(f"  - Falsos Positivos: {len(fp)} ({len(fp)/len(df)*100:.2f}%)")\n',
                'print(f"  - Falsos Negativos: {len(fn)} ({len(fn)/len(df)*100:.2f}%)")\n',
                '\n',
                'print(f"\\n🔴 EJEMPLOS DE FALSOS POSITIVOS (Real Negativo, Predicho Positivo):")\n',
                'for idx in fp.head(2).index:\n',
                '    print(f"  • \\\"{df.loc[idx, \\"review\\"][:60]}...\\\"\\n    Score: {df.loc[idx, \\"vader_compound\\"]:.3f}")\n',
                '\n',
                'print(f"\\n🟢 EJEMPLOS DE FALSOS NEGATIVOS (Real Positivo, Predicho Negativo):")\n',
                'for idx in fn.head(2).index:\n',
                '    print(f"  • \\\"{df.loc[idx, \\"review\\"][:60]}...\\\"\\n    Score: {df.loc[idx, \\"vader_compound\\"]:.3f}")'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## 📊 Paso 13: RESUMEN EJECUTIVO\n',
                '\n',
                '### ¿Qué hace esta celda?\n',
                'Presenta un resumen completo del análisis con recomendaciones.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'print("\\n" + "=" * 80)\n',
                'print("RESUMEN EJECUTIVO - ANÁLISIS DE SENTIMIENTOS")\n',
                'print("=" * 80)\n',
                '\n',
                'print(f"\\n📊 ESTADÍSTICAS DEL DATASET")\n',
                'print(f"   • Total de reseñas: {len(df)}")\n',
                'print(f"   • Positivas: {(df[\\"sentiment\\"] == 1).sum()} ({(df[\\"sentiment\\"] == 1).sum()/len(df)*100:.2f}%)")\n',
                'print(f"   • Negativas: {(df[\\"sentiment\\"] == 0).sum()} ({(df[\\"sentiment\\"] == 0).sum()/len(df)*100:.2f}%)")\n',
                '\n',
                'print(f"\\n🎯 RENDIMIENTO DEL MODELO VADER")\n',
                'print(f"   • Accuracy:  {accuracy*100:.2f}%")\n',
                'print(f"   • Precision: {precision:.4f}")\n',
                'print(f"   • Recall:    {recall:.4f}")\n',
                'print(f"   • F1-Score:  {f1:.4f}")\n',
                '\n',
                'print(f"\\n📈 ANÁLISIS DE ERRORES")\n',
                'print(f"   • Predicciones correctas: {len(correctly_classified)} ({len(correctly_classified)/len(df)*100:.2f}%)")\n',
                'print(f"   • Predicciones incorrectas: {len(misclassified)} ({len(misclassified)/len(df)*100:.2f}%)")\n',
                '\n',
                'if accuracy >= 0.80:\n',
                '    print(f"\\n✓ CONCLUSIÓN: Excelente rendimiento de VADER en este dataset")\n',
                'elif accuracy >= 0.70:\n',
                '    print(f"\\n✓ CONCLUSIÓN: Buen rendimiento de VADER")\n',
                'else:\n',
                '    print(f"\\n⚠ CONCLUSIÓN: Considera usar modelos más sofisticados (Deep Learning)")\n',
                '\n',
                'print("\\n" + "=" * 80)'
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

with open('2_Analisis_Sentimientos_VADER.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook2, f, ensure_ascii=False, indent=1)

print('✓ Notebook 2 mejorado y guardado')
