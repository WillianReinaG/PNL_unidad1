# ICESI NLP Unidad 1 - Procesamiento de Lenguaje Natural

Proyecto académico para el análisis de Procesamiento de Lenguaje Natural (NLP) usando Python, spaCy y NLTK.

## 📚 Descripción

Este proyecto contiene dos notebooks Jupyter que demuestran técnicas fundamentales de Procesamiento de Lenguaje Natural:

### 1. **Análisis NLP con spaCy** (`1_NLP_spacy_ElCorazonDelator.ipynb`)
Análisis completo del texto "El Corazón Delator" de Edgar Allan Poe usando spaCy.

**Características:**
- ✅ Tokenización de texto en español
- ✅ Análisis de sintagmas nominales (noun chunks)
- ✅ Extracción de verbos
- ✅ Reconocimiento de Entidades Nombradas (NER)
- ✅ Conteo de tokens y oraciones
- ✅ Análisis detallado de tokens (POS tags, dependency parsing, lemmatización)
- ✅ Uso de matchers para búsqueda de patrones
- ✅ Estadísticas de frecuencia de palabras

**Tecnologías:**
- Python 3.8+
- spaCy 3.x
- Modelo: `es_core_news_sm`

### 2. **Análisis de Sentimientos con NLTK VADER** (`2_Analisis_Sentimientos_VADER.ipynb`)
Análisis de sentimientos en reseñas de productos de Amazon usando NLTK con lexicon VADER.

**Características:**
- ✅ Limpieza y preprocesamiento de texto
- ✅ Análisis de sentimientos con VADER
- ✅ Conversión de scores a predicciones binarias
- ✅ Cálculo de métricas de calidad (Accuracy, Precision, Recall, F1-Score)
- ✅ Matriz de confusión
- ✅ Visualizaciones gráficas
- ✅ Análisis de casos mal clasificados
- ✅ Análisis de palabras más influyentes
- ✅ Análisis de confiabilidad por rangos de scores

**Tecnologías:**
- Python 3.8+
- NLTK 3.x
- scikit-learn
- pandas, numpy, matplotlib, seaborn

## 📁 Estructura del Proyecto

```
icesi_NLP_unidad1/
├── 1_NLP_spacy_ElCorazonDelator.ipynb        # Notebook de análisis NLP
├── 2_Analisis_Sentimientos_VADER.ipynb       # Notebook de análisis de sentimientos
├── El_corazón_delator.txt                    # Texto fuente (español)
├── amazon_cells_labelled.txt                 # Dataset de reseñas
├── requirements.txt                          # Dependencias de Python
├── README.md                                  # Este archivo
└── .gitignore                                 # Archivos a ignorar en git
```

## 🚀 Inicio Rápido

### Opción 1: Ejecutar Localmente

#### Requisitos:
- Python 3.8 o superior
- pip o conda

#### Instalación:

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/icesi_NLP_unidad1.git
cd icesi_NLP_unidad1

# Crear un entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Iniciar Jupyter
jupyter notebook
```

#### Ejecutar los notebooks:
1. Abre `1_NLP_spacy_ElCorazonDelator.ipynb`
2. Ejecuta todas las celdas (Kernel → Run All)
3. Repite para el segundo notebook

### Opción 2: Ejecutar en Google Colab

1. Sube los archivos `.ipynb` a Google Drive
2. Haz clic derecho → Abrir con → Google Colaboratory
3. Los notebooks descargarán automáticamente los modelos necesarios
4. Carga los archivos `.txt` cuando se solicite

## 📊 Resultados Esperados

### Notebook 1 (spaCy):
- Total de tokens: ~2,400
- Total de oraciones: ~45
- Entidades nombradas reconocidas: Personas, eventos históricos
- Análisis detallado con POS tags, lemmas, y dependency parsing

### Notebook 2 (VADER):
- Análisis de ~1,000 reseñas de Amazon
- Matriz de confusión y métricas de clasificación
- Visualizaciones de distribución de sentimientos
- Palabras más influyentes en sentimientos positivos/negativos

## 🔧 Dependencias

Ver `requirements.txt` para la lista completa de paquetes.

**Principales:**
- `spacy` - Procesamiento de lenguaje natural
- `nltk` - Natural Language Toolkit
- `scikit-learn` - Machine Learning
- `pandas` - Análisis de datos
- `matplotlib` / `seaborn` - Visualización
- `jupyter` - Notebooks interactivos

## 📝 Notas Importantes

### Para el Notebook 1 (spaCy):
- El modelo de español se descargará automáticamente
- El primer ejecutable puede demorar unos minutos

### Para el Notebook 2 (VADER):
- VADER está optimizado para análisis de sentimientos en redes sociales
- Los thresholds de clasificación pueden ajustarse según el caso de uso

### Compatibilidad con Google Colab:
- Ambos notebooks incluyen código para cargar archivos desde Drive
- Las instalaciones se manejan automáticamente

## 📚 Conceptos Clave

### Procesamiento de Lenguaje Natural (NLP):
- **Tokenización**: División del texto en palabras/tokens
- **POS Tagging**: Etiquetado de partes del lenguaje (sustantivo, verbo, etc.)
- **Lemmatización**: Reducción de palabras a su forma canónica
- **NER**: Reconocimiento de entidades nombradas (personas, lugares, etc.)
- **Dependency Parsing**: Análisis sintáctico

### Análisis de Sentimientos:
- **VADER**: Lexicon-based sentiment analyzer optimizado para redes sociales
- **Compound Score**: Puntuación normalizada de -1 (muy negativo) a +1 (muy positivo)
- **Métricas**: Accuracy, Precision, Recall, F1-Score

## 🎓 Recursos Educativos

- [Documentación de spaCy](https://spacy.io/)
- [Documentación de NLTK](https://www.nltk.org/)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)

## 👨‍💻 Autores

- **Juan Manuel Hurtado Angulo**
- **Manuel Alberto González González**
- **Willian Alberto Reina García**

## 🎓 Información Académica

**Asignatura:** Procesamiento de Lenguaje Natural

**Tutor:** Luis Ferro Díez

**Institución:** ICESI

## 📄 Licencia

Este proyecto está bajo licencia MIT. Ver archivo LICENSE para más detalles.

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor, abre un issue o un pull request.

---

**Nota:** Este proyecto ha sido diseñado para ser compatible con Google Colab, permitiendo su ejecución en línea sin necesidad de instalar dependencias localmente.
