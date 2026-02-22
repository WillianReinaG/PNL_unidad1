# Registro de Cambios - PNL Unidad 1

## Cambios Recientes

### 2026-02-22
- ✅ **Corregida carga de archivos en ambos notebooks**
  - Los notebooks ahora descargan archivos directamente desde GitHub
  - Compatible con Google Colab
  - URL de descarga: `https://raw.githubusercontent.com/WillianReinaG/PNL_unidad1/main/`
  
- **Notebook 1 (spaCy)**: Celda 7 - Carga desde GitHub
  - Usa `urllib.request` para descargar `El_corazón_delator.txt`
  - Fallback: intenta cargar localmente si es necesario
  
- **Notebook 2 (VADER)**: Celda 7 - Carga desde GitHub
  - Usa `urllib.request` y `StringIO` para descargar `amazon_cells_labelled.txt`
  - Fallback: intenta cargar localmente si es necesario

### Compatibilidad
- ✅ Funciona en máquina local
- ✅ Funciona en Google Colab
- ✅ Sin dependencia de archivos locales en Colab

