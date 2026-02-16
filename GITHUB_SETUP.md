# Instrucciones para crear y sincronizar el proyecto en GitHub

## Paso 1: Crear un repositorio en GitHub

1. Ve a https://github.com/new
2. Completa los campos:
   - **Repository name:** `icesi_NLP_unidad1`
   - **Description:** Procesamiento de Lenguaje Natural con spaCy y NLTK - Proyecto ICESI
   - **Visibility:** Público (o Privado según preferencia)
   - NO inicialices el repositorio con README (ya lo tenemos)
   - NO inicialices con .gitignore (ya lo tenemos)

3. Haz clic en "Create repository"

## Paso 2: Conectar el repositorio local con GitHub

Una vez creado el repositorio en GitHub, verás instrucciones. Ejecuta en PowerShell (en la carpeta del proyecto):

```powershell
# Agregar el repositorio remoto
git remote add origin https://github.com/TU_USUARIO/icesi_NLP_unidad1.git

# Cambiar la rama a 'main' (GitHub utiliza 'main' por defecto)
git branch -M main

# Empujar los cambios
git push -u origin main
```

**Nota:** Reemplaza `TU_USUARIO` con tu usuario de GitHub

## Paso 3: Verificar la sincronización

```powershell
# Verificar que todo se subió correctamente
git remote -v
git log --oneline
```

Deberías ver:
```
commit 3b1ffa2 - Commit inicial: Notebooks NLP y análisis de sentimientos con documentación
```

## Pasos rápidos completos:

```powershell
# En la carpeta del proyecto
cd "C:\Users\bebes\Documents\MIAA\3. SEMESTRE\2. NLP TRANSFORMES\icesi_NLP_unidad1"

# Confirmar que el repositorio local existe
git status

# Agregar el remoto (reemplazar TU_USUARIO)
git remote add origin https://github.com/TU_USUARIO/icesi_NLP_unidad1.git

# Cambiar a main
git branch -M main

# Empujar
git push -u origin main
```

## Si ya tenías configurado el origen remoto:

```powershell
git remote remove origin
git remote add origin https://github.com/TU_USUARIO/icesi_NLP_unidad1.git
git push -u origin main
```

## Clonar el repositorio en otra máquina:

Una vez esté en GitHub, puedes clonarlo con:

```powershell
git clone https://github.com/TU_USUARIO/icesi_NLP_unidad1.git
cd icesi_NLP_unidad1
pip install -r requirements.txt
```

## Para futuros cambios:

```powershell
# Realizar cambios en los archivos

# Agregar cambios
git add .

# Hacer commit
git commit -m "Descripción de cambios"

# Empujar a GitHub
git push origin main
```

---

**Estructura del proyecto subida:**
✅ 1_NLP_spacy_ElCorazonDelator.ipynb
✅ 2_Analisis_Sentimientos_VADER.ipynb  
✅ El_corazón_delator.txt
✅ amazon_cells_labelled.txt
✅ requirements.txt
✅ README.md
✅ .gitignore

¡El proyecto está listo para sincronizar con GitHub!
