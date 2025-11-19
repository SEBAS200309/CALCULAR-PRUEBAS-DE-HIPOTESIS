# 📌 Calculadora Estadística – Tests de Hipótesis, Correlación y Regresión Lineal  
**Versión rama: `pruebas`**

Este proyecto es una aplicación interactiva desarrollada en **Streamlit** que permite realizar análisis estadístico a partir de archivos CSV, incluyendo:

- Estadísticas básicas  
- Pruebas de hipótesis para la media (Z y T)  
- Correlación de Pearson  
- Regresión lineal por mínimos cuadrados  
- Test de significancia de la pendiente  
- Visualización de distribuciones (normal / t-student)

Es ideal para clases de estadística y análisis de datos en contextos académicos.

---

# 📂 Estructura del Proyecto
/
├── streamlit_app.py # Interfaz principal Streamlit (rama pruebas)
├── funciones.py # Cálculos estadísticos, correlación y regresión
├── requirements.txt # Dependencias del proyecto
└── README.md
---
# 🧪 Diferencias entre ramas

## 🔹 Rama **main**
- Contiene la implementación original.
- Sin control de muestras dinámicas.
- Cálculos de Pearson y regresión más simples.
- Menos controles interactivos y sin visualizaciones de significancia.

## 🔹 Rama **pruebas** (esta)
Versión extendida y mejorada:

### ✔ Funcionalidades avanzadas
- Pruebas Z y T configurables  
- Hipótesis configurables (>, <, ≠)  
- p-valor calculado según la prueba seleccionada  
- Regresión lineal completa con:
  - Pendiente, intercepto  
  - r, r²  
  - Error estándar  
  - p-valor  
  - t-test para pendiente  
- Gráficos:
  - Regresión lineal
  - Distribución normal / t-student con zonas críticas
- Clasificación automática de la correlación

### ✔ Mejor organización
- Funciones separadas en `funciones.py`
- Mejor claridad en el código
- Flujo claro para análisis completos

---

# ▶️ Cómo ejecutar la aplicación

## 1️⃣ Crear un entorno virtual

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```
### Linux/Mac OS
```bash
python3 -m venv venv
source venv/bin/activate
```

## 2️⃣ Instalar las dependencias
```bash
pip install -r requirements.txt
```
##3️⃣ Ejecutar la app
```bash
streamlit run streamlit_app.py
```
# 🔧 Funcionalidades principales
## ✔ 1. Carga y exploración de datos

* Carga de CSV

* Exploración de columnas

* Vista previa con head()

## ✔ 2. Estadísticas descriptivas

Incluye:

- Media

- Desviación estándar (muestral y poblacional)

- Tamaño de la muestra

- Distribuciones

## ✔ 3. Pruebas de hipótesis

Incluye:

### Test Z

- σ conocido

- Z calculado

- p-valor

- Conclusión automática

### Test T

- σ desconocido

- t calculado

- p-valor

- Gráfica t con zonas críticas

## ✔ 4. Regresión lineal

* Selección de tamaño de muestra (slider)

* Cálculo de:

    - Pendiente

    - Intercepto

    - r


    - p-valor de la pendiente

- Gráfico de regresión

- Test de significancia de la pendiente

- Gráfica de distribución t con zona de rechazo

# 🐋 Sobre Docker y Codespaces

Aunque la estructura está pensada para ejecutarse en contenedores, esta versión no incluye aún el Dockerfile ni la configuración devcontainer, debido a:

* Limitaciones de almacenamiento en Codespaces

* Necesidad de probar Streamlit en entorno remoto

* La prioridad actual es la estabilidad de la versión local

📜 Licencia

Este proyecto está liberado bajo la licencia Creative Commons Zero 1.0 Universal (CC0 1.0).

Esto significa:

* Puedes copiar, modificar, distribuir y usar el proyecto sin restricciones.

* Puedes usarlo para cualquier propósito, incluso comercial.

* No es obligatorio dar atribución (aunque siempre es bienvenida).

# 🙌 Autor

Desarrollado por Sebastián con soporte técnico y estructural generado por ChatGPT.