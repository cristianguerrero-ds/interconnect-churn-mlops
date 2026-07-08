
# 📊 Interconnect Churn Predictor v3 — MLOps Platform

Plataforma analítica e interactiva de MLOps para la predicción, gestión y auditoría en tiempo real del riesgo de deserción de clientes (*Churn*) en servicios de telecomunicaciones.

---

## 💡 Descripción General

El **Sistema Analítico de Retención - Interconnect** integra un modelo de Machine Learning entrenado y desplegado en un entorno web interactivo construido con **Streamlit**. La plataforma permite a los equipos de analítica y operaciones comerciales evaluar el riesgo de deserción individual o masivo, activar estrategias de retención personalizadas y auditar los resultados comerciales con persistencia bidireccional en la nube y fallback local.

---

*## Enlace: [https://cristianguerrero-ds-interconnect-churn-mlops-app-ulibfz.streamlit.app/#procesamiento-masivo-por-lotes]*



![alt text](image-1.png)

![alt text](image-5.png)

![alt text](image-3.png)

![alt text](image-4.png)
---

## 🔥 Características Principales

* **🔎 Consulta Individual e Inferencia en Tiempo Real:** Evaluación puntual de clientes mediante formularios dinámicos, cálculo automático de probabilidad de abandono y generación de propuestas comerciales automatizadas.
* **📂 Procesamiento Masivo por Lotes (Batch Inference):** Carga de archivos CSV heterogéneos, normalización automática de columnas (*feature engineering* y desinfectado de nombres), imputación de valores faltantes por defecto e inferencia escalable en lote.
* **🛠️ Gestión Integral de Datos (CRUD):** Módulo para buscar clientes por nombre, ID de cliente o ID de registro, con capacidades directas de edición de atributos y eliminación de registros en la base de datos.
* **☁️ Persistencia Doble (Google Sheets + CSV Local Fallback):** Conectividad nativa con **Google Sheets API** a través de `streamlit-gsheets` para actualización en tiempo real, respaldada por un sistema automático de persistencia local en `data/historico_sheets.csv`.
* **📈 Dashboards Operativos e Indicadores KPI:** Cuadro de mando interactivo con KPIs financieros (cartera expuesta, tasa de conversión comercial, volumen de riesgo) y gráficos de distribución y tendencia.
* **🔄 Bucle de Retroalimentación Comercial:** Captura del resultado de la gestión comercial (*Pendiente, Contrató Servicio, No Aceptó*) para futuras fases de reentrenamiento (*Active Learning*).

---

## 🏗️ Arquitectura de la Aplicación

```text
               ┌────────────────────────────────────────┐
               │         Interfaz Streamlit App         │
               └───────────────────┬────────────────────┘
                                   │
      ┌────────────────────────────┼────────────────────────────┐
      ▼                            ▼                            ▼
┌───────────────┐          ┌───────────────┐          ┌──────────────────┐
│ Tab 1: Manual │          │ Tab 2: Batch  │          │ Tab 3: CRUD      │
│ Inferencia    │          │ CSV Lotes     │          │ Historial        │
└───────┬───────┘          └───────┬───────┘          └─────────┬────────┘
        │                          │                            │
        └──────────────────────────┼────────────────────────────┘
                                   ▼
                ┌──────────────────────────────────┐
                │   Pipeline Inferencia / Modelo   │
                │     (modelo_interconnect.pkl)    │
                └──────────────────┬───────────────┘
                                   │
                   ┌───────────────┴───────────────┐
                   ▼                               ▼
      ┌─────────────────────────┐     ┌──────────────────────────┐
      │  Google Sheets (Cloud)  │     │ CSV Local (data/fallback)│
      └─────────────────────────┘     └──────────────────────────┘

```

---

## 🛠️ Estructura del Proyecto

```text
interconnection-abandonment-mlops/
│
├── .gitignore                         # Excludes environments, local databases, and temporary caches
├── .streamlit/
│   └── config.toml                    # Theme customization (Dark & Indigo setup)
├── README.md                          # Professional project documentation
├── app.py                             # Interactive Streamlit application (Single & Bulk predictions)
├── generate_test_csv.py               # Auxiliary script to generate mock customers for bulk testing
├── modelo_interconnect.pkl            # Trained and serialized production model (Ensure it's generated!)
├── requirements.txt                   # Production dependencies and library versions
├── train.py                           # Data preparation, pipeline fitting, and serialization script
│
├── data/                              # Raw source datasets (Contract, Personal, Internet, Phone)
├── data_reentrenamiento/               # Local directory for saving newly collected feedback loops
├── notebook/                          # Development EDA and prototyping phase
└── src/                               # Modular application backend
    ├── __init__.py
    └── pipeline_config.py             # Feature engineering schemas & Scikit-Learn pipeline definitions


---

## 🚀 Instalación y Configuración Local

### 1. Clonar el repositorio

```bash
git clone [https://github.com/tu-usuario/interconnect-churn-mlops.git](https://github.com/tu-usuario/interconnect-churn-mlops.git)
cd interconnect-churn-mlops

```

### 2. Crear y activar entorno virtual

```bash
python -m venv venv
# En Windows (Git Bash / PowerShell):
source venv/Scripts/activate
# En Linux/macOS:
source venv/bin/activate

```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt

```

### 4. Ejecutar la aplicación

```bash
streamlit run app.py

```

---

## 📊 Variables Entrada del Modelo

| Variable | Tipo | Descripción |
| --- | --- | --- |
| `type` | Categorical | Tipo de contrato (*month-to-month, one_year, two_year*) |
| `paperless_billing` | Categorical | Factura electrónica (*yes, no*) |
| `payment_method` | Categorical | Método de pago (*electronic_check, credit_card, etc.*) |
| `monthly_charges` | Numeric | Cargo cobrado mensualmente |
| `total_charges` | Numeric | Cargo acumulado histórico |
| `internet_service` | Categorical | Tipo de servicio de internet (*dsl, fiber_optic, no_contract*) |
| `months_of_age` | Numeric | Antigüedad del cliente en meses |

---

## ✒️ Autor

* **Cristian Guerrero / Científico de Datos:** Portfolio de Proyectos MLOps
* **Modelo:** Clasificador supervisado optimizado para reducción de tasa de abandono (*Churn*).
