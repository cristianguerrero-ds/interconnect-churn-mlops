import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

# Intentar conexión opcional a Google Sheets; si no está disponible, usamos un CSV local como fallback
try:
    from streamlit_gsheets import GSheetsConnection  # type: ignore
    conn_sheets = st.connection("gsheets", type=GSheetsConnection)
except Exception:
    conn_sheets = None

st.set_page_config(page_title="Interconnect Churn Predictor v3", layout="wide")

st.title("📊 Sistema Analítico de Retención - Interconnect")
st.markdown("Plataforma avanzada de MLOps con persistencia en Google Sheets, auditoría comercial y bucle de retroalimentación.")

def inicializar_or_obtener_datos():
    """Lee la hoja de cálculo (si hay conexión) o un CSV local; inicializa si no existe."""
    columnas = [
        "id", "fecha", "nombre", "identificador", "telefono", "genero", "jubilado", "pareja",
        "dependientes", "antiguedad", "internet", "seguridad", "backup", "proteccion", "soporte",
        "telefonia", "streaming_tv", "streaming_movies", "contrato", "factura_electronica",
        "metodo_pago", "cargo_mensual", "cargo_total", "prediccion", "probabilidad",
        "propuesta_comercial", "venta_cerrada"
    ]

    # Si hay conexión a Sheets, intentar leerla
    if conn_sheets is not None:
        try:
            df = conn_sheets.read(ttl=0)
            if df is None or df.empty or len(df.columns) < 5:
                df_vacio = pd.DataFrame(columns=columnas)
                conn_sheets.update(data=df_vacio)
                return df_vacio
            return df
        except Exception:
            pass

    # Fallback local: archivo CSV en workspace
    local_path = os.path.join("data", "historico_sheets.csv")
    if os.path.exists(local_path):
        try:
            df = pd.read_csv(local_path)
            return df
        except Exception:
            return pd.DataFrame(columns=columnas)
    else:
        return pd.DataFrame(columns=columnas)


def guardar_registros_en_sheets(nuevos_registros_df):
    """Guarda registros en Sheets si está disponible, si no, en CSV local (data/historico_sheets.csv)."""
    df_historico = inicializar_or_obtener_datos()

    # Calcular IDs incrementales autogestionados
    if not df_historico.empty:
        try:
            ultimo_id = pd.to_numeric(df_historico["id"]).max()
            if np.isnan(ultimo_id):
                ultimo_id = 0
        except Exception:
            ultimo_id = 0
    else:
        ultimo_id = 0

    nuevos_registros_df.insert(0, "id", range(int(ultimo_id) + 1, int(ultimo_id) + 1 + len(nuevos_registros_df)))

    df_final = pd.concat([df_historico, nuevos_registros_df], ignore_index=True)

    if conn_sheets is not None:
        try:
            conn_sheets.update(data=df_final)
            return
        except Exception:
            pass

    # Guardar localmente como fallback
    local_path = os.path.join("data", "historico_sheets.csv")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    df_final.to_csv(local_path, index=False)


df_actual_data = inicializar_or_obtener_datos()

# Cargar el modelo de producción si existe
modelo_path = "modelo_interconnect.pkl"
try:
    modelo_produccion = joblib.load(modelo_path)
except Exception:
    modelo_produccion = None

class DummyModel:
    def predict(self, X):
        return np.zeros((len(X),), dtype=int)
    def predict_proba(self, X):
        return np.vstack([1 - np.zeros((len(X),)), np.zeros((len(X),))]).T

if modelo_produccion is None:
    modelo_produccion = DummyModel()


# ==========================================
# TAB 1: CONSULTA INDIVIDUAL (Sección de Guardado)
# ==========================================
tabs = st.tabs(["Consulta Individual", "Carga Masiva", "Historial", "Estadísticas"])
tab1, tab2, tab3, tab4 = tabs

with tab1:
    st.subheader("🔎 Consulta Individual")
    name_ui = st.text_input("Nombre del Cliente")
    id_ui = st.text_input("Identificador (ID)")
    phone_ui = st.text_input("Teléfono")
    gender_ui = st.selectbox("Género", options=["male", "female"], index=0)
    senior_ui = st.checkbox("Jubilado / Senior")
    partner_ui = st.selectbox("¿Tiene pareja?", options=["yes", "no"], index=1)
    dependents_ui = st.selectbox("¿Tiene dependientes?", options=["yes", "no"], index=1)
    months_of_age = st.number_input("Meses de antigüedad", min_value=0, value=12)
    internet_service_ui = st.selectbox("Servicio Internet", options=["dsl", "fiber_optic", "no_contract"], index=0)
    multiple_lines_ui = st.selectbox("Líneas Múltiples", options=["no_contract", "no", "yes"], index=1)
    type_contract_ui = st.selectbox("Tipo de Contrato", options=["month-to-month", "one_year", "two_year"], index=0)
    paperless_billing_ui = st.selectbox("Factura Electrónica", options=["yes", "no"], index=0)
    payment_method_ui = st.selectbox("Método de Pago", options=["electronic_check", "mailed_check", "bank_transfer", "credit_card"], index=0)
    monthly_charges = st.number_input("Cargo Mensual", min_value=0.0, value=50.0)
    total_charges = st.number_input("Cargo Total", min_value=0.0, value=600.0)
    online_security_ui = st.selectbox("Online Security", options=["yes", "no", "no_info"], index=2)
    online_backup_ui = st.selectbox("Online Backup", options=["yes", "no", "no_info"], index=2)
    device_protection_ui = st.selectbox("Device Protection", options=["yes", "no", "no_info"], index=2)
    tech_support_ui = st.selectbox("Tech Support", options=["yes", "no", "no_info"], index=2)
    streaming_tv_ui = st.selectbox("Streaming TV", options=["yes", "no", "no_info"], index=2)
    streaming_movies_ui = st.selectbox("Streaming Movies", options=["yes", "no", "no_info"], index=2)

    if st.button("🔍 Evaluar y Guardar en Historial"):
        input_data = {
            'type': type_contract_ui,
            'paperless_billing': paperless_billing_ui,
            'payment_method': payment_method_ui,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'gender': gender_ui,
            'senior_citizen': int(senior_ui),
            'partner': partner_ui,
            'dependents': dependents_ui,
            'internet_service': internet_service_ui,
            'online_security': online_security_ui,
            'online_backup': online_backup_ui,
            'device_protection': device_protection_ui,
            'tech_support': tech_support_ui,
            'streaming_tv': streaming_tv_ui,
            'streaming_movies': streaming_movies_ui,
            'multiple_lines': multiple_lines_ui,
            'months_of_age': months_of_age,
            'month_registration': 6,
            'quarter_registration': 2
        }
        df_input = pd.DataFrame([input_data])

        prediccion = int(modelo_produccion.predict(df_input)[0])
        probabilidad = float(modelo_produccion.predict_proba(df_input)[0][1]) if hasattr(modelo_produccion, 'predict_proba') else 0.0

        propuesta = "Mantener monitoreo estándar."

        nueva_consulta = pd.DataFrame([{
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "nombre": name_ui,
            "identificador": id_ui,
            "telefono": phone_ui,
            "genero": gender_ui,
            "jubilado": int(senior_ui),
            "pareja": partner_ui,
            "dependientes": dependents_ui,
            "antiguedad": float(months_of_age),
            "internet": internet_service_ui,
            "seguridad": online_security_ui,
            "backup": online_backup_ui,
            "proteccion": device_protection_ui,
            "soporte": tech_support_ui,
            "telefonia": multiple_lines_ui,
            "streaming_tv": streaming_tv_ui,
            "streaming_movies": streaming_movies_ui,
            "contrato": type_contract_ui,
            "factura_electronica": paperless_billing_ui,
            "metodo_pago": payment_method_ui,
            "cargo_mensual": float(monthly_charges),
            "cargo_total": float(total_charges),
            "prediccion": "Riesgo Alto" if prediccion == 1 else "Estable",
            "probabilidad": probabilidad,
            "propuesta_comercial": propuesta,
            "venta_cerrada": "Pendiente"
        }])

        guardar_registros_en_sheets(nueva_consulta)

        df_actualizado = inicializar_or_obtener_datos()
        if not df_actualizado.empty:
            try:
                st.session_state['id_actual'] = int(df_actualizado["id"].iloc[-1])
            except Exception:
                st.session_state['id_actual'] = None
        st.session_state['nombre_actual'] = name_ui
        st.success(f"✅ Consulta de {name_ui} indexada en el histórico.")

    if 'id_actual' in st.session_state:
        st.markdown("---")
        st.subheader("🤝 Cierre de la Gestión Comercial")
        resultado_gestion = st.radio("Resultado final de la oferta:", ["Pendiente", "Contrató Servicio ✔️", "No Aceptó ❌"], horizontal=True)
        if st.button("💾 Confirmar Decisión del Cliente"):
            df_operativo = inicializar_or_obtener_datos()
            if 'id' in df_operativo.columns and st.session_state.get('id_actual') is not None:
                idx_target = df_operativo[df_operativo["id"].astype(str) == str(st.session_state['id_actual'])].index
                if not idx_target.empty:
                    df_operativo.loc[idx_target, "venta_cerrada"] = resultado_gestion
                    guardar_registros_en_sheets(df_operativo)
                    st.success("📈 Registro actualizado en el histórico.")

# ==========================================
# TAB 2: CARGA MASIVA DE CLIENTES (CORREGIDO)
# ==========================================
with tab2:
    st.subheader("📂 Procesamiento Masivo por Lotes")
    st.markdown("Cargue archivos estructurados en formato CSV para procesar预测 y auditorías en bloque.")
    archivo_subido = st.file_uploader("Seleccione el archivo CSV con los datos de clientes:", type=["csv"], key="batch_file_uploader")
    
    if archivo_subido is not None:
        df_usuarios = pd.read_csv(archivo_subido)
        with st.spinner("Procesando datos y ejecutando inferencia en lote..."):
            
            # 1. Crear una copia limpia para la inferencia del modelo
            X_masivo = df_usuarios.copy()
            
            # 2. Remover columnas de identificación que NO van al modelo
            columnas_no_features = ['customer_id', 'Prediccion_Churn', 'Probabilidad_Abandono', 'Estatus_Riesgo', 'nombre', 'telefono']
            X_masivo = X_masivo.drop(columns=columnas_no_features, errors='ignore')
            
            # 3. Lista maestra de columnas EXACTAS que exige tu Pipeline de Scikit-Learn
            # (Asegúrate de que coincidan con las de tu train.py)
            columnas_requeridas = [
                'type', 'paperless_billing', 'payment_method', 'monthly_charges', 'total_charges', 
                'gender', 'senior_citizen', 'partner', 'dependents', 'internet_service', 
                'online_security', 'online_backup', 'device_protection', 'tech_support', 
                'streaming_tv', 'streaming_movies', 'multiple_lines', 'months_of_age', 
                'month_registration', 'quarter_registration'
            ]
            
            # 4. Inyección defensiva: Si falta alguna columna en el CSV, la creamos con un valor neutro
            valores_defecto = {
                'type': 'month_to_month', 'paperless_billing': 'no', 'payment_method': 'electronic_check',
                'monthly_charges': 50.0, 'total_charges': 50.0, 'gender': 'male', 'senior_citizen': 0,
                'partner': 'no', 'dependents': 'no', 'internet_service': 'fiber_optic',
                'online_security': 'no', 'online_backup': 'no', 'device_protection': 'no',
                'tech_support': 'no', 'streaming_tv': 'no', 'streaming_movies': 'no',
                'multiple_lines': 'no', 'months_of_age': 12, 'month_registration': 6, 'quarter_registration': 2
            }
            
            for col in columnas_requeridas:
                if col not in X_masivo.columns:
                    X_masivo[col] = valores_defecto[col]
            
            # 5. Forzar el cálculo de cargos totales si faltan o son nulos
            X_masivo['monthly_charges'] = pd.to_numeric(X_masivo['monthly_charges'], errors='coerce').fillna(55.0)
            X_masivo['months_of_age'] = pd.to_numeric(X_masivo['months_of_age'], errors='coerce').fillna(12)
            X_masivo['total_charges'] = X_masivo['total_charges'].fillna(X_masivo['monthly_charges'] * X_masivo['months_of_age'])
            
            # 6. REORDENAR ESTRICTO: El orden de las columnas debe ser idéntico al de entrenamiento
            X_masivo = X_masivo[columnas_requeridas]
            
            # 7. Ejecutar predicción sin riesgo de desalineación
            preds = modelo_produccion.predict(X_masivo)
            probs = modelo_produccion.predict_proba(X_masivo)[:, 1]
            
# ==========================================
# TAB 3: HISTORIAL
# ==========================================
with tab3:
    st.subheader("📜 Historial de Auditoría Comercial (Local/Sheets)")
    df_historial = inicializar_or_obtener_datos()

    if not df_historial.empty:
        df_historial = df_historial.iloc[::-1].reset_index(drop=True)
        csv_data = df_historial.to_csv(index=False).encode('utf-8-sig')
        c_descarga, c_borrado = st.columns([1, 1])
        with c_descarga:
            st.download_button("📥 Descargar Dataset Manual (CSV)", data=csv_data, file_name="dataset_churn_sheets.csv", mime="text/csv")

        df_vista = df_historial.copy()
        df_vista['probabilidad'] = pd.to_numeric(df_vista.get('probabilidad', 0), errors='coerce').map(lambda n: f"{n:.1%}" if not np.isnan(n) else "N/D")
        st.dataframe(df_vista, use_container_width=True)

        with c_borrado:
            if st.button("🗑️ Vaciar Historial de Datos"):
                columnas = list(df_historial.columns)
                df_vacio = pd.DataFrame(columns=columnas)
                guardar_registros_en_sheets(df_vacio)
                st.experimental_rerun()
    else:
        st.info("No se registran consultas guardadas en el histórico.")


# ==========================================
# TAB 4: ESTADÍSTICAS
# ==========================================
with tab4:
    st.subheader("📈 Cuadro de Mando Operativo & KPIs (Real-Time)")
    df_stats = inicializar_or_obtener_datos()

    if not df_stats.empty:
        df_stats['cargo_mensual'] = pd.to_numeric(df_stats.get('cargo_mensual', 0), errors='coerce').fillna(0.0)

        total_casos = len(df_stats)
        riesgo_alto = len(df_stats[df_stats.get('prediccion') == "Riesgo Alto"]) if 'prediccion' in df_stats.columns else 0
        pct_riesgo = (riesgo_alto / total_casos) if total_casos > 0 else 0
        fuga_financiera = df_stats[df_stats.get('prediccion') == "Riesgo Alto"]['cargo_mensual'].sum() if 'prediccion' in df_stats.columns else 0
        ventas_exitosas = len(df_stats[df_stats.get('venta_cerrada') == "Contrató Servicio ✔️"]) if 'venta_cerrada' in df_stats.columns else 0
        tasa_conversion = (ventas_exitosas / total_casos) if total_casos > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Consultas", value=f"{total_casos} u.")
        with c2:
            st.metric("Índice de Alerta", value=f"{pct_riesgo:.1%}", delta=f"{riesgo_alto} En Riesgo")
        with c3:
            st.metric("Cartera Expuesta", value=f"${fuga_financiera:,.2f}")
        with c4:
            st.metric("Conversión Comercial", value=f"{tasa_conversion:.1%}", delta=f"{ventas_exitosas} Ganadas")

        st.markdown("---")
    else:
        st.info("No hay datos disponibles para KPIs.")
