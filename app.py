# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Interconnect Churn Predictor", layout="wide")

st.title("📊 Sistema de Retención de Clientes - Interconnect")
st.markdown("Herramienta analítica de consulta en tiempo real e ingesta masiva para asesores comerciales.")

@st.cache_resource
def cargar_pipeline():
    return joblib.load('modelo_interconnect.pkl')

try:
    modelo_produccion = cargar_pipeline()
except FileNotFoundError:
    st.error("❌ Archivo de modelo no detectado. Ejecuta primero 'python train.py' localmente.")
    st.stop()

# Creación de pestañas para dividir las funciones de la app
tab1, tab2 = st.tabs(["👤 Consulta Individual", "📂 Carga Masiva de Clientes"])

# ==========================================
# TAB 1: CONSULTA INDIVIDUAL (FORMULARIO)
# ==========================================
with tab1:
    st.subheader("📝 Perfil del Cliente a Evaluar")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Género", ["male", "female"])
        senior_citizen = st.selectbox("¿Es Jubilado/Senior Citizen?", [0, 1])
        partner = st.selectbox("¿Tiene Pareja?", ["yes", "no"])
        dependents = st.selectbox("¿Tiene Dependientes?", ["yes", "no"])
        months_of_age = st.number_input("Antigüedad en meses", min_value=0.0, max_value=100.0, value=12.0)

    with col2:
        internet_service = st.selectbox("Servicio de Internet", ["Fiber optic", "DSL", "no_contract"])
        online_security = st.selectbox("Seguridad en Línea", ["yes", "no", "no_contract"])
        online_backup = st.selectbox("Copia de Seguridad", ["yes", "no", "no_contract"])
        device_protection = st.selectbox("Protección de Dispositivo", ["yes", "no", "no_contract"])
        tech_support = st.selectbox("Soporte Técnico", ["yes", "no", "no_contract"])

    with col3:
        multiple_lines = st.selectbox("Líneas Telefónicas", ["yes", "no", "no_contract"])
        streaming_tv = st.selectbox("Streaming TV", ["yes", "no", "no_contract"])
        streaming_movies = st.selectbox("Streaming Películas", ["yes", "no", "no_contract"])
        type_contract = st.selectbox("Tipo de Contrato", ["month_to_month", "one_year", "two_year"])
        paperless_billing = st.selectbox("Factura Electrónica", ["yes", "no"])
        payment_method = st.selectbox("Método de Pago", ["electronic_check", "mailed_check", "bank_transfer_automatic", "credit_card_automatic"])

    st.markdown("---")
    st.subheader("💳 Comportamiento Financiero (Calculado Automáticamente)")

    cargo_base = 0.0
    if internet_service == "Fiber optic":
        cargo_base += 75.0
    elif internet_service == "DSL":
        cargo_base += 45.0

    servicios_digitales = [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
    servicios_activos = sum(1 for s in servicios_digitales if s == "yes")
    cargo_base += (servicios_activos * 5.5)

    if multiple_lines == "yes":
        cargo_base += 30.0
    elif multiple_lines == "no":
        cargo_base += 20.0

    if internet_service == "no_contract" and multiple_lines == "no_contract":
        cargo_base = 0.0

    monthly_charges = cargo_base
    total_charges = monthly_charges * months_of_age

    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Cargo Mensual Estimado ($)", value=float(monthly_charges), disabled=True, format="%.2f")
    with c2:
        st.number_input("Cargos Totales Acumulados Estimados ($)", value=float(total_charges), disabled=True, format="%.2f")

    st.markdown("---")

    if st.button("🔍 Evaluar Riesgo de Deserción", type="primary"):
        input_data = {
            'type': type_contract, 'paperless_billing': paperless_billing, 'payment_method': payment_method,
            'monthly_charges': monthly_charges, 'total_charges': total_charges, 'gender': gender,
            'senior_citizen': senior_citizen, 'partner': partner, 'dependents': dependents,
            'internet_service': internet_service, 'online_security': online_security, 'online_backup': online_backup,
            'device_protection': device_protection, 'tech_support': tech_support, 'streaming_tv': streaming_tv,
            'streaming_movies': streaming_movies, 'multiple_lines': multiple_lines, 'months_of_age': months_of_age,
            'month_registration': 6, 'quarter_registration': 2
        }
        df_input = pd.DataFrame([input_data])
        
        prediccion = modelo_produccion.predict(df_input)[0]
        probabilidad = modelo_produccion.predict_proba(df_input)[0][1]
        
        st.markdown("### 📊 Diagnóstico del Modelo")
        if prediccion == 1:
            st.error(f"🚨 **ALTO RIESGO DE DESERCIÓN (CHURN).** El cliente presenta alta propensión a cancelar.")
            st.metric(label="Índice de Riesgo", value=f"{probabilidad:.1%}", delta="Estado Crítico", delta_color="inverse")
            if months_of_age <= 6 and internet_service == "Fiber optic":
                st.warning("⚠️ **Insight de Negocio:** Este cliente pertenece al grupo de mayor riesgo histórico (Nuevos con Fibra Óptica - Churn del 65.7%).")
        else:
            st.success(f"🟢 **CLIENTE ESTABLE.** Baja probabilidad detectada de cancelación de servicios.")
            st.metric(label="Índice de Riesgo", value=f"{probabilidad:.1%}", delta="Estable")

# ==========================================
# TAB 2: CARGA MASIVA (DATAFRAME / CSV)
# ==========================================
with tab2:
    st.subheader("📂 Procesamiento por Lotes")
    st.markdown("""
    Suba un archivo **CSV** que contenga las columnas de los clientes nuevos. 
    *Nota: El archivo debe incluir identificadores y características del servicio de los usuarios (no requiere las columnas de costos, el sistema las calculará solo).*
    """)
    
    # Botón para descargar una plantilla de ejemplo
    ejemplo = pd.DataFrame([{
        'customer_id': '0000-XXXXX', 'gender': 'male', 'senior_citizen': 0, 'partner': 'yes', 'dependents': 'no',
        'months_of_age': 10.5, 'internet_service': 'Fiber optic', 'online_security': 'yes', 'online_backup': 'no',
        'device_protection': 'yes', 'tech_support': 'no', 'streaming_tv': 'yes', 'streaming_movies': 'no',
        'multiple_lines': 'yes', 'type': 'month_to_month', 'paperless_billing': 'yes', 'payment_method': 'electronic_check'
    }])
    st.download_button(label="📥 Descargar plantilla CSV de ejemplo", data=ejemplo.to_csv(index=False), file_name="plantilla_interconnect.csv", mime="text/csv")
    
    st.markdown("---")
    archivo_subido = st.file_uploader("Seleccione el archivo CSV de clientes", type=["csv"])
    
    if archivo_subido is not None:
        df_usuarios = pd.read_csv(archivo_subido)
        st.success("✅ Archivo cargado correctamente.")
        
        with st.spinner("🔢 Calculando tarifas financieras y corriendo modelo..."):
            # 1. Automatización de costos masiva aplicando tus reglas de negocio
            costo_internet = np.select(
                [df_usuarios['internet_service'] == 'Fiber optic', df_usuarios['internet_service'] == 'DSL'],
                [75.0, 45.0], default=0.0
            )
            
            servicios = ['online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
            conteo_servicios = df_usuarios[servicios].apply(lambda row: sum(1 for s in row if s == 'yes'), axis=1)
            costo_adicionales = conteo_servicios * 5.5
            
            costo_telefono = np.select(
                [df_usuarios['multiple_lines'] == 'yes', df_usuarios['multiple_lines'] == 'no'],
                [30.0, 20.0], default=0.0
            )
            
            # Forzar tarifa cero si no tiene ningún servicio contratado
            tarifa_mensual = costo_internet + costo_adicionales + costo_telefono
            tarifa_mensual = np.where((df_usuarios['internet_service'] == 'no_contract') & (df_usuarios['multiple_lines'] == 'no_contract'), 0.0, tarifa_mensual)
            
            df_usuarios['monthly_charges'] = tarifa_mensual
            df_usuarios['total_charges'] = df_usuarios['monthly_charges'] * df_usuarios['months_of_age']
            
            # Variables estacionales requeridas por el Pipeline
            df_usuarios['month_registration'] = 6
            df_usuarios['quarter_registration'] = 2
            
            # 2. Ejecutar la predicción masiva usando el Pipeline maestro
            # Quitamos 'customer_id' si existe para que el modelo no se confunda, pero lo preservamos en el reporte final
            X_masivo = df_usuarios.drop(columns=['customer_id'], errors='ignore')
            
            df_usuarios['Prediccion_Churn'] = modelo_produccion.predict(X_masivo)
            df_usuarios['Probabilidad_Abandono'] = modelo_produccion.predict_proba(X_masivo)[:, 1]
            
            # Hacer más amigable el resultado visual
            df_usuarios['Estatus_Riesgo'] = np.where(df_usuarios['Prediccion_Churn'] == 1, "🚨 ALTO RIESGO", "🟢 Estable")
            df_usuarios['Probabilidad_Abandono'] = df_usuarios['Probabilidad_Abandono'].map(lambda n: f"{n:.1%}")
            
            # Ordenar columnas para mostrar lo más importante primero
            columnas_finales = ['customer_id', 'Estatus_Riesgo', 'Probabilidad_Abandono', 'months_of_age', 'monthly_charges', 'total_charges', 'internet_service', 'type']
            df_mostrar = df_usuarios[[col for col in columnas_finales if col in df_usuarios.columns]]
            
            st.subheader("📊 Resultados del Análisis Masivo")
            st.dataframe(df_mostrar, use_container_width=True)
            
            # 3. Permitir descargar el reporte final procesado
            csv_salida = df_usuarios.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Reporte de Riesgos Completo (CSV)",
                data=csv_salida,
                file_name="reporte_riesgo_interconnect.csv",
                mime="text/csv",
                type="primary")