import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
from datetime import datetime

st.set_page_config(page_title="Interconnect Churn Predictor v2", layout="wide")

st.title("📊 Sistema Avanzado de Retención de Clientes - Interconnect")
st.markdown("Herramienta corporativa con persistencia de datos, historial comercial y analíticas de control.")

# ==========================================
# 💾 CONFIGURACIÓN DE BASE DE DATOS (SQLite)
# ==========================================
def conectar_db():
    conn = sqlite3.connect('historial_churn.db')
    return conn

def inicializar_db():
    conn = conectar_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS consultas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT,
            genero TEXT,
            jubilado TEXT,
            pareja TEXT,
            dependientes TEXT,
            antiguedad REAL,
            internet TEXT,
            seguridad TEXT,
            backup TEXT,
            proteccion TEXT,
            soporte TEXT,
            telefonia TEXT,
            streaming_tv TEXT,
            streaming_movies TEXT,
            contrato TEXT,
            factura_electronica TEXT,
            metodo_pago TEXT,
            cargo_mensual REAL,
            cargo_total REAL,
            prediccion TEXT,
            probabilidad REAL,
            propuesta_comercial TEXT
        )
    ''')
    conn.commit()
    conn.close()

inicializar_db()

# ==========================================
# 🧠 CARGA DEL MODELO
# ==========================================
@st.cache_resource
def cargar_pipeline():
    return joblib.load('modelo_interconnect.pkl')

try:
    modelo_produccion = cargar_pipeline()
except FileNotFoundError:
    st.error("❌ Archivo de modelo no detectado. Ejecuta primero 'python train.py' localmente.")
    st.stop()

# Diccionarios de Homologación (Punto 2: Interfaz limpia -> Formato del Modelo)
si_no_map = {"Sí": "yes", "No": "no"}
internet_map = {"Fibra Óptica": "Fiber optic", "DSL": "DSL", "No contrata Internet": "no_contract"}
servicio_ext_map = {"Sí": "yes", "No": "no", "No contrata Internet": "no_contract"}
telefono_map = {"Líneas Múltiples": "yes", "Línea Única": "no", "No contrata Teléfono": "no_contract"}
contrato_map = {"Mes a Mes": "month_to_month", "Un Año": "one_year", "Dos Años": "two_year"}
pago_map = {
    "Cheque Electrónico": "electronic_check",
    "Cheque Enviado": "mailed_check",
    "Transferencia Bancaria Automática": "bank_transfer_automatic",
    "Tarjeta de Crédito Automática": "credit_card_automatic"
}

# Creación de pestañas expandidas
tab1, tab2, tab3, tab4 = st.tabs(["👤 Consulta Individual", "📂 Carga Masiva", "📜 Historial de Consultas", "📈 Estadísticas del Sistema"])

# ==========================================
# TAB 1: CONSULTA INDIVIDUAL (FORMULARIO HOMOLOGADO + GUARDADO)
# ==========================================
with tab1:
    st.subheader("📝 Perfil del Cliente a Evaluar")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender_ui = st.selectbox("Género", ["Masculino", "Femenino"])
        senior_ui = st.selectbox("¿Es Jubilado?", ["No", "Sí"])
        partner_ui = st.selectbox("¿Tiene Pareja?", ["Sí", "No"])
        dependents_ui = st.selectbox("¿Tiene Dependientes?", ["Sí", "No"])
        months_of_age = st.number_input("Antigüedad en meses", min_value=0.0, max_value=100.0, value=12.0)

    with col2:
        internet_service_ui = st.selectbox("Servicio de Internet", list(internet_map.keys()))
        
        # Si no tiene internet, bloqueamos u homologamos los sub-servicios automáticamente
        if internet_service_ui == "No contrata Internet":
            online_security_ui = online_backup_ui = device_protection_ui = tech_support_ui = "No contrata Internet"
            st.info("ℹ️ Servicios digitales desactivados (Sin Internet).")
        else:
            online_security_ui = st.selectbox("Seguridad en Línea", ["Sí", "No"])
            online_backup_ui = st.selectbox("Copia de Seguridad", ["Sí", "No"])
            device_protection_ui = st.selectbox("Protección de Dispositivo", ["Sí", "No"])
            tech_support_ui = st.selectbox("Soporte Técnico", ["Sí", "No"])

    with col3:
        multiple_lines_ui = st.selectbox("Líneas Telefónicas", list(telefono_map.keys()))
        
        if internet_service_ui == "No contrata Internet":
            streaming_tv_ui = streaming_movies_ui = "No contrata Internet"
        else:
            streaming_tv_ui = st.selectbox("Streaming TV", ["Sí", "No"])
            streaming_movies_ui = st.selectbox("Streaming Películas", ["Sí", "No"])
            
        type_contract_ui = st.selectbox("Tipo de Contrato", list(contrato_map.keys()))
        paperless_billing_ui = st.selectbox("Factura Electrónica", ["Sí", "No"])
        payment_method_ui = st.selectbox("Método de Pago", list(pago_map.keys()))

    # Traducir UI al formato nativo que entiende el modelo
    gender = "male" if gender_ui == "Masculino" else "female"
    senior_citizen = 1 if senior_ui == "Sí" else 0
    partner = si_no_map[partner_ui]
    dependents = si_no_map[dependents_ui]
    internet_service = internet_map[internet_service_ui]
    online_security = servicio_ext_map[online_security_ui]
    online_backup = servicio_ext_map[online_backup_ui]
    device_protection = servicio_ext_map[device_protection_ui]
    tech_support = servicio_ext_map[tech_support_ui]
    multiple_lines = telefono_map[multiple_lines_ui]
    streaming_tv = servicio_ext_map[streaming_tv_ui]
    streaming_movies = servicio_ext_map[streaming_movies_ui]
    type_contract = contrato_map[type_contract_ui]
    paperless_billing = si_no_map[paperless_billing_ui]
    payment_method = pago_map[payment_method_ui]

    st.markdown("---")
    st.subheader("💳 Comportamiento Financiero")

    cargo_base = 0.0
    if internet_service == "Fiber optic": cargo_base += 75.0
    elif internet_service == "DSL": cargo_base += 45.0

    servicios_digitales = [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
    servicios_activos = sum(1 for s in servicios_digitales if s == "yes")
    cargo_base += (servicios_activos * 5.5)

    if multiple_lines == "yes": cargo_base += 30.0
    elif multiple_lines == "no": cargo_base += 20.0
    if internet_service == "no_contract" and multiple_lines == "no_contract": cargo_base = 0.0

    monthly_charges = cargo_base
    total_charges = monthly_charges * months_of_age

    c1, c2 = st.columns(2)
    with c1: st.number_input("Cargo Mensual Estimado ($)", value=float(monthly_charges), disabled=True, format="%.2f")
    with c2: st.number_input("Cargos Totales Acumulados ($)", value=float(total_charges), disabled=True, format="%.2f")

    st.markdown("---")

    if st.button("🔍 Evaluar y Guardar en Historial", type="primary"):
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
        
        # Generación de la propuesta comercial basada en el riesgo (Punto 1)
        propuesta = "Mantener monitoreo estándar."
        st.markdown("### 📊 Diagnóstico del Modelo")
        
        if prediccion == 1:
            st.error(f"🚨 **ALTO RIESGO DE DESERCIÓN.** Riesgo: {probabilidad:.1%}")
            if months_of_age <= 6 and internet_service == "Fiber optic":
                propuesta = "CRÍTICO: Otorgar descuento del 20% por 6 meses en Plan Fibra y llamada prioritaria de fidelización."
                st.warning(f"💡 **Estrategia Recomendada:** {propuesta}")
            else:
                propuesta = "ALTO: Ofrecer migración a contrato anual con tarifa congelada."
                st.info(f"💡 **Estrategia Recomendada:** {propuesta}")
        else:
            st.success(f"🟢 **CLIENTE ESTABLE.** Riesgo: {probabilidad:.1%}")
            if type_contract == "month_to_month":
                propuesta = "MODERADO: Ofrecer descuento menor si migra a contrato de 1 año para asegurar permanencia."
                st.caption(f"💡 **Oportunidad de Negocio:** {propuesta}")

        # Guardar en base de datos SQLite (Punto 1)
        conn = conectar_db()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO consultas (fecha, genero, jubilado, pareja, dependientes, antiguedad, internet, seguridad, backup, proteccion, soporte, telefonia, streaming_tv, streaming_movies, contrato, factura_electronica, metodo_pago, cargo_mensual, cargo_total, prediccion, probabilidad, propuesta_comercial)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.now().strftime("%Y-%m-%d %H:%M"), gender_ui, senior_ui, partner_ui, dependents_ui, months_of_age, internet_service_ui, online_security_ui, online_backup_ui, device_protection_ui, tech_support_ui, multiple_lines_ui, streaming_tv_ui, streaming_movies_ui, type_contract_ui, paperless_billing_ui, payment_method_ui, monthly_charges, total_charges, "Riesgo Alto" if prediccion == 1 else "Estable", float(probabilidad), propuesta))
        conn.commit()
        conn.close()
        st.toast("✅ Consulta guardada en el historial con éxito.")

# ==========================================
# TAB 2: CARGA MASIVA 
# ==========================================
with tab2:
    st.subheader("📂 Procesamiento por Lotes")
    archivo_subido = st.file_uploader("Seleccione el archivo CSV de clientes", type=["csv"], key="masivo_uploader")
    if archivo_subido is not None:
        df_usuarios = pd.read_csv(archivo_subido)
        with st.spinner("Procesando..."):
            costo_internet = np.select([df_usuarios['internet_service'] == 'Fiber optic', df_usuarios['internet_service'] == 'DSL'], [75.0, 45.0], default=0.0)
            servicios = ['online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
            conteo_servicios = df_usuarios[servicios].apply(lambda row: sum(1 for s in row if s == 'yes'), axis=1)
            costo_telefono = np.select([df_usuarios['multiple_lines'] == 'yes', df_usuarios['multiple_lines'] == 'no'], [30.0, 20.0], default=0.0)
            tarifa_mensual = costo_internet + (conteo_servicios * 5.5) + costo_telefono
            df_usuarios['monthly_charges'] = np.where((df_usuarios['internet_service'] == 'no_contract') & (df_usuarios['multiple_lines'] == 'no_contract'), 0.0, tarifa_mensual)
            df_usuarios['total_charges'] = df_usuarios['monthly_charges'] * df_usuarios['months_of_age']
            df_usuarios['month_registration'] = 6
            df_usuarios['quarter_registration'] = 2
            
            X_masivo = df_usuarios.drop(columns=['customer_id', 'Prediccion_Churn', 'Probabilidad_Abandono', 'Estatus_Riesgo'], errors='ignore')
            preds = modelo_produccion.predict(X_masivo)
            probs = modelo_produccion.predict_proba(X_masivo)[:, 1]
            
            # Guardar registros masivos en el historial común
            conn = conectar_db()
            for idx, row in df_usuarios.iterrows():
                prop_masiva = "Ofrecer descuento promocional." if preds[idx] == 1 else "Monitoreo estándar."
                conn.cursor().execute('''
                    INSERT INTO consultas (fecha, genero, jubilado, pareja, dependientes, antiguedad, internet, seguridad, backup, proteccion, soporte, telefonia, streaming_tv, streaming_movies, contrato, factura_electronica, metodo_pago, cargo_mensual, cargo_total, prediccion, probabilidad, propuesta_comercial)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (datetime.now().strftime("%Y-%m-%d %H:%M"), row.get('gender','N/D'), str(row.get('senior_citizen',0)), row.get('partner','N/D'), row.get('dependents','N/D'), float(row['months_of_age']), row['internet_service'], 'N/D', 'N/D', 'N/D', 'N/D', row['multiple_lines'], 'N/D', 'N/D', row['type'], 'N/D', row['payment_method'], float(row['monthly_charges']), float(row['total_charges']), "Riesgo Alto" if preds[idx] == 1 else "Estable", float(probs[idx]), prop_masiva))
            conn.commit()
            conn.close()
            
            st.success(f"📊 {len(df_usuarios)} clientes procesados e indexados en el historial.")

# ==========================================
# TAB 3: HISTORIAL DE PERSONAS CONSULTADAS (Punto 3)
# ==========================================
with tab3:
    st.subheader("📜 Historial de Auditoría Comercial")
    st.markdown("Listado completo de evaluaciones almacenadas localmente.")
    
    conn = conectar_db()
    df_historial = pd.read_sql_query("SELECT id, fecha, contrato, internet, cargo_mensual, prediccion, probabilidad, propuesta_comercial FROM consultas ORDER BY id DESC", conn)
    conn.close()
    
    if not df_historial.empty:
        df_historial['probabilidad'] = df_historial['probabilidad'].map(lambda n: f"{n:.1%}")
        st.dataframe(df_historial, use_container_width=True)
        
        if st.button("🗑️ Vaciar Historial de Datos", type="secondary"):
            conn = conectar_db()
            conn.cursor().execute("DELETE FROM consultas")
            conn.commit()
            conn.close()
            st.rerun()
    else:
        st.info("Aún no hay consultas registradas en el sistema.")

# ==========================================
# TAB 4: ESTADÍSTICAS GENERALES (Punto 4)
# ==========================================
with tab4:
    st.subheader("📈 Cuadro de Mando Operativo")
    
    conn = conectar_db()
    df_stats = pd.read_sql_query("SELECT prediccion, cargo_mensual, contrato FROM consultas", conn)
    conn.close()
    
    if not df_stats.empty:
        c1, c2, c3 = st.columns(3)
        
        total_casos = len(df_stats)
        riesgo_alto = len(df_stats[df_stats['prediccion'] == "Riesgo Alto"])
        pct_riesgo = (riesgo_alto / total_casos) if total_casos > 0 else 0
        fuga_financiera = df_stats[df_stats['prediccion'] == "Riesgo Alto"]['cargo_mensual'].sum()
        
        with c1:
            st.metric("Total Consultas", value=total_casos)
        with c2:
            st.metric("Clientes en Riesgo Alto", value=riesgo_alto, delta=f"{pct_riesgo:.1%} del total", delta_color="inverse")
        with c3:
            st.metric("Cartera en Riesgo Mensual", value=f"${fuga_financiera:,.2f} USD")
            
        st.markdown("---")
        
        # Gráficos nativos simples de Streamlit para control estadístico
        st.markdown("#### Distribución de Alertas por Tipo de Contrato")
        contratos_churn = df_stats.groupby(['contrato', 'prediccion']).size().unstack(fill_value=0)
        st.bar_chart(contratos_churn)
        
    else:
        st.info("Registre datos en el sistema para generar métricas estadísticas.")