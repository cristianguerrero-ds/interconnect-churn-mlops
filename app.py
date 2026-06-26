import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
from datetime import datetime

# ==========================================
# 📐 CONFIGURACIÓN DE PÁGINA E IDENTIDAD
# ==========================================
st.set_page_config(page_title="Interconnect Churn Predictor v3", layout="wide")

# ==========================================
# 🎨 ESTILOS CSS PERSONALIZADOS: DARK & BLUR
# ==========================================
st.markdown("""
    <style>
    /* Efecto Glassmorphism/Blur aplicado a todos los contenedores con borde de Streamlit */
    div[data-testid="stVerticalBlockBorderWithHeader"] {
        background: rgba(30, 41, 59, 0.45) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        border-radius: 16px !important;
        padding: 25px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
        margin-bottom: 15px !important;
    }

    /* Subtítulos acentuados en Azul Índigo claro */
    h2, h3, h4 {
        color: #818CF8 !important;
        font-weight: 600 !important;
    }
    
    /* Separadores sutiles */
    hr {
        border-color: rgba(99, 102, 241, 0.15) !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Sistema Analítico de Retención - Interconnect")
st.markdown("Plataforma avanzada de MLOps con persistencia de datos, auditoría comercial y bucle de retroalimentación.")

# ==========================================
# 💾 CONFIGURACIÓN DE BASE DE DATOS (SQLite)
# ==========================================
def conectar_db():
    return sqlite3.connect('historial_churn_v1.db')

def inicializar_db():
    conn = conectar_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS consultas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT,
            nombre TEXT,
            identificador TEXT,
            telefono TEXT,
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
            propuesta_comercial TEXT,
            venta_cerrada TEXT DEFAULT 'Pendiente'
        )
    ''')
    conn.commit()
    conn.close()

inicializar_db()

# ==========================================
# 🧠 CARGA DEL PIPELINE MAESTRO
# ==========================================
@st.cache_resource
def cargar_pipeline():
    return joblib.load('modelo_interconnect.pkl')

try:
    modelo_produccion = cargar_pipeline()
except FileNotFoundError:
    st.error("❌ Archivo de modelo 'modelo_interconnect.pkl' no detectado. Ejecuta primero tu script de entrenamiento.")
    st.stop()

# Diccionarios globales de mapeo y homologación (Interfaz limpia en Español -> Entrada del Modelo)
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

# Creación de las pestañas de navegación
tab1, tab2, tab3, tab4 = st.tabs(["👤 Consulta Individual", "📂 Carga Masiva", "📜 Historial de Consultas", "📈 Estadísticas del Sistema"])

# ==========================================
# TAB 1: CONSULTA INDIVIDUAL
# ==========================================
with tab1:
    st.subheader("📝 Perfil del Cliente a Evaluar")
    c_perfil1, c_perfil2 = st.columns(2)
    with c_perfil1:
        name_ui = st.text_input("Nombre del Cliente", value="Cliente Ejemplo")
        gender_ui = st.selectbox("Género", ["Masculino", "Femenino"])
        phone_ui = st.text_input("Teléfono de Contacto", value="300-000-0000")
        partner_ui = st.selectbox("¿Tiene Pareja?", ["Sí", "No"])
        
    with c_perfil2:
        id_ui = st.text_input("ID de Cliente", value="0000-0000")
        senior_ui = st.selectbox("¿Es Jubilado?", ["No", "Sí"])
        dependents_ui = st.selectbox("¿Tiene Dependientes?", ["Sí", "No"])
        months_of_age = st.number_input("Antigüedad en meses", min_value=0.0, max_value=100.0, value=12.0)

    st.markdown("---")
    st.subheader("🌐 Servicios Digitales e Internet")
    c_int1, c_int2 = st.columns(2)
    with c_int1:
        internet_service_ui = st.selectbox("Servicio de Internet", list(internet_map.keys()))
    with c_int2:
        if internet_service_ui == "No contrata Internet":
            online_security_ui = online_backup_ui = device_protection_ui = tech_support_ui = "No contrata Internet"
            st.info("ℹ️ Servicios digitales desactivados de forma automática (Sin Internet).")
        else:
            online_security_ui = st.selectbox("Seguridad en Línea", ["Sí", "No"])
            online_backup_ui = st.selectbox("Copia de Seguridad", ["Sí", "No"])
            device_protection_ui = st.selectbox("Protección de Dispositivo", ["Sí", "No"])
            tech_support_ui = st.selectbox("Soporte Técnico", ["Sí", "No"])

    st.markdown("---")
    st.subheader("📞 Telefonía")
    c_tel1, c_tel2, c_tel3 = st.columns(3)
    with c_tel1:
        multiple_lines_ui = st.selectbox("Líneas Telefónicas", list(telefono_map.keys()))
        if multiple_lines_ui == "No contrata Teléfono":
            st.caption("ℹ️ El cliente no cuenta con servicio de telefonía activo.")
        
    with c_tel2:
        if internet_service_ui == "No contrata Internet":
            streaming_tv_ui = streaming_movies_ui = "No contrata Internet"
            st.caption("ℹ️ Servicios de Streaming desactivados.")
        else:
            streaming_tv_ui = st.selectbox("Streaming TV", ["Sí", "No"])
            
    with c_tel3:
        if internet_service_ui == "No contrata Internet":
            streaming_tv_ui = streaming_movies_ui = "No contrata Internet"
            st.caption("ℹ️ Servicios de Streaming desactivados.")
        else:
            streaming_movies_ui = st.selectbox("Streaming Películas", ["Sí", "No"])

    
    st.markdown("---")
    st.subheader("📝 Contrato")
    c_cont1, c_cont2, c_cont3 = st.columns(3)
    with c_cont1:
        type_contract_ui = st.selectbox("Tipo de Contrato", list(contrato_map.keys()))
    with c_cont2:
        payment_method_ui = st.selectbox("Método de Pago", list(pago_map.keys()))
    with c_cont3:
        paperless_billing_ui = st.selectbox("Factura Electrónica", ["Sí", "No"])

    # Mapeo y Traducción de Variables hacia el formato que requiere el Modelo Entrenado
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

    # Cálculos Financieros basados en Reglas Corporativas
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

    st.markdown("---")
    st.subheader("💳 Comportamiento Financiero")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1: st.number_input("Cargo Mensual Estimado ($)", value=float(monthly_charges), disabled=True, format="%.2f")
        with c2: st.number_input("Cargos Totales Acumulados ($)", value=float(total_charges), disabled=True, format="%.2f")

    st.markdown("---")

    # Ejecución del Diagnóstico y Guardado Inicial
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
        
        propuesta = "Mantener monitoreo estándar."
        
        with st.container(border=True):
            st.markdown("### 📊 Diagnóstico del Modelo")
            if prediccion == 1:
                st.error(f"🚨 **ALTO RIESGO DE DESERCIÓN.** Riesgo: {probabilidad:.1%}")
                if months_of_age <= 6 and internet_service == "Fiber optic":
                    propuesta = f"CRÍTICO: Otorgar descuento a {name_ui} en Plan Fibra y llamada prioritaria."
                    st.warning(f"💡 **Estrategia Recomendada:** {propuesta}")
                else:
                    propuesta = "ALTO: Ofrecer migración a contrato anual con tarifa congelada."
                    st.info(f"💡 **Estrategia Recomendada:** {propuesta}")
            else:
                st.success(f"🟢 **CLIENTE ESTABLE.** Riesgo: {probabilidad:.1%}")
                if type_contract == "month_to_month":
                    propuesta = "MODERADO: Ofrecer incentivo menor para migrar a contrato de 1 año."
                    st.caption(f"💡 **Oportunidad de Negocio:** {propuesta}")

        # Inserción en la base de datos local SQLite con estado 'Pendiente'
        fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M")
        conn = conectar_db()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO consultas (fecha, nombre, identificador, telefono, genero, jubilado, pareja, dependientes, antiguedad, internet, seguridad, backup, proteccion, soporte, telefonia, streaming_tv, streaming_movies, contrato, factura_electronica, metodo_pago, cargo_mensual, cargo_total, prediccion, probabilidad, propuesta_comercial, venta_cerrada)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Pendiente')
        ''', (fecha_actual, name_ui, id_ui, phone_ui, gender_ui, senior_ui, partner_ui, dependents_ui, months_of_age, internet_service_ui, online_security_ui, online_backup_ui, device_protection_ui, tech_support_ui, multiple_lines_ui, streaming_tv_ui, streaming_movies_ui, type_contract_ui, paperless_billing_ui, payment_method_ui, monthly_charges, total_charges, "Riesgo Alto" if prediccion == 1 else "Estable", float(probabilidad), propuesta))
        
        st.session_state['id_actual'] = cursor.lastrowid
        st.session_state['nombre_actual'] = name_ui
        conn.commit()
        conn.close()
        st.toast(f"✅ Consulta de {name_ui} indexada correctamente.")

    # Bucle de Realimentación Comercial (Feedback Loop)
    if 'id_actual' in st.session_state:
        st.markdown("---")
        st.subheader("🤝 Cierre de la Gestión Comercial")
        with st.container(border=True):
            st.markdown(f"**¿El cliente `{st.session_state['nombre_actual']}` aceptó la propuesta comercial?**")
            resultado_gestion = st.radio("Resultado final de la oferta:", ["Pendiente", "Contrató Servicio ✔️", "No Aceptó ❌"], horizontal=True)
            
            if st.button("💾 Confirmar Decisión del Cliente", type="secondary"):
                conn = conectar_db()
                cursor = conn.cursor()
                cursor.execute("UPDATE consultas SET venta_cerrada = ? WHERE id = ?", (resultado_gestion, st.session_state['id_actual']))
                conn.commit()
                conn.close()
                st.success("📈 Registro actualizado. Información lista para futuros reentrenamientos.")

# ==========================================
# TAB 2: CARGA MASIVA DE CLIENTES
# ==========================================
with tab2:
    st.subheader("📂 Procesamiento por Lotes")
    st.markdown("Cargue archivos CSV masivos para auditar bases de datos operativas completas.")
    archivo_subido = st.file_uploader("Seleccione el archivo CSV de clientes", type=["csv"], key="masivo_uploader")
    
    if archivo_subido is not None:
        df_usuarios = pd.read_csv(archivo_subido)
        with st.spinner("Procesando Base de Datos..."):
            costo_internet = np.select([df_usuarios['internet_service'] == 'Fiber optic', df_usuarios['internet_service'] == 'DSL'], [75.0, 45.0], default=0.0)
            servicios = ['online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
            conteo_servicios = df_usuarios[servicios].apply(lambda row: sum(1 for s in row if s == 'yes'), axis=1)
            costo_telefono = np.select([df_usuarios['multiple_lines'] == 'yes', df_usuarios['multiple_lines'] == 'no'], [30.0, 20.0], default=0.0)
            tarifa_mensual = costo_internet + (conteo_servicios * 5.5) + costo_telefono
            df_usuarios['monthly_charges'] = np.where((df_usuarios['internet_service'] == 'no_contract') & (df_usuarios['multiple_lines'] == 'no_contract'), 0.0, tarifa_mensual)
            df_usuarios['total_charges'] = df_usuarios['monthly_charges'] * df_usuarios['months_of_age']
            df_usuarios['month_registration'] = 6
            df_usuarios['quarter_registration'] = 2
            
            X_masivo = df_usuarios.drop(columns=['customer_id', 'Prediccion_Churn', 'Probabilidad_Abandono', 'Estatus_Riesgo', 'nombre'], errors='ignore')
            preds = modelo_produccion.predict(X_masivo)
            probs = modelo_produccion.predict_proba(X_masivo)[:, 1]
            
            conn = conectar_db()
            for idx, row in df_usuarios.iterrows():
                prop_masiva = "Ofrecer descuento de retención." if preds[idx] == 1 else "Monitoreo estándar."
                nom_masivo = row.get('nombre', 'Carga Masiva')
                id_masivo = row.get('customer_id', 'N/D')
                tel_masivo = row.get('telefono', 'Sin Teléfono') # Captura columna si existe en el CSV
                
                conn.cursor().execute('''
                    INSERT INTO consultas (fecha, nombre, identificador, telefono, genero, jubilado, pareja, dependientes, antiguedad, internet, seguridad, backup, proteccion, soporte, telefonia, streaming_tv, streaming_movies, contrato, factura_electronica, metodo_pago, cargo_mensual, cargo_total, prediccion, probabilidad, propuesta_comercial, venta_cerrada)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Pendiente')
                ''', (
                    datetime.now().strftime("%Y-%m-%d %H:%M"), 
                    str(nom_masivo), 
                    str(id_masivo), 
                    str(tel_masivo),  # 👈 ¡Listo! Agregado en su posición correcta
                    row.get('gender','male'), 
                    str(row.get('senior_citizen',0)), 
                    row.get('partner','no'), 
                    row.get('dependents','no'), 
                    float(row['months_of_age']), 
                    row['internet_service'], 
                    'N/D', 'N/D', 'N/D', 'N/D', 
                    row['multiple_lines'], 
                    'N/D', 'N/D', 
                    row['type'], 
                    'N/D', 
                    row['payment_method'], 
                    float(row['monthly_charges']), 
                    float(row['total_charges']), 
                    "Riesgo Alto" if preds[idx] == 1 else "Estable", 
                    float(probs[idx]), 
                    prop_masiva
                ))
            conn.commit()
            conn.close()
            
            st.success(f"📊 Se procesaron y guardaron {len(df_usuarios)} registros en el historial.")

# ==========================================
# TAB 3: HISTORIAL DE PERSONAS CONSULTADAS
# ==========================================
with tab3:
    st.subheader("📜 Historial de Auditoría Comercial")
    conn = conectar_db()
    df_historial = pd.read_sql_query("SELECT * FROM consultas ORDER BY id DESC", conn)
    conn.close()
    
    if not df_historial.empty:
        # 📂 PREPARACIÓN DEL CSV PARA REENTRENAMIENTO (Descarga Manual)
        csv_data = df_historial.to_csv(index=False).encode('utf-8-sig')
        
        c_descarga, c_borrado = st.columns([1, 1])
        with c_descarga:
            st.download_button(
                label="📥 Descargar Dataset Manual (CSV)",
                data=csv_data,
                file_name=f"dataset_reentrenamiento_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                type="primary"
            )
            
        st.markdown("---")
        st.markdown("### 📁 Exportar Localmente a una Carpeta")
        
        # Campo de texto para indicar la ruta de la carpeta donde deseas guardar tus datos
        # Nota: Puedes usar una ruta relativa simple como 'data_reentrenamiento' o una ruta absoluta
        ruta_carpeta = st.text_input("Ruta de la carpeta de destino:", value="data_reentrenamiento")
        
        if st.button("💾 Guardar automáticamente en Carpeta", type="secondary"):
            import os
            try:
                # Crear la carpeta de forma segura si aún no existe en tu computadora
                if not os.path.exists(ruta_carpeta):
                    os.makedirs(ruta_carpeta)
                
                # Definir el nombre del archivo con la marca de tiempo actual
                nombre_archivo = f"dataset_churn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)
                
                # Escribir el DataFrame completo directamente al almacenamiento local
                df_historial.to_csv(ruta_completa, index=False, encoding='utf-8-sig')
                st.success(f"🚀 ¡Éxito! Dataset exportado correctamente en: `{ruta_completa}`")
            except Exception as e:
                st.error(f"❌ Error al escribir en el directorio: {e}")
            
        st.markdown("---")
        
        # Clon opcional para visualización limpia en la UI
        df_vista = df_historial.copy()
        df_vista['probabilidad'] = df_vista['probabilidad'].map(lambda n: f"{n:.1%}")
        st.dataframe(df_vista, use_container_width=True)
        
        with c_borrado:
            if st.button("🗑️ Vaciar Historial de Datos", type="secondary"):
                conn = conectar_db()
                conn.cursor().execute("DROP TABLE IF EXISTS consultas")
                conn.commit()
                conn.close()
                st.rerun()
    else:
        st.info("No se registran consultas guardadas en el sistema local.")

# ==========================================
# TAB 4: ESTADÍSTICAS DEL SISTEMA (DASHBOARD)
# ==========================================
with tab4:
    st.subheader("📈 Cuadro de Mando Operativo & KPIs")
    conn = conectar_db()
    df_stats = pd.read_sql_query("SELECT fecha, prediccion, cargo_mensual, contrato, internet, metodo_pago, venta_cerrada FROM consultas", conn)
    conn.close()
    
    if not df_stats.empty:
        total_casos = len(df_stats)
        riesgo_alto = len(df_stats[df_stats['prediccion'] == "Riesgo Alto"])
        pct_riesgo = (riesgo_alto / total_casos) if total_casos > 0 else 0
        fuga_financiera = df_stats[df_stats['prediccion'] == "Riesgo Alto"]['cargo_mensual'].sum()
        ventas_exitosas = len(df_stats[df_stats['venta_cerrada'] == "Contrató Servicio ✔️"])
        tasa_conversion = (ventas_exitosas / total_casos) if total_casos > 0 else 0
        
        # Grid superior de 4 KPIs
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            with st.container(border=True):
                st.metric("Total Consultas", value=f"{total_casos} u.")
        with c2:
            with st.container(border=True):
                st.metric("Índice de Alerta", value=f"{pct_riesgo:.1%}", delta=f"{riesgo_alto} En Riesgo", delta_color="inverse")
        with c3:
            with st.container(border=True):
                st.metric("Cartera Expuesta", value=f"${fuga_financiera:,.2f}")
        with c4:
            with st.container(border=True):
                st.metric("Conversión Comercial", value=f"{tasa_conversion:.1%}", delta=f"{ventas_exitosas} Ganadas")

        st.markdown("---")
        
        # Gráficos corporativos adaptados al Dark Mode
# --- Gráficos corporativos adaptados al Dark Mode (Optimizados contra errores de color) ---
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            with st.container(border=True):
                st.markdown("#### 📑 Alertas por Tipo de Contrato")
                contratos_churn = df_stats.groupby(['contrato', 'prediccion']).size().unstack(fill_value=0)
                # Eliminamos color fijo para evitar StreamlitColorLengthError si faltan categorías
                st.bar_chart(contratos_churn)
                
        with col_g2:
            with st.container(border=True):
                st.markdown("#### 🌐 Riesgo según Tecnología de Internet")
                internet_churn = df_stats.groupby(['internet', 'prediccion']).size().unstack(fill_value=0)
                st.bar_chart(internet_churn)
                
        st.markdown("---")
        col_g3, col_g4 = st.columns([2, 1])
        with col_g3:
            with st.container(border=True):
                st.markdown("#### 📉 Evolución Temporal de Análisis")
                df_stats['fecha_corta'] = df_stats['fecha'].str.slice(0, 10)
                linea_temporal = df_stats.groupby(['fecha_corta', 'prediccion']).size().unstack(fill_value=0)
                st.line_chart(linea_temporal)
                
        with col_g4:
            with st.container(border=True):
                st.markdown("#### 💳 Canales de Pago")
                pagos_dist = df_stats.groupby('metodo_pago').size()
                st.bar_chart(pagos_dist)
    else:
        st.info("Registre información en el sistema para calcular las proyecciones y estadísticas.")