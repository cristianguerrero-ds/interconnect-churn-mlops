import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_gsheets import GSheetsConnection
from datetime import datetime

# ==========================================
# 📐 CONFIGURACIÓN DE PÁGINA E IDENTIDAD
# ==========================================
st.set_page_config(
    page_title="Interconnect Churn Predictor v3", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilos CSS personalizados para UI Dark Premium y contenedores Glassmorphism
st.markdown("""
    <style>
    /* Fondo principal y textos */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Encabezados */
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }
    
    /* Pestañas (Tabs) estilo moderno */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #1E293B;
        padding: 8px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #6366F1;
        background-color: #0F172A;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366F1 !important;
        color: #FFFFFF !important;
    }
    
    /* Contenedores con efecto de cristalización (Glassmorphism) */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    /* Customización de inputs y selectores */
    .stTextInput>div>div>input, .stSelectbox>div>div, .stNumberInput>div>div>input {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Sistema Analítico de Retención - Interconnect")
st.markdown("Plataforma avanzada de MLOps con persistencia en Google Sheets, auditoría comercial y bucle de retroalimentación en tiempo real.")

# ==========================================
# 💾 INTERFAZ DE CONEXIÓN A GOOGLE SHEETS
# ==========================================
conn_sheets = st.connection("gsheets", type=GSheetsConnection)

def inicializar_or_obtener_datos():
    """Lee la hoja de cálculo en tiempo real o genera la plantilla base si está vacía."""
    try:
        # Se fuerza ttl=0 para omitir el almacenamiento en caché y leer datos actualizados por lotes
        df = conn_sheets.read(ttl=0)
        if df.empty or len(df.columns) < 5:
            raise ValueError("Estructura vacía detectada.")
        return df
    except Exception:
        columnas = [
            "id", "fecha", "nombre", "identificador", "telefono", "genero", "jubilado", "pareja", 
            "dependientes", "antiguedad", "internet", "seguridad", "backup", "proteccion", "soporte", 
            "telefonia", "streaming_tv", "streaming_movies", "contrato", "factura_electronica", 
            "metodo_pago", "cargo_mensual", "cargo_total", "prediccion", "probabilidad", 
            "propuesta_comercial", "venta_cerrada"
        ]
        df_vacio = pd.DataFrame(columns=columnas)
        conn_sheets.update(data=df_vacio)
        return df_vacio

def guardar_registros_en_sheets(nuevos_registros_df):
    """Sincroniza y escribe nuevos clientes asignando IDs incrementales continuos."""
    df_historico = inicializar_or_obtener_datos()
    
    if not df_historico.empty:
        ultimo_id = pd.to_numeric(df_historico["id"], errors='coerce').max()
        if np.isnan(ultimo_id): ultimo_id = 0
    else:
        ultimo_id = 0
        
    nuevos_registros_df.insert(0, "id", range(int(ultimo_id) + 1, int(ultimo_id) + 1 + len(nuevos_registros_df)))
    
    # Concatenación y actualización limpia del archivo remoto
    df_final = pd.concat([df_historico, nuevos_registros_df], ignore_index=True)
    # Convertir todas las columnas a string o tipos nativos serializables para evitar bloqueos en la API
    for col in df_final.columns:
        if df_final[col].dtype == object:
            df_final[col] = df_final[col].astype(str)
            
    conn_sheets.update(data=df_final)

# Asegurar sincronización al iniciar la instancia
df_actual_data = inicializar_or_obtener_datos()

# ==========================================
# 🧠 CARGA DEL ARTIFACTO DE MACHINE LEARNING
# ==========================================
@st.cache_resource
def cargar_pipeline_produccion():
    """Carga el modelo y pipeline Scikit-Learn serializado."""
    return joblib.load('modelo_interconnect.pkl')

try:
    modelo_produccion = cargar_pipeline_produccion()
except Exception as e:
    st.error(f"❌ Error crítico: No se encontró el archivo 'modelo_interconnect.pkl'. Por favor ejecuta 'train.py' localmente. Detalles: {e}")
    st.stop()

# Diccionarios de equivalencia lingüística para la interfaz de usuario
dicc_genero = {"Masculino": "male", "Femenino": "female"}
dicc_binario = {"Sí": "yes", "No": "no"}
dicc_internet = {"Fibra Óptica": "fiber_optic", "DSL": "dsl", "No tiene": "no"}
dicc_contrato = {"Mes a Mes": "month_to_month", "Un Año": "one_year", "Dos Años": "two_year"}
dicc_pago = {
    "Cheque Electrónico": "electronic_check",
    "Cheque Enviado": "mailed_check",
    "Transferencia Bancaria (Auto)": "bank_transfer_(automatic)",
    "Tarjeta de Crédito (Auto)": "credit_card_(automatic)"
}

# Organizar módulos operativos mediante pestañas
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Consulta Individual", 
    "📂 Carga por Lotes (CSV)", 
    "📜 Historial de Auditoría", 
    "📈 Dashboard Ejecutivo"
])

# ==========================================
# TAB 1: CONSULTA INDIVIDUAL
# ==========================================
with tab1:
    st.subheader("🎯 Evaluación de Riesgo en Tiempo Real")
    st.markdown("Ingrese el perfil completo del cliente para calcular la probabilidad de abandono y desplegar estrategias.")
    
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        colA, colB, colC = st.columns(3)
        with colA:
            name_ui = st.text_input("Nombre Completo:", value="Cliente Nuevo")
        with colB:
            id_ui = st.text_input("Identificador de Cliente (ID):", value="0000-AAAA")
        with colC:
            phone_ui = st.text_input("Número Telefónico:", value="Sin Teléfono")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 👤 Perfil Sociodemográfico")
        c1, c2, c3, c4 = st.columns(4)
        with c1: gender_ui = st.selectbox("Género:", list(dicc_genero.keys()))
        with c2: senior_ui = st.selectbox("¿Es Adulto Mayor / Jubilado?:", list(dicc_binario.keys()))
        with c3: partner_ui = st.selectbox("¿Tiene Pareja?:", list(dicc_binario.keys()))
        with c4: dependents_ui = st.selectbox("¿Tiene Dependientes / Hijos?:", list(dicc_binario.keys()))
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🔌 Servicios Contratados & Configuración Técnica")
        c1, c2, c3 = st.columns(3)
        with c1: internet_service_ui = st.selectbox("Servicio de Internet:", list(dicc_internet.keys()))
        with c2: multiple_lines_ui = st.selectbox("Servicio de Telefonía (Líneas Múltiples):", ["No tiene", "Línea Única", "Líneas Múltiples"])
        with c3: type_contract_ui = st.selectbox("Tipo de Contrato:", list(dicc_contrato.keys()))
        
        c4, c5, c6, c7 = st.columns(4)
        with c4: online_security_ui = st.selectbox("Seguridad en Línea:", ["No", "Sí", "Sin Internet"])
        with c5: online_backup_ui = st.selectbox("Respaldo en la Nube (Backup):", ["No", "Sí", "Sin Internet"])
        with c6: device_protection_ui = st.selectbox("Protección de Dispositivos:", ["No", "Sí", "Sin Internet"])
        with c7: tech_support_ui = st.selectbox("Soporte Técnico Avanzado:", ["No", "Sí", "Sin Internet"])
        
        c8, c9, c10 = st.columns(3)
        with c8: streaming_tv_ui = st.selectbox("Streaming TV:", ["No", "Sí", "Sin Internet"])
        with c9: streaming_movies_ui = st.selectbox("Streaming Películas:", ["No", "Sí", "Sin Internet"])
        with c10: paperless_billing_ui = st.selectbox("Factura Electrónica:", list(dicc_binario.keys()))
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 💳 Historial Comercial y Financiero")
        c1, c2 = st.columns(2)
        with c1: payment_method_ui = st.selectbox("Método de Pago:", list(dicc_pago.keys()))
        with c2: months_of_age = st.number_input("Antigüedad del Cliente (Meses):", min_value=1, max_value=120, value=12)
        
        # Mapeo estricto para cálculo automático de tarifas financieras (Lógica del Negocio)
        internet_service = dicc_internet[internet_service_ui]
        multiple_lines = "no" if multiple_lines_ui == "No tiene" else ("yes" if multiple_lines_ui == "Líneas Múltiples" else "no")
        
        costo_fijo_base = 20.0
        costo_internet = 40.0 if internet_service == "fiber_optic" else (25.0 if internet_service == "dsl" else 0.0)
        costo_telefono = 10.0 if multiple_lines == "yes" else (5.0 if multiple_lines_ui == "Línea Única" else 0.0)
        costo_extras = 0.0
        for svc in [online_security_ui, online_backup_ui, device_protection_ui, tech_support_ui, streaming_tv_ui, streaming_movies_ui]:
            if svc == "Sí": costo_extras += 5.0
            
        monthly_charges = costo_fijo_base + costo_internet + costo_telefono + costo_extras
        total_charges = monthly_charges * months_of_age
        
        col_m1, col_m2 = st.columns(2)
        with col_m1: st.info(f"💵 Cargo Mensual Estimado: **${monthly_charges:,.2f}**")
        with col_m2: st.info(f"📈 Cargo Total Acumulado: **${total_charges:,.2f}**")
        st.markdown('</div>', unsafe_allow_html=True)

    # Botón de ejecución y guardado
    if st.button("🔍 Evaluar y Registrar en Google Sheets", type="primary"):
        # Adaptación exacta del payload de features para la tubería Scikit-Learn
        gender = dicc_genero[gender_ui]
        senior_citizen = 1 if senior_ui == "Sí" else 0
        partner = dicc_binario[partner_ui]
        dependents = dicc_binario[dependents_ui]
        type_contract = dicc_contrato[type_contract_ui]
        paperless_billing = dicc_binario[paperless_billing_ui]
        payment_method = dicc_pago[payment_method_ui]
        
        input_data = {
            'type': type_contract, 'paperless_billing': paperless_billing, 'payment_method': payment_method,
            'monthly_charges': monthly_charges, 'total_charges': total_charges, 'gender': gender,
            'senior_citizen': senior_citizen, 'partner': partner, 'dependents': dependents,
            'internet_service': internet_service, 'online_security': "yes" if online_security_ui == "Sí" else "no", 
            'online_backup': "yes" if online_backup_ui == "Sí" else "no", 'device_protection': "yes" if device_protection_ui == "Sí" else "no", 
            'tech_support': "yes" if tech_support_ui == "Sí" else "no", 'streaming_tv': "yes" if streaming_tv_ui == "Sí" else "no", 
            'streaming_movies': "yes" if streaming_movies_ui == "Sí" else "no", 'multiple_lines': multiple_lines, 
            'months_of_age': months_of_age, 'month_registration': 6, 'quarter_registration': 2
        }
        
        df_input = pd.DataFrame([input_data])
        
        prediccion = modelo_produccion.predict(df_input)[0]
        probabilidad = modelo_produccion.predict_proba(df_input)[0][1]
        
        # Estrategias comerciales dinámicas mapeadas automáticamente
        if prediccion == 1:
            propuesta = "Ofrecer plan de retención: Descuento del 25% en cargos fijos por 6 meses o actualización de velocidad sin costo adicional."
        else:
            propuesta = "Mantener monitoreo estándar. Plan de fidelización cruzada (Cross-selling) en servicios de entretenimiento."
            
        # Generar DataFrame formateado listo para indexar
        nueva_consulta = pd.DataFrame([{
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "nombre": name_ui,
            "identificador": id_ui,
            "telefono": phone_ui,
            "genero": gender_ui,
            "jubilado": senior_ui,
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
            "probabilidad": float(probabilidad),
            "propuesta_comercial": propuesta,
            "venta_cerrada": "Pendiente"
        }])
        
        with st.spinner("Sincronizando con la nube de Google..."):
            guardar_registros_en_sheets(nueva_consulta)
            
        df_actualizado = inicializar_or_obtener_datos()
        st.session_state['id_actual'] = int(df_actualizado["id"].iloc[-1])
        st.session_state['nombre_actual'] = name_ui
        
        # Despliegue visual de resultados
        st.markdown("---")
        st.subheader("💡 Diagnóstico del Modelo")
        c_r1, c_r2 = st.columns(2)
        with c_r1:
            if prediccion == 1:
                st.error(f"🚨 **ALERTA DE ABANDONO DETECTADA**\n\nEl cliente tiene una alta propensión a cancelar el servicio.")
            else:
                st.success(f"✔️ **CLIENTE FIDELIZADO / ESTABLE**\n\nEl comportamiento del perfil indica estabilidad operacional.")
        with c_r2:
            st.metric(label="Probabilidad de Churn", value=f"{probabilidad:.2%}")
            
        st.info(f"📋 **Estrategia Recomendada:** {propuesta}")

    # Bucle de Realimentación Comercial (Feedback Loop)
    if 'id_actual' in st.session_state:
        st.markdown("---")
        st.subheader("🤝 Cierre de la Gestión Comercial")
        with st.container(border=True):
            st.markdown(f"**¿El cliente `{st.session_state['nombre_actual']}` aceptó la propuesta comercial sugerida?**")
            resultado_gestion = st.radio("Resultado final de la oferta:", ["Pendiente", "Contrató Servicio ✔️", "No Aceptó ❌"], horizontal=True)
            
            if st.button("💾 Confirmar Decisión del Cliente", type="secondary"):
                with st.spinner("Actualizando traza de auditoría comercial..."):
                    df_operativo = inicializar_or_obtener_datos()
                    idx_target = df_operativo[df_operativo["id"].astype(str) == str(st.session_state['id_actual'])].index
                    if not idx_target.empty:
                        df_operativo.loc[idx_target, "venta_cerrada"] = resultado_gestion
                        conn_sheets.update(data=df_operativo)
                        st.success("📈 Registro comercial actualizado correctamente en Google Sheets. Datos listos para reentrenamiento futuro.")

# ==========================================
# TAB 2: CARGA MASIVA DE CLIENTES
# ==========================================
with tab2:
    st.subheader("📂 Procesamiento Masivo por Lotes")
    st.markdown("Cargue archivos estructurados en formato CSV para procesar auditorías y predicciones en bloque.")
    archivo_subido = st.file_uploader("Seleccione el archivo CSV con los datos de clientes:", type=["csv"], key="batch_file_uploader")
    
    if archivo_subido is not None:
        df_usuarios = pd.read_csv(archivo_subido)
        with st.spinner("Procesando datos y ejecutando inferencia en lote..."):
            # Lógica matemática de normalización y asignación de tarifas ausentes si aplica
            if 'monthly_charges' not in df_usuarios.columns:
                df_usuarios['monthly_charges'] = 55.0
            if 'total_charges' not in df_usuarios.columns:
                df_usuarios['total_charges'] = df_usuarios['monthly_charges'] * df_usuarios.get('months_of_age', 12)
                
            # Limpieza limpia de payload para evitar quiebres en el pipeline
            X_masivo = df_usuarios.drop(columns=['customer_id', 'Prediccion_Churn', 'Probabilidad_Abandono', 'Estatus_Riesgo', 'nombre', 'telefono'], errors='ignore')
            
            # Forzar columnas requeridas por el pipeline si el CSV de prueba viene simplificado
            if 'month_registration' not in X_masivo.columns: X_masivo['month_registration'] = 6
            if 'quarter_registration' not in X_masivo.columns: X_masivo['quarter_registration'] = 2
                
            preds = modelo_produccion.predict(X_masivo)
            probs = modelo_produccion.predict_proba(X_masivo)[:, 1]
            
            lote_nuevos = []
            for idx, row in df_usuarios.iterrows():
                prop_masiva = "Ofrecer descuento de retención corporativo." if preds[idx] == 1 else "Monitoreo estándar de cuenta."
                lote_nuevos.append({
                    "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "nombre": row.get('nombre', 'Carga Masiva'),
                    "identificador": row.get('customer_id', 'N/D'),
                    "telefono": row.get('telefono', 'Sin Teléfono'),
                    "genero": row.get('gender', 'male'),
                    "jubilado": str(row.get('senior_citizen', 0)),
                    "pareja": row.get('partner', 'no'),
                    "dependientes": row.get('dependents', 'no'),
                    "antiguedad": float(row.get('months_of_age', 12)),
                    "internet": row.get('internet_service', 'fiber_optic'),
                    "seguridad": 'N/D', "backup": 'N/D', "proteccion": 'N/D', "soporte": 'N/D',
                    "telefonia": row.get('multiple_lines', 'no'),
                    "streaming_tv": 'N/D', "streaming_movies": 'N/D',
                    "contrato": row.get('type', 'month_to_month'),
                    "factura_electronica": 'N/D',
                    "metodo_pago": row.get('payment_method', 'electronic_check'),
                    "cargo_mensual": float(row.get('monthly_charges', 0.0)),
                    "cargo_total": float(row.get('total_charges', 0.0)),
                    "prediccion": "Riesgo Alto" if preds[idx] == 1 else "Estable",
                    "probabilidad": float(probs[idx]),
                    "propuesta_comercial": prop_masiva,
                    "venta_cerrada": "Pendiente"
                })
                
            guardar_registros_en_sheets(pd.DataFrame(lote_nuevos))
            st.success(f"🚀 ¡Éxito! Se procesaron y enviaron {len(df_usuarios)} registros de manera directa a Google Sheets.")

# ==========================================
# TAB 3: HISTORIAL DE PERSONAS CONSULTADAS
# ==========================================
with tab3:
    st.subheader("📜 Historial de Auditoría Comercial (Nube)")
    st.markdown("Trazabilidad completa de consultas almacenadas de forma persistente en Google Sheets.")
    df_historial = inicializar_or_obtener_datos()
    
    if not df_historial.empty:
        # Reversión de orden para mostrar las últimas auditorías registradas en el tope visual
        df_historial = df_historial.iloc[::-1].reset_index(drop=True)
        
        # Generación dinámica del CSV en memoria para bucles de reentrenamiento remoto (MLOps Loop)
        csv_data = df_historial.to_csv(index=False).encode('utf-8-sig')
        c_descarga, c_borrado = st.columns([1, 1])
        with c_descarga:
            st.download_button(
                label="📥 Descargar Dataset Completo (CSV)", 
                data=csv_data, 
                file_name=f"dataset_reentrenamiento_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", 
                mime="text/csv", 
                type="primary"
            )
            
        st.markdown("---")
        
        df_vista = df_historial.copy()
        df_vista['probabilidad'] = pd.to_numeric(df_vista['probabilidad'], errors='coerce').map(lambda n: f"{n:.1%}" if not np.isnan(n) else "N/D")
        st.dataframe(df_vista, use_container_width=True)
        
        with c_borrado:
            if st.button("🗑️ Vaciar Historial Completo", type="secondary"):
                with st.spinner("Borrando registros en la nube..."):
                    columnas = list(df_historial.columns)
                    df_vacio = pd.DataFrame(columns=columnas)
                    conn_sheets.update(data=df_vacio)
                    st.rerun()
    else:
        st.info("No se registran consultas guardadas en el sistema local o remoto.")

# ==========================================
# TAB 4: ESTADÍSTICAS DEL SISTEMA (DASHBOARD)
# ==========================================
with tab4:
    st.subheader("📈 Cuadro de Mando Operativo & KPIs")
    df_stats = inicializar_or_obtener_datos()
    
    if not df_stats.empty:
        df_stats['cargo_mensual'] = pd.to_numeric(df_stats['cargo_mensual'], errors='coerce').fillna(0.0)
        
        total_casos = len(df_stats)
        riesgo_alto = len(df_stats[df_stats['prediccion'].str.contains("Riesgo Alto", na=False, case=False)])
        pct_riesgo = (riesgo_alto / total_casos) if total_casos > 0 else 0
        fuga_financiera = df_stats[df_stats['prediccion'].str.contains("Riesgo Alto", na=False, case=False)]['cargo_mensual'].sum()
        ventas_exitosas = len(df_stats[df_stats['venta_cerrada'].str.contains("Contrató", na=False, case=False)])
        tasa_conversion = (ventas_exitosas / total_casos) if total_casos > 0 else 0
        
        # Grid métrico de KPIs
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Consultas", value=f"{total_casos} u.")
        with c2: st.metric("Índice de Alerta", value=f"{pct_riesgo:.1%}", delta=f"{riesgo_alto} En Riesgo", delta_color="inverse")
        with c3: st.metric("Cartera Expuesta", value=f"${fuga_financiera:,.2f}")
        with c4: st.metric("Conversión Comercial", value=f"{tasa_conversion:.1%}", delta=f"{ventas_exitosas} Ganadas")
        
        st.markdown("---")
        
        # Distribución de predicciones básicas
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("#### 🚨 Distribución de Estados de Riesgo")
            conteo_pred = df_stats['prediccion'].value_counts()
            st.bar_chart(conteo_pred, color="#6366F1")
        with col_g2:
            st.markdown("#### 🤝 Estatus de Conversión Comercial")
            conteo_ventas = df_stats['venta_cerrada'].value_counts()
            st.bar_chart(conteo_ventas, color="#10B981")
    else:
        st.info("Cargue o evalúe registros para desplegar indicadores analíticos automatizados.")