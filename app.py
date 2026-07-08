import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_gsheets import GSheetsConnection  # 👈 Nueva conexión oficial
from datetime import datetime

# ==========================================
# 📐 CONFIGURACIÓN DE PÁGINA E IDENTIDAD
# ==========================================
st.set_page_config(page_title="Interconnect Churn Predictor v3", layout="wide")

# (Tus estilos CSS personalizados de Glassmorphism se mantienen exactamente igual aquí...)

st.title("📊 Sistema Analítico de Retención - Interconnect")
st.markdown("Plataforma avanzada de MLOps con persistencia en Google Sheets, auditoría comercial y bucle de retroalimentación.")

# ==========================================
# 💾 CONFIGURACIÓN DE GOOGLE SHEETS
# ==========================================
# Establecemos la conexión utilizando los secretos del entorno
conn_sheets = st.connection("gsheets", type=GSheetsConnection)

def inicializar_or_obtener_datos():
    """Lee la hoja de cálculo actual o la inicializa con las columnas correctas si está vacía."""
    try:
        # Intenta leer los datos existentes (limpiando caché para datos en tiempo real)
        df = conn_sheets.read(ttl=0)
        if df.empty or len(df.columns) < 5:
            raise ValueError("Sheet vacía")
        return df
    except Exception:
        # Estructura de columnas idéntica a tu diseño de base de datos original
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
    """Concatena los nuevos registros con el histórico y actualiza Google Sheets."""
    df_historico = inicializar_or_obtener_datos()
    
    # Calcular IDs incrementales autogestionados
    if not df_historico.empty:
        ultimo_id = pd.to_numeric(df_historico["id"]).max()
        if np.isnan(ultimo_id): ultimo_id = 0
    else:
        ultimo_id = 0
        
    nuevos_registros_df.insert(0, "id", range(int(ultimo_id) + 1, int(ultimo_id) + 1 + len(nuevos_registros_df)))
    
    # Unificar tipos para evitar errores de codificación
    df_final = pd.concat([df_historico, nuevos_registros_df], ignore_index=True)
    conn_sheets.update(data=df_final)

# Asegurar la inicialización al arrancar
df_actual_data = inicializar_or_obtener_datos()

# (Los diccionarios globales de mapeo y la carga del pipeline .pkl se mantienen idénticos...)

# ==========================================
# TAB 1: CONSULTA INDIVIDUAL (Sección de Guardado)
# ==========================================
# ... [Código de inputs de interfaz y lógicas de costos financieros se mantienen igual] ...

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
        # ... [Tus condicionales de recomendaciones lógicas se mantienen igual] ...
        
        # Estructurar la fila formateada exactamente para la auditoría
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
        
        guardar_registros_en_sheets(nueva_consulta)
        
        # Recuperar el ID asignado para el feedback loop inmediato
        df_actualizado = inicializar_or_obtener_datos()
        st.session_state['id_actual'] = int(df_actualizado["id"].iloc[-1])
        st.session_state['nombre_actual'] = name_ui
        st.toast(f"✅ Consulta de {name_ui} indexada en Google Sheets.")

    # Bucle de Realimentación Comercial (Feedback Loop)
    if 'id_actual' in st.session_state:
        st.markdown("---")
        st.subheader("🤝 Cierre de la Gestión Comercial")
        with st.container(border=True):
            st.markdown(f"**¿El cliente `{st.session_state['nombre_actual']}` aceptó la propuesta comercial?**")
            resultado_gestion = st.radio("Resultado final de la oferta:", ["Pendiente", "Contrató Servicio ✔️", "No Aceptó ❌"], horizontal=True)
            
            if st.button("💾 Confirmar Decisión del Cliente", type="secondary"):
                df_operativo = inicializar_or_obtener_datos()
                # Localizar la fila por el ID autogestionado y actualizar el estado
                idx_target = df_operativo[df_operativo["id"].astype(str) == str(st.session_state['id_actual'])].index
                if not idx_target.empty:
                    df_operativo.loc[idx_target, "venta_cerrada"] = resultado_gestion
                    conn_sheets.update(data=df_operativo)
                    st.success("📈 Registro actualizado en la nube. Datos listos para el pipeline de reentrenamiento.")

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
            # ... [Tus cálculos financieros vectorizados (costo_internet, tarifa_mensual, etc.) se mantienen igual] ...
            X_masivo = df_usuarios.drop(columns=['customer_id', 'Prediccion_Churn', 'Probabilidad_Abandono', 'Estatus_Riesgo', 'nombre', 'telefono'], errors='ignore')
            preds = modelo_produccion.predict(X_masivo)
            probs = modelo_produccion.predict_proba(X_masivo)[:, 1]
            
            lote_nuevos = []
            for idx, row in df_usuarios.iterrows():
                prop_masiva = "Ofrecer descuento de retención." if preds[idx] == 1 else "Monitoreo estándar."
                lote_nuevos.append({
                    "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "nombre": row.get('nombre', 'Carga Masiva'),
                    "identificador": row.get('customer_id', 'N/D'),
                    "telefono": row.get('telefono', 'Sin Teléfono'),
                    "genero": row.get('gender','male'),
                    "jubilado": str(row.get('senior_citizen',0)),
                    "pareja": row.get('partner','no'),
                    "dependientes": row.get('dependents','no'),
                    "antiguedad": float(row['months_of_age']),
                    "internet": row['internet_service'],
                    "seguridad": 'N/D', "backup": 'N/D', "proteccion": 'N/D', "soporte": 'N/D',
                    "telefonia": row['multiple_lines'],
                    "streaming_tv": 'N/D', "streaming_movies": 'N/D',
                    "contrato": row['type'],
                    "factura_electronica": 'N/D',
                    "metodo_pago": row['payment_method'],
                    "cargo_mensual": float(row['monthly_charges']),
                    "cargo_total": float(row['total_charges']),
                    "prediccion": "Riesgo Alto" if preds[idx] == 1 else "Estable",
                    "probabilidad": float(probs[idx]),
                    "propuesta_comercial": prop_masiva,
                    "venta_cerrada": "Pendiente"
                })
            
            guardar_registros_en_sheets(pd.DataFrame(lote_nuevos))
            st.success(f"📊 Se procesaron y enviaron {len(df_usuarios)} registros a Google Sheets.")

# ==========================================
# TAB 3: HISTORIAL DE PERSONAS CONSULTADAS
# ==========================================
with tab3:
    st.subheader("📜 Historial de Auditoría Comercial (Nube)")
    df_historial = inicializar_or_obtener_datos()
    
    if not df_historial.empty:
        # Invertimos el orden visualmente para mostrar los últimos registros arriba
        df_historial = df_historial.iloc[::-1].reset_index(drop=True)
        
        csv_data = df_historial.to_csv(index=False).encode('utf-8-sig')
        c_descarga, c_borrado = st.columns([1, 1])
        with c_descarga:
            st.download_button("📥 Descargar Dataset Manual (CSV)", data=csv_data, file_name="dataset_churn_sheets.csv", mime="text/csv", type="primary")
            
        # ... [El guardado automático opcional en carpetas locales se mantiene igual si lo deseas] ...
        
        df_vista = df_historial.copy()
        df_vista['probabilidad'] = pd.to_numeric(df_vista['probabilidad'], errors='coerce').map(lambda n: f"{n:.1%}" if not np.isnan(n) else "N/D")
        st.dataframe(df_vista, use_container_width=True)
        
        with c_borrado:
            if st.button("🗑️ Vaciar Historial de Datos", type="secondary"):
                # Sobrescribir la Sheet dejándola únicamente con las cabeceras limpias
                columnas = list(df_historial.columns)
                df_vacio = pd.DataFrame(columns=columnas)
                conn_sheets.update(data=df_vacio)
                st.rerun()
    else:
        st.info("No se registran consultas guardadas en la hoja de cálculo en la nube.")

# ==========================================
# TAB 4: ESTADÍSTICAS DEL SISTEMA (DASHBOARD)
# ==========================================
with tab4:
    st.subheader("📈 Cuadro de Mando Operativo & KPIs (Real-Time)")
    df_stats = inicializar_or_obtener_datos()
    
    if not df_stats.empty:
        # Asegurar casteo correcto de tipos numéricos para los indicadores
        df_stats['cargo_mensual'] = pd.to_numeric(df_stats['cargo_mensual'], errors='coerce').fillna(0.0)
        
        total_casos = len(df_stats)
        riesgo_alto = len(df_stats[df_stats['prediccion'] == "Riesgo Alto"])
        pct_riesgo = (riesgo_alto / total_casos) if total_casos > 0 else 0
        fuga_financiera = df_stats[df_stats['prediccion'] == "Riesgo Alto"]['cargo_mensual'].sum()
        ventas_exitosas = len(df_stats[df_stats['venta_cerrada'] == "Contrató Servicio ✔️"])
        tasa_conversion = (ventas_exitosas / total_casos) if total_casos > 0 else 0
        
        # Grid superior de 4 KPIs
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Consultas", value=f"{total_casos} u.")
        with c2: st.metric("Índice de Alerta", value=f"{pct_riesgo:.1%}", delta=f"{riesgo_alto} En Riesgo", delta_color="inverse")
        with c3: st.metric("Cartera Expuesta", value=f"${fuga_financiera:,.2f}")
        with c4: st.metric("Conversión Comercial", value=f"{tasa_conversion:.1%}", delta=f"{ventas_exitosas} Ganadas")

        st.markdown("---")
        # (El renderizado dinámico de tus gráficos adaptados sigue aquí sin cambios...)