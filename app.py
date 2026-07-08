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
# DEFINICIÓN DE TABS
# ==========================================
tabs = st.tabs(["Consulta Individual", "Carga Masiva", "Historial", "Estadísticas"])
tab1, tab2, tab3, tab4 = tabs

# ==========================================
# TAB 1: CONSULTA INDIVIDUAL
# ==========================================
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

        propuesta = "Ofrecer plan de retención comercial" if prediccion == 1 else "Mantener monitoreo estándar."

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
                    
                    # Para sobreescribir la base completa modificada en vez de duplicar registros
                    if conn_sheets is not None:
                        try:
                            conn_sheets.update(data=df_operativo)
                        except Exception:
                            pass
                    local_path = os.path.join("data", "historico_sheets.csv")
                    df_operativo.to_csv(local_path, index=False)
                    st.success("📈 Registro actualizado en el histórico.")

# ==========================================
# TAB 2: CARGA MASIVA DE CLIENTES
# ==========================================
with tab2:
    st.subheader("📂 Procesamiento Masivo por Lotes")
    st.markdown("Cargue archivos estructurados en formato CSV para procesar predicciones y auditorías en bloque.")
    archivo_subido = st.file_uploader("Seleccione el archivo CSV con los datos de clientes:", type=["csv"], key="batch_file_uploader")
    
    if archivo_subido is not None:
        df_usuarios = pd.read_csv(archivo_subido)
        
        with st.spinner("Procesando datos y ejecutando inferencia en lote..."):
            # 1. Copia limpia y normalización de nombres de columnas del CSV subido
            X_masivo = df_usuarios.copy()
            X_masivo.columns = X_masivo.columns.str.lower().str.replace(' ', '_').str.strip()
            
            # 2. Remover columnas de metadata comercial que no procesa el modelo matemático
            columnas_no_features = ['customer_id', 'prediccion_churn', 'probabilidad_abandono', 'estatus_riesgo', 'nombre', 'telefono', 'id', 'fecha', 'venta_cerrada', 'propuesta_comercial']
            X_masivo = X_masivo.drop(columns=columnas_no_features, errors='ignore')
            
            # 3. Mapeo estricto exigido por tu Pipeline de Scikit-Learn
            columnas_requeridas = [
                'type', 'paperless_billing', 'payment_method', 'monthly_charges', 'total_charges', 
                'gender', 'senior_citizen', 'partner', 'dependents', 'internet_service', 
                'online_security', 'online_backup', 'device_protection', 'tech_support', 
                'streaming_tv', 'streaming_movies', 'multiple_lines', 'months_of_age', 
                'month_registration', 'quarter_registration'
            ]
            
            # 4. Inyección de valores neutros en caso de que falte alguna columna en el archivo externo
            valores_defecto = {
                'type': 'month-to-month', 'paperless_billing': 'no', 'payment_method': 'electronic_check',
                'monthly_charges': 50.0, 'total_charges': 50.0, 'gender': 'male', 'senior_citizen': 0,
                'partner': 'no', 'dependents': 'no', 'internet_service': 'fiber_optic',
                'online_security': 'no', 'online_backup': 'no', 'device_protection': 'no',
                'tech_support': 'no', 'streaming_tv': 'no', 'streaming_movies': 'no',
                'multiple_lines': 'no', 'months_of_age': 12, 'month_registration': 6, 'quarter_registration': 2
            }
            
            for col in columnas_requeridas:
                if col not in X_masivo.columns:
                    X_masivo[col] = valores_defecto[col]
            
            # 5. Sanitizar y limpiar tipos de datos numéricos provenientes del archivo cargado
            X_masivo['monthly_charges'] = pd.to_numeric(X_masivo['monthly_charges'], errors='coerce').fillna(valores_defecto['monthly_charges'])
            X_masivo['months_of_age'] = pd.to_numeric(X_masivo['months_of_age'], errors='coerce').fillna(valores_defecto['months_of_age'])
            X_masivo['total_charges'] = pd.to_numeric(X_masivo['total_charges'], errors='coerce')
            X_masivo['total_charges'] = X_masivo['total_charges'].fillna(X_masivo['monthly_charges'] * X_masivo['months_of_age'])
            
            # 6. Reordenación estructural final
            X_masivo = X_masivo[columnas_requeridas]
            
            # 7. Inferencia matemática
            preds = modelo_produccion.predict(X_masivo)
            probs = modelo_produccion.predict_proba(X_masivo)[:, 1] if hasattr(modelo_produccion, 'predict_proba') else np.zeros(len(X_masivo))
            
            # 8. Mapear y preparar estructura para el histórico (Google Sheets o Local fallback)
            df_lote = pd.DataFrame()
            df_lote['fecha'] = [datetime.now().strftime("%Y-%m-%d %H:%M")] * len(df_usuarios)
            
            # Columnas comerciales opcionales del archivo cargado
            df_usuarios.columns = df_usuarios.columns.str.lower().str.replace(' ', '_').str.strip()
            df_lote['nombre'] = df_usuarios['nombre'] if 'nombre' in df_usuarios.columns else df_usuarios['customer_id'] if 'customer_id' in df_usuarios.columns else "Cliente_Masivo"
            df_lote['identificador'] = df_usuarios['customer_id'] if 'customer_id' in df_usuarios.columns else df_usuarios['identificador'] if 'identificador' in df_usuarios.columns else "N/D"
            df_lote['telefono'] = df_usuarios['telefono'] if 'telefono' in df_usuarios.columns else "N/D"
            
            # Llenado de variables mapeadas correspondientes a las columnas requeridas
            df_lote['genero'] = X_masivo['gender']
            df_lote['jubilado'] = X_masivo['senior_citizen'].astype(int)
            df_lote['pareja'] = X_masivo['partner']
            df_lote['dependientes'] = X_masivo['dependents']
            df_lote['antiguedad'] = X_masivo['months_of_age'].astype(float)
            df_lote['internet'] = X_masivo['internet_service']
            df_lote['seguridad'] = X_masivo['online_security']
            df_lote['backup'] = X_masivo['online_backup']
            df_lote['proteccion'] = X_masivo['device_protection']
            df_lote['soporte'] = X_masivo['tech_support']
            df_lote['telefonia'] = X_masivo['multiple_lines']
            df_lote['streaming_tv'] = X_masivo['streaming_tv']
            df_lote['streaming_movies'] = X_masivo['streaming_movies']
            df_lote['contrato'] = X_masivo['type']
            df_lote['factura_electronica'] = X_masivo['paperless_billing']
            df_lote['metodo_pago'] = X_masivo['payment_method']
            df_lote['cargo_mensual'] = X_masivo['monthly_charges'].astype(float)
            df_lote['cargo_total'] = X_masivo['total_charges'].astype(float)
            
            # Resultados del Modelo
            df_lote['prediccion'] = ["Riesgo Alto" if p == 1 else "Estable" for p in preds]
            df_lote['probabilidad'] = probs.astype(float)
            df_lote['propuesta_comercial'] = ["Ofrecer plan de retención comercial" if p == 1 else "Mantener monitoreo estándar." for p in preds]
            df_lote['venta_cerrada'] = ["Pendiente"] * len(df_usuarios)
            
            # Guardar registros procesados
            guardar_registros_en_sheets(df_lote)
            st.success(f"🎉 ¡Inferencia exitosa! Se han procesado y guardado {len(df_lote)} registros en el historial.")
            st.dataframe(df_lote[['identificador', 'nombre', 'cargo_mensual', 'prediccion', 'probabilidad']].head())

# ==========================================
# TAB 3: HISTORIAL & OPERACIONES CRUD (CORREGIDO)
# ==========================================
with tab3:
    st.subheader("📜 Historial de Auditoría e Interfaz de Modificación")
    df_crud = inicializar_or_obtener_datos()

    if not df_crud.empty:
        st.markdown("### 🛠️ Buscar, Actualizar o Eliminar Clientes")
        
        c_busqueda, c_criterio = st.columns([2, 1])
        with c_criterio:
            tipo_busqueda = st.selectbox("Buscar por:", ["Nombre", "Identificador Único (ID Cliente)", "ID de Registro Interno"])
        with c_busqueda:
            query_busqueda = st.text_input("Ingrese el término de búsqueda:")

        df_filtrado = pd.DataFrame()
        if query_busqueda:
            if tipo_busqueda == "Nombre":
                df_filtrado = df_crud[df_crud['nombre'].astype(str).str.contains(query_busqueda, case=False, na=False)]
            elif tipo_busqueda == "Identificador Único (ID Cliente)":
                df_filtrado = df_crud[df_crud['identificador'].astype(str).str.contains(query_busqueda, case=False, na=False)]
            else:
                df_filtrado = df_crud[df_crud['id'].astype(str) == str(query_busqueda)]

        if not df_filtrado.empty:
            st.write(f"🔍 Se encontraron {len(df_filtrado)} coincidencias:")
            st.dataframe(df_filtrado[['id', 'identificador', 'nombre', 'telefono', 'prediccion', 'venta_cerrada']])
            
            # Añadimos un key único para congelar el estado interno durante el renderizado dinámico
            seleccion_id = st.selectbox(
                "Seleccione el ID de Registro Interno exacto para Gestionar/Modificar:", 
                options=df_filtrado['id'].tolist(),
                key="crud_select_user"
            )
            
            # --- LA SOLUCIÓN AL BUG ---
            # Extraemos el DataFrame final que coincide exactamente con el ID seleccionado
            df_registro = df_crud[df_crud['id'] == str(seleccion_id)]
            
            # Validamos estrictamente si el DataFrame tiene contenido antes de aplicar .iloc[0]
            if not df_registro.empty:
                registro_exacto = df_registro.iloc[0]
                idx_original = df_registro.index[0]
                
                st.markdown("#### Formulario de Edición del Cliente")
                col_ed1, col_ed2, col_ed3 = st.columns(3)
                
                with col_ed1:
                    nuevo_nombre = st.text_input("Nombre", value=str(registro_exacto['nombre']))
                    nuevo_tel = st.text_input("Teléfono", value=str(registro_exacto['telefono']))
                    # Control de seguridad por si el contrato actual no coincide con las opciones por defecto
                    idx_contrato = ["month-to-month", "one_year", "two_year"].index(registro_exacto['contrato']) if registro_exacto['contrato'] in ["month-to-month", "one_year", "two_year"] else 0
                    nuevo_contrato = st.selectbox("Contrato Actual", ["month-to-month", "one_year", "two_year"], index=idx_contrato)
                
                with col_ed2:
                    nuevo_identificador = st.text_input("ID Cliente", value=str(registro_exacto['identificador']))
                    nuevo_cargo = st.number_input("Cargo Mensual ($)", value=float(registro_exacto['cargo_mensual']))
                    idx_gestion = ["Pendiente", "Contrató Servicio ✔️", "No Aceptó ❌"].index(registro_exacto['venta_cerrada']) if registro_exacto['venta_cerrada'] in ["Pendiente", "Contrató Servicio ✔️", "No Aceptó ❌"] else 0
                    nueva_gestion = st.selectbox("Estatus Gestión Comercial", ["Pendiente", "Contrató Servicio ✔️", "No Aceptó ❌"], index=idx_gestion)

                with col_ed3:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    btn_actualizar = st.button("🔄 Guardar Cambios", use_container_width=True)
                    btn_eliminar = st.button("🗑️ Eliminar Cliente del Sistema", use_container_width=True, type="primary")

                if btn_actualizar:
                    df_crud.at[idx_original, 'nombre'] = nuevo_nombre
                    df_crud.at[idx_original, 'telefono'] = nuevo_tel
                    df_crud.at[idx_original, 'contrato'] = nuevo_contrato
                    df_crud.at[idx_original, 'identificador'] = nuevo_identificador
                    df_crud.at[idx_original, 'cargo_mensual'] = nuevo_cargo
                    df_crud.at[idx_original, 'venta_cerrada'] = nueva_gestion
                    
                    actualizar_fuente_datos(df_crud)
                    st.success("✨ ¡Información actualizada con éxito en la base de datos!")
                    st.rerun()

                if btn_eliminar:
                    df_crud = df_crud.drop(idx_original).reset_index(drop=True)
                    actualizar_fuente_datos(df_crud)
                    st.warning("❌ El registro ha sido eliminado del sistema de manera definitiva.")
                    st.rerun()
            else:
                st.caption("Cargando datos de registro...") # Evita el parpadeo y la caída de la UI
                
        elif query_busqueda:
            st.error("⚠️ No se encontraron clientes con los criterios especificados.")

        st.markdown("---")
        st.markdown("<h3>📋 Vista General del Historial Completo</h3>", unsafe_allow_html=True)
        df_vista = df_crud.copy().iloc[::-1]
        st.dataframe(df_vista, use_container_width=True)
    else:
        st.info("No se registran consultas guardadas en el histórico.")
        
# ==========================================
# TAB 4: ESTADÍSTICAS & DASHBOARDS
# ==========================================

with tab4:
    st.subheader("📈 Cuadro de Mando Operativo & KPIs (Real-Time)")
    df_stats = inicializar_or_obtener_datos()

    if not df_stats.empty:
        # Sanitizar columnas numéricas cruciales
        df_stats['cargo_mensual'] = pd.to_numeric(df_stats.get('cargo_mensual', 0), errors='coerce').fillna(0.0)
        df_stats['antiguedad'] = pd.to_numeric(df_stats.get('antiguedad', 0), errors='coerce').fillna(0.0)

        total_casos = len(df_stats)
        riesgo_alto = len(df_stats[df_stats.get('prediccion') == "Riesgo Alto"]) if 'prediccion' in df_stats.columns else 0
        pct_riesgo = (riesgo_alto / total_casos) if total_casos > 0 else 0
        fuga_financiera = df_stats[df_stats.get('prediccion') == "Riesgo Alto"]['cargo_mensual'].sum() if 'prediccion' in df_stats.columns else 0
        ventas_exitosas = len(df_stats[df_stats.get('venta_cerrada') == "Contrató Servicio ✔️"]) if 'venta_cerrada' in df_stats.columns else 0
        tasa_conversion = (ventas_exitosas / total_casos) if total_casos > 0 else 0

        # Fila 1: KPIs Principales
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
        
        # Fila 2: Distribución de Riesgo e Impacto por Tipo de Contrato
        g1, g2 = st.columns(2)
        
        with g1:
            st.write("🎯 **Volumen de Clientes por Estado de Riesgo**")
            if 'prediccion' in df_stats.columns:
                df_counts = df_stats['prediccion'].value_counts().to_frame()
                st.bar_chart(df_counts, color="#FF4B4B")
            else:
                st.info("Faltan datos de predicciones.")
                
        with g2:
            st.write("💳 **Pérdida de Cartera Mensual por Tipo de Contrato**")
            if 'contrato' in df_stats.columns and 'prediccion' in df_stats.columns:
                df_fuga_contrato = df_stats[df_stats['prediccion'] == "Riesgo Alto"].groupby('contrato')['cargo_mensual'].sum()
                st.bar_chart(df_fuga_contrato, color="#29B5E8")
            else:
                st.info("Faltan datos de contratos para evaluar la fuga.")

        st.markdown("---")
        
        # Fila 3: Tendencia Temporal e Impacto del Tipo de Internet
        g3, g4 = st.columns(2)
        
        with g3:
            st.write("📅 **Evolución del Cargo Mensual Registrado**")
            if 'fecha' in df_stats.columns:
                # Agrupar por fecha limpia (Y-m-d) para ver el comportamiento
                df_stats['fecha_dia'] = df_stats['fecha'].str.slice(0, 10)
                df_linea = df_stats.groupby('fecha_dia')['cargo_mensual'].mean()
                st.area_chart(df_linea, color="#77933C")
            else:
                st.info("Falta columna de fecha temporal.")
                
        with g4:
            st.write("🌐 **Estado de Riesgo según Tipo de Conectividad**")
            if 'internet' in df_stats.columns and 'prediccion' in df_stats.columns:
                # Crear tabla cruzada para ver el riesgo según el internet
                df_pivot = pd.crosstab(df_stats['internet'], df_stats['prediccion'])
                st.bar_chart(df_pivot)
            else:
                st.info("Faltan variables de internet para segmentar.")
                
    else:
        st.info("No hay datos disponibles en el histórico para estructurar los Dashboards.")