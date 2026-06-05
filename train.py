# train.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from src.pipeline_config import pipeline_final

print("🔄 1. Iniciando carga masiva de datos independientes...")
df_contract = pd.read_csv('data/contract.csv')
df_personal = pd.read_csv('data/personal.csv')
df_internet = pd.read_csv('data/internet.csv')
df_phone = pd.read_csv('data/phone.csv')

# Normalización de nombres de columnas a snake_case
df_contract.columns = ['customer_id', 'begin_date', 'end_date', 'type', 'paperless_billing', 'payment_method', 'monthly_charges', 'total_charges']
df_personal.columns = ['customer_id', 'gender', 'senior_citizen', 'partner', 'dependents']
df_internet.columns = ['customer_id', 'internet_service', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
df_phone.columns = ['customer_id', 'multiple_lines']

print("🔗 2. Ejecutando unificación de fuentes mediante left joins...")
df = df_contract.merge(df_personal, on='customer_id', how='left') \
                .merge(df_internet, on='customer_id', how='left') \
                .merge(df_phone, on='customer_id', how='left')

print("🧹 3. Ejecutando limpieza inicial de tipos de datos...")
df['total_charges'] = pd.to_numeric(df['total_charges'].replace(' ', np.nan), errors='coerce')
df = df.dropna(subset=['total_charges']) # Remoción de 11 clientes sin historial

df['begin_date'] = pd.to_datetime(df['begin_date'])
df['end_date'] = pd.to_datetime(df['end_date'].replace('No', pd.NaT))
df['churned'] = df['end_date'].notna().astype(int)

# Ingeniería de variables temporales base
fecha_referencia = df['begin_date'].max()
df['months_of_age'] = (fecha_referencia - df['begin_date']).dt.days / 30.44
df['month_registration'] = df['begin_date'].dt.month
df['quarter_registration'] = df['begin_date'].dt.quarter

# Relleno consistente de nulos para servicios no contratados
columnas_servicios = ['internet_service', 'online_security', 'online_backup', 'device_protection', 
                      'tech_support', 'streaming_tv', 'streaming_movies', 'multiple_lines']
df[columnas_servicios] = df[columnas_servicios].fillna('no_contract')

# Separación de variables predictoras (Features) y objetivo (Target)
X = df.drop(columns=['customer_id', 'begin_date', 'end_date', 'churned'])
y = df['churned']

# Split 75/25 estratificado para mantener proporción exacta del desbalance (26.5%)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=12345, stratify=y)

print("🏋️‍♂️ 4. Entrenando el Pipeline Maestro con LightGBM...")
pipeline_final.fit(X_train, y_train)

print("💾 5. Exportando artefacto del modelo entrenado...")
joblib.dump(pipeline_final, 'modelo_interconnect.pkl')
print("✅ Proceso de entrenamiento finalizado. Archivo 'modelo_interconnect.pkl' creado con éxito.")