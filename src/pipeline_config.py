# src/pipeline_config.py
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb

def crear_caracteristicas_negocio(X):
    """
    Función de ingeniería de características basada en tu análisis estratégico.
    Se ejecuta automáticamente dentro del Pipeline.
    """
    X_out = X.copy()
    
    # 1. Segmentación por tipo de servicio (Mapeo lógico)
    has_internet = X_out['internet_service'] != 'no_contract'
    has_phone = X_out['multiple_lines'] != 'no_contract'
    
    X_out['service_type'] = np.select(
        condlist=[(has_internet & has_phone), (has_internet & ~has_phone), (~has_internet & has_phone)],
        choicelist=['both_services', 'internet_only', 'phone_only'],
        default='no_services'
    )
    
    # 2. Segmentación por antigüedad
    X_out['customer_segment'] = np.select(
        condlist=[(X_out['months_of_age'] <= 6), (X_out['months_of_age'] <= 24)],
        choicelist=['new_customer', 'established'],
        default='loyal_customer'
    )
    
    return X_out

# Convertimos la función en un transformador compatible con Scikit-Learn
feature_engineering_transformer = FunctionTransformer(crear_caracteristicas_negocio)

# Definición de columnas para el ColumnTransformer
num_cols = ['monthly_charges', 'total_charges', 'months_of_age', 'month_registration', 'quarter_registration']
cat_cols = ['type', 'payment_method', 'multiple_lines', 'internet_service', 
            'online_security', 'online_backup', 'device_protection', 'tech_support', 
            'streaming_tv', 'streaming_movies', 'paperless_billing', 'partner', 
            'dependents', 'gender', 'service_type', 'customer_segment']

# Preprocesador central paralelo
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
    ]
)

# Pipeline Maestro Integrado (Mismo orden para entrenamiento y producción)
pipeline_final = Pipeline(steps=[
    ('feature_engineering', feature_engineering_transformer),
    ('preprocessor', preprocessor),
    ('classifier', lgb.LGBMClassifier(
        random_state=12345,
        n_estimators=20,
        class_weight='balanced',
        verbose=-1
    ))
])