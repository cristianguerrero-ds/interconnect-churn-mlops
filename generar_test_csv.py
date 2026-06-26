import pandas as pd
import random

# Configurar semilla para reproducibilidad
random.seed(42)

# Listas de datos realistas para la simulación de Interconnect
nombres_ficticios = [
    "Juan Carlos Pérez", "María Camila Restrepo", "Andrés Felipe Guerrero", "Diana Marcela Silva",
    "Luis Eduardo Gómez", "Ana Sofía Patiño", "Carlos Mario Aristizábal", "Laura Valentina Becerra",
    "Jorge Eliécer Torres", "Paula Andrea Mendoza", "Gustavo Adolfo Rincón", "Sandra Milena Duarte",
    "oscar Iván Castellanos", "Martha Cecilia Rojas", "santiago Alejandro Suárez", "Angela María Martínez",
    "Ricardo Antonio Franco", "Claudia Patricia Ortiz", "Nelson Enrique Benítez", "Gloria Inés Cardona"
]

servicios_internet = ["Fiber optic", "DSL", "no_contract"]
tipos_contrato = ["month_to_month", "one_year", "two_year"]
metodos_pago = ["electronic_check", "mailed_check", "bank_transfer_automatic", "credit_card_automatic"]
opciones_si_no = ["yes", "no"]

datos_20_clientes = []

for i in range(20):
    # Generar un ID con el formato real de Interconnect (ej. 1234-ABCD)
    customer_id = f"{random.randint(1000, 9999)}-{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}{chr(random.randint(65, 90))}"
    
    # Generar un número celular ficticio de Colombia
    telefono = f"3{random.randint(10, 25)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
    
    # Simular una lógica de antigüedad según el tipo de contrato para consistencia de datos
    tipo_c = random.choice(tipos_contrato)
    if tipo_c == "month_to_month":
        months_of_age = float(random.randint(1, 18))
    elif tipo_c == "one_year":
        months_of_age = float(random.randint(12, 36))
    else:
        months_of_age = float(random.randint(24, 72))

    internet = random.choice(servicios_internet)
    
    # Regla: Si no contrata internet, las variables de servicios digitales van en "no_contract"
    if internet == "no_contract":
        sec, bck, prot, sup, tv, mov = "no_contract", "no_contract", "no_contract", "no_contract", "no_contract", "no_contract"
    else:
        sec = random.choice(opciones_si_no)
        bck = random.choice(opciones_si_no)
        prot = random.choice(opciones_si_no)
        sup = random.choice(opciones_si_no)
        tv = random.choice(opciones_si_no)
        mov = random.choice(opciones_si_no)

    registro = {
        "customer_id": customer_id,
        "nombre": nombres_ficticios[i],
        "telefono": telefono,
        "gender": random.choice(["male", "female"]),
        "senior_citizen": random.choice([0, 1]),
        "partner": random.choice(opciones_si_no),
        "dependents": random.choice(opciones_si_no),
        "months_of_age": months_of_age,
        "internet_service": internet,
        "online_security": sec,
        "online_backup": bck,
        "device_protection": prot,
        "tech_support": sup,
        "multiple_lines": random.choice(["yes", "no", "no_contract"]),
        "streaming_tv": tv,
        "streaming_movies": mov,
        "type": tipo_c,
        "paperless_billing": random.choice(opciones_si_no),
        "payment_method": random.choice(metodos_pago)
    }
    datos_20_clientes.append(registro)

# Crear DataFrame y exportar a CSV
df_test = pd.DataFrame(datos_20_clientes)
df_test.to_csv("clientes_prueba.csv", index=False)

print("✅ Archivo 'clientes_prueba.csv' generado con éxito con 20 registros.")