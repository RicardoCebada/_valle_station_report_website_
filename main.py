import os
import json
import requests
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import markdown # Nueva librería para convertir a HTML
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Configuración de credenciales mediante variables de entorno (Crucial para GitHub Actions)
THINGSPEAK_CHANNEL_ID = os.environ.get("THINGSPEAK_CHANNEL_ID")
THINGSPEAK_READ_KEY = os.environ.get("THINGSPEAK_READ_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

def obtener_datos_historicos():
    url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json?api_key={THINGSPEAK_READ_KEY}&days=60&average=60"
    respuesta = requests.get(url)
    if respuesta.status_code != 200: return "Error"
        
    datos = respuesta.json()
    lecturas = []
    for entrada in datos.get("feeds", []):
        if entrada.get("field1") is None: continue
        lecturas.append({
            "tiempo": entrada.get("created_at"),
            "presion": entrada.get("field1"),
            "temp_sensor1": entrada.get("field2"),
            "humedad": entrada.get("field3"),
            "temp_sensor2": entrada.get("field4"),
            "radiacion_vis": entrada.get("field5"),
            "radiacion_ir": entrada.get("field6"),
            "radiacion_uv": entrada.get("field7"),
            "temp_sensor3": entrada.get("field8")
        })
    return json.dumps(lecturas, indent=2)

def generar_graficos(datos_json_str, directorio):
    datos = json.loads(datos_json_str)
    df = pd.DataFrame(datos)
    df['tiempo'] = pd.to_datetime(df['tiempo'])
    for col in ['temp_sensor1', 'temp_sensor2', 'temp_sensor3', 'humedad']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Gráfico Temperaturas
    plt.figure(figsize=(10, 5))
    plt.plot(df['tiempo'], df['temp_sensor1'], label='Sensor 1')
    plt.plot(df['tiempo'], df['temp_sensor2'], label='Sensor 2')
    plt.plot(df['tiempo'], df['temp_sensor3'], label='Sensor 3')
    plt.title('Historial de Temperaturas')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(directorio, "grafico_temperaturas.png"))
    plt.close()
    
    # Gráfico Humedad
    plt.figure(figsize=(10, 5))
    plt.plot(df['tiempo'], df['humedad'], label='Humedad Relativa', color='teal')
    plt.title('Historial de Humedad')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(directorio, "grafico_humedad.png"))
    plt.close()

def guardar_reporte_html(contenido, telemetria_cruda):
    """Convierte el contenido a HTML y crea un index.html con diseño."""
    contenido_seguro = "".join([b["text"] for b in contenido if isinstance(b, dict) and "text" in b]) if isinstance(contenido, list) else str(contenido)
    
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    generar_graficos(telemetria_cruda, directorio_actual)
    
    # Armar documento Markdown
    md_text = f"# 📡 Reporte de Anomalías y Tendencias IoT\n"
    md_text += f"**Última actualización:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    md_text += f"**Estaciones:** Zona Tláhuac - Chalco\n\n---\n\n"
    md_text += contenido_seguro
    md_text += "\n\n## Análisis Gráfico de Tendencias\n\n"
    md_text += "![Gráfico de Temperaturas](grafico_temperaturas.png)\n\n"
    md_text += "![Gráfico de Humedad](grafico_humedad.png)\n"
    
    # Convertir a HTML
    html_body = markdown.markdown(md_text)
    
    # Inyectar CSS y estructura web
    html_completo = f"""<!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reporte Meteorológico Tláhuac-Chalco</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 2rem; color: #333; background-color: #f9f9f9; }}
            .container {{ background-color: #fff; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; margin-top: 1rem; }}
            h1, h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 0.5rem; }}
        </style>
    </head>
    <body>
        <div class="container">
            {html_body}
        </div>
    </body>
    </html>"""
    
    ruta_completa = os.path.join(directorio_actual, "index.html")
    with open(ruta_completa, "w", encoding="utf-8") as archivo:
        archivo.write(html_completo)

# 2. Pipeline de ejecución
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un analista de hardware IoT procesando 2 meses de datos. Responde: 1. Día más caluroso y frío. 2. Horas de mayor radiación. 3. Falla en sensores. Usa formato Markdown."),
    ("human", "{datos_telemetria}")
])

agente_analista = prompt | llm

if __name__ == "__main__":
    telemetria = obtener_datos_historicos()
    if telemetria != "Error":
        for _ in range(3):
            try:
                resultado = agente_analista.invoke({"datos_telemetria": telemetria})
                guardar_reporte_html(resultado.content, telemetria)
                print("✅ index.html generado con éxito.")
                break
            except Exception:
                time.sleep(10)