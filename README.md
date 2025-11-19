# Agente de Consultas sobre Formas de Pago

Agente desarrollado para resolver consultas sobre formas de pago de la tienda Tu Tiendita.com. Compatible con Gemini, OpenAI y Ollama.

---

## Descripción

Este agente utiliza IA para responder preguntas relacionadas con las formas de pago disponibles en la tienda. Incluye información sobre pagos con Yape, Plin, tarjeta, pagos contra entrega, y políticas de envío.

## Formas de Pago Soportadas

- Yape, Plin o con tarjeta desde la página web
- Pagos contra entrega solo en Lima
- Los envíos a provincia se realizan previo pago
- Solo se hace envío dentro de Perú

Para Yape/Plin:
- Pago al número 999 999 999 a nombre de Juanito Perez
- Enviar comprobante de pago por WhatsApp al mismo número

---

## Instalación y Configuración

1. **Crea y activa el entorno virtual de Python:**
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Instala dependencias:**
   ```
   pip install -r requirements.txt
   ```

3. **Configura el archivo `.env`:**
   - `LLM_PROVIDER`: openai, ollama, o gemini (por defecto: openai)
   - `MODEL_NAME`: Nombre del modelo (por defecto: gpt-4o-mini)
   - `MODEL_TEMPERATURE`: Temperatura del modelo (por defecto: 0.2)
   - Para OpenAI: `OPENAI_API_KEY`
   - Para Gemini: `GOOGLE_API_KEY`
   - Para Ollama: `OLLAMA_BASE_URL` (por defecto: http://localhost:11434)

4. **Ejecuta el agente:**
   ```
   python3 main.py
   ```

La API estará disponible en `http://localhost:8000`.

---

## Uso de la API

Envía una solicitud POST a `/payment_agent_search` con el siguiente JSON:

```json
{
  "text": "¿Cuáles son las formas de pago disponibles?",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "temperature": 0.2
}
```

Respuesta:

```json
{
  "result": "Respuesta del agente..."
}
```

---

## Notas

- Asegúrate de tener las claves API configuradas para el proveedor seleccionado.
- Para Ollama, asegúrate de que el servidor esté corriendo localmente.

---

## Estructura del Proyecto

- `main.py`: API principal del agente de pagos.
- `requirements.txt`: Dependencias de Python.
- `Dockerfile`: Para contenerización.
- `docker-compose.yml`: Configuración de Docker.

---
