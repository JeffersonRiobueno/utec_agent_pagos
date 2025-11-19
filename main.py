import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

SYSTEM_PROMPT = (
    """Eres Maria una asistente de la tienda Tu Tiendita.com, tu función es resolver preguntas o dudas relacionado a las formas de pago de la tienda 

[Formas de Pago]
- Yape, plin o con tarjeta desde la pagina web
- Pagos contra entrega solo en Lima
- Los envios a provincia se realizan previo pago
- Solo se hace envio dentro de Peru

[Yape/Plim]
- Los pagos por Yape o Plin se deben realizar al numero 999 999 999 a nombre de Juanito Perez
- Enviar el comprobante de pago por whatsapp al mismo numero
- Una vez realizado el pago enviar el comprobante por whatsapp

"""
)

app = FastAPI()

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # openai | ollama | gemini
DEFAULT_MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")          # por proveedor
DEFAULT_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.2"))

# (Opcional) URLs/keys por proveedor
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # requerido si usas gemini

class ProductAgentRequest(BaseModel):
    text: str
    provider: Optional[str] = DEFAULT_PROVIDER
    model: Optional[str] = DEFAULT_MODEL
    temperature: Optional[float] = DEFAULT_TEMPERATURE

class ProductAgentResponse(BaseModel):
    result: str



# =========================
# Fábrica de LLMs
# =========================
def make_llm(
    provider: str,
    model: str,
    temperature: float
):
    provider = (provider or DEFAULT_PROVIDER).lower()

    if provider == "openai":
        # Requiere: OPENAI_API_KEY
        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "ollama":
        # Requiere: Ollama corriendo localmente o remoto
        # Modelos típicos: "llama3.1", "qwen2.5", "phi3", etc.
        return ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=temperature)

    if provider == "gemini":
        # Requiere: GOOGLE_API_KEY
        if not GOOGLE_API_KEY:
            raise RuntimeError("Falta GOOGLE_API_KEY para usar Gemini.")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=GOOGLE_API_KEY)

    raise ValueError(f"Proveedor LLM no soportado: {provider}. Usa: openai | ollama | gemini")

@app.post("/payment_agent", response_model=ProductAgentResponse)
def payment_agent_endpoint(req: ProductAgentRequest):

    llm = make_llm(req.provider, req.model, req.temperature)
    tools = []
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}")
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # Ejecuta el agente de forma completamente automática
    result = executor.invoke({"input": req.text})
    # El resultado puede estar en diferentes campos según el modelo
    if isinstance(result, dict) and "output" in result:
        return ProductAgentResponse(result=str(result["output"]))
    return ProductAgentResponse(result=str(result))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
