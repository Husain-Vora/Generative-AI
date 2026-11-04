from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq

# from langchain.chat_models import init_chat_model
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
# from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGroq(model = "openai/gpt-oss-120b",
                 temperature=0.9,
                 max_tokens=100)
# model = init_chat_model("groq:openai/gpt-oss-120b")
# model = ChatNVIDIA(model = "mistralai/mistral-large-4-1t-a52b")
# model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite", temperature=0)

response = model.invoke("write a poem on AI")

print(response.content)
