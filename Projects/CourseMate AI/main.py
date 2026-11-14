from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# data = TextLoader("/home/husain_vora/AI_Workspace/Generative-AI/Projects/CourseMate AI/document loaders/notes.txt")
data = PyPDFLoader("/home/husain_vora/AI_Workspace/Generative-AI/Projects/CourseMate AI/document loaders/GRU.pdf")

docs = data.load()
template = ChatPromptTemplate.from_messages([
    ("system",
     """
    You are a AI that summarizes the text
    """),
    ("human",
     """
    {data}
    """)
])

model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")

prompt = template.format_messages(data = docs)

result = model.invoke(prompt)

print(result.text)
