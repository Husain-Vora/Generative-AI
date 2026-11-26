from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

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

