from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# data = TextLoader("/home/husain_vora/AI_Workspace/Generative-AI/Projects/CourseMate AI/document loaders/notes.txt")
data = PyPDFLoader("/home/husain_vora/AI_Workspace/Generative-AI/Projects/CourseMate AI/document loaders/deeplearning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)


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
