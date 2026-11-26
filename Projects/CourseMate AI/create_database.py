# load pdf
# split into chunks
# create the embeddings
# store into chroma db
import warnings
warnings.filterwarnings("ignore", message=".*langchain-community.*")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("document loaders/deeplearning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)
chunks = [c for c in chunks if c.page_content.strip()]   # drop empty chunks

embedding_model = HuggingFaceEmbeddings()

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print(f"Stored {len(chunks)} chunks in chroma_db")
