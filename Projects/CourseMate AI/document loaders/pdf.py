from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("/home/husain_vora/AI_Workspace/Generative-AI/Projects/CourseMate AI/document loaders/deeplearning.pdf")

docs = data.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 10,
)

chunks = splitter.split_documents(docs)

print(len(chunks))
