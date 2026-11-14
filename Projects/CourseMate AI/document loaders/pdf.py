from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("/home/husain_vora/AI_Workspace/Generative-AI/Projects/CourseMate AI/document loaders/GRU.pdf")

docs = data.load()
print(docs[14])
