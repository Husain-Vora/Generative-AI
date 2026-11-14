from langchain_community.document_loaders import TextLoader

data = TextLoader("/home/husain_vora/AI_Workspace/Generative-AI/Projects/CourseMate AI/document loaders/notes.txt")

docs = data.load()
print(len(docs))
