from dotenv import load_dotenv
load_dotenv()

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

embeddings = NVIDIAEmbeddings(
    model="nvidia/nemotron-3.5-embed-8b",
    dimensions=64
)

texts = [
    "This film was great",
    "The weather is cold"
]

vector = embeddings.embed_documents(texts)

print(vector)
