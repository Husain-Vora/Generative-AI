import arxiv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class ArxivRetriever(BaseRetriever):
    load_max_docs: int = 3
    load_all_available_meta: bool = True

    def _get_relevant_documents(self, query, *, run_manager=None):
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=self.load_max_docs)
        return [
            Document(
                page_content=r.summary,
                metadata={
                    "Title": r.title,
                    "Authors": ", ".join(a.name for a in r.authors),
                    "Published": r.published.date().isoformat(),
                    "url": r.entry_id,
                },
            )
            for r in client.results(search)
        ]


# Create a retriever
retriever = ArxivRetriever(
    load_max_docs=3,
    load_all_available_meta=True
)


# query arxiv
docs = retriever.invoke("Large Language Models")

# print results
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print("Title:", doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    print("Summary:", doc.page_content)
