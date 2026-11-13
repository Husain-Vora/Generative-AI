import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

st.set_page_config(page_title="Movie Information Extractor", page_icon="🎬")

model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")

class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
    Extract Movie information from the paragraph
    {format_instructions}
    """
    ),
    ("human",
     """
    {paragraph}
    """
    )

])

st.title("🎬 Movie Information Extractor")
st.caption("Paste a movie paragraph and get the key information.")

para = st.text_area("Paragraph", height=250, placeholder="Paste your movie paragraph here...")

if st.button("Extract Information", type="primary"):
    if not para.strip():
        st.warning("Please enter a paragraph first.")
    else:
        with st.spinner("Extracting..."):
            final_prompt = prompt.invoke(
                {"paragraph": para,
                 "format_instructions": parser.get_format_instructions()
                }
            )
            response = model.invoke(final_prompt)

            movie = parser.parse(response.text)

        st.subheader("Result")
        st.code(movie.model_dump_json(indent=2), language="json")
