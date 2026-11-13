from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

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

para = input("Give your paragraph: ")

final_prompt = prompt.invoke(
    {"paragraph": para,
     "format_instructions": parser.get_format_instructions()
    }
)
response = model.invoke(final_prompt)

movie = parser.parse(response.text)
print(movie.model_dump_json(indent=2))
