from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id = "openai-community/gpt2",
    task = "text-generation",
    pipeline_kwargs = {
        "max_new_tokens": 512,
        "do_sample": False,
        "repetition_penalty": 1.03,
    }
)

chat_model = ChatHuggingFace(llm=llm)

result = chat_model.invoke("what is Data Science?")

print(result.content)
