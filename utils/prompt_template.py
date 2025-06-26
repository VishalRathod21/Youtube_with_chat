from langchain_core.prompts import PromptTemplate

def get_prompt_template():
    template = """
    Answer the following question using ONLY the context provided.
    If the context is not enough, just say \"I don't know\".

    Context:
    {context}

    Question: {question}
    """
    return PromptTemplate.from_template(template)