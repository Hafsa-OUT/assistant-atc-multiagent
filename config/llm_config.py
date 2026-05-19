from langchain_community.llms import Ollama

def get_llm(provider="ollama", model="llama3", temperature=0.0):
    return Ollama(model=model, temperature=temperature)

default_llm = Ollama(model="llama3")