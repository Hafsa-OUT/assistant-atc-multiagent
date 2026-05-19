"""
Outils RAG multi-corpus ATC.
"""
from pathlib import Path
from langchain_core.tools import Tool
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings

default_llm = OllamaLLM(model="llama3")
embeddings = OllamaEmbeddings(model="llama3")

BASE_DIR = Path(__file__).parent.parent
CORPUS_PROCEDURES_PATH = BASE_DIR / "data" / "corpus_procedures"
CORPUS_URGENCES_PATH   = BASE_DIR / "data" / "corpus_urgences"


def build_vectorstore(corpus_path: Path, store_name: str) -> FAISS:
    save_path = BASE_DIR / "data" / f"{store_name}_faiss"

    if save_path.exists():
        print(f"[RAG] Chargement: {store_name}")
        return FAISS.load_local(
            str(save_path), embeddings,
            allow_dangerous_deserialization=True
        )

    print(f"[RAG] Construction: {store_name}")
    loader = DirectoryLoader(
        str(corpus_path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(save_path))
    return vectorstore


def build_retriever_tool(corpus_path, store_name, tool_name, description):
    vectorstore = build_vectorstore(corpus_path, store_name)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})

    def run_retrieval(query: str) -> str:
        docs = retriever.invoke(query)
        if not docs:
            return "Aucun document pertinent trouvé."
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt  = f"Question: {query}\n\nContexte:\n{context}\n\nRéponds en français."
        answer  = default_llm.invoke(prompt)
        sources = [doc.metadata.get("source", "inconnu") for doc in docs]
        return f"{answer}\n\nSources: {', '.join(set(sources))}"

    return Tool(
        name=tool_name,
        func=run_retrieval,
        description=description,
    )


def get_rag_tools() -> list:
    tools = []

    try:
        tools.append(build_retriever_tool(
            corpus_path=CORPUS_PROCEDURES_PATH,
            store_name="procedures",
            tool_name="search_procedures",
            description="Procédures ATC: séparation aéronefs, phraséologie OACI, espaces aériens.",
        ))
    except Exception as e:
        print(f"[WARNING] Corpus procédures: {e}")

    try:
        tools.append(build_retriever_tool(
            corpus_path=CORPUS_URGENCES_PATH,
            store_name="urgences",
            tool_name="search_urgences",
            description="Urgences ATC: MAYDAY, PAN-PAN, NORDO, windshear, météo dangereuse.",
        ))
    except Exception as e:
        print(f"[WARNING] Corpus urgences: {e}")

    return tools