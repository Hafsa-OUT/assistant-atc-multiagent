# Assistant ATC Multi-Agent 

Système multi-agent intelligent d'assistance aux contrôleurs 
de la circulation aérienne, développé avec LangChain et LangGraph.

## Architecture
- **Supervisor** : orchestre le workflow entre les agents
- **Researcher** : recherche RAG dans le bon corpus (procédures ou urgences)
- **Analyst** : analyse les résultats et détecte les situations critiques
- **Responder** : formule la réponse finale opérationnelle

## Tech Stack
- LangChain
- LangGraph
- FAISS
- Ollama (llama3)
- Streamlit
- uv

## Installation
```bash
git clone https://github.com/Hafsa-OUT/assistant-atc-multiagent
cd assistant-atc-multiagent
uv venv
uv sync
```

## Configuration
Créer un fichier `.env` :
## Lancer l'interface
streamlit run ui/app.py

## Structure du projet
├── agents/        # Les 4 agents
├── tools/         # Outils RAG
├── graph/         # Graphe LangGraph
├── prompts/       # Prompts des agents
├── config/        # Configuration LLM
├── data/          # Corpus ATC
└── ui/            # Interface Streamlit

## Auteur
Hafsa Outgadirt — Master SDIA