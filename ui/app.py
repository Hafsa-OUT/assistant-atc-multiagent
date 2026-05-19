"""
Interface Web Streamlit - Assistant ATC Multi-Agent
Lancer avec: streamlit run ui/app.py
"""
import uuid
import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.workflow import run_agent, resume_after_human

# ── Configuration page ──────────────────────────────
st.set_page_config(
    page_title="Assistant ATC Multi-Agent",
    page_icon="✈️",
    layout="centered",
)

st.title("✈️ Assistant ATC Multi-Agent")
st.caption("Système multi-agent LangGraph | RAG multi-corpus | Human-in-the-loop")
st.divider()

# ── Session ─────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_human" not in st.session_state:
    st.session_state.awaiting_human = False
if "pending_analysis" not in st.session_state:
    st.session_state.pending_analysis = ""

# ── Historique ──────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Human-in-the-Loop ───────────────────────────────
if st.session_state.awaiting_human:
    st.warning("⚠️ **Validation superviseur requise** avant de continuer")

    with st.expander("📋 Situation à valider", expanded=True):
        st.markdown(st.session_state.pending_analysis)

    feedback = st.text_area(
        "Commentaire du superviseur (optionnel)",
        placeholder="Ajoutez vos remarques ici...",
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Approuver", use_container_width=True, type="primary"):
            with st.spinner("Reprise en cours..."):
                try:
                    final_state = resume_after_human(
                        thread_id=st.session_state.thread_id,
                        approved=True,
                        feedback=feedback,
                    )
                    st.session_state.awaiting_human = False
                    answer = final_state.get("final_answer", "Erreur.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"✅ *Validé par superviseur humain*\n\n{answer}",
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur: {e}")

    with col2:
        if st.button("❌ Refuser", use_container_width=True):
            st.session_state.awaiting_human = False
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    "❌ Situation refusée par le superviseur.\n\n"
                    f"**Raison**: {feedback or 'Non spécifiée.'}\n\n"
                    "Veuillez contacter directement les autorités compétentes."
                ),
            })
            st.rerun()

    st.stop()

# ── Input utilisateur ───────────────────────────────
user_input = st.chat_input("Posez votre question ATC...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Recherche en cours..."):
            try:
                final_state = run_agent(
                    question=user_input,
                    thread_id=st.session_state.thread_id,
                )

                final_answer = final_state.get("final_answer", "")

                if final_answer == "__AWAITING_HUMAN_VALIDATION__":
                    st.session_state.awaiting_human = True
                    st.session_state.pending_analysis = final_state.get(
                        "analysis_result", "Analyse non disponible."
                    )
                    st.warning("⚠️ Validation superviseur requise.")
                    st.rerun()

                else:
                    needs_human = final_state.get("needs_human_validation", False)
                    approved = final_state.get("human_approved", False)

                    if needs_human and approved:
                        badge = "✅ *Validé par superviseur humain*\n\n"
                    else:
                        badge = "🤖 *Réponse générée automatiquement*\n\n"

                    full_response = badge + final_answer
                    st.markdown(full_response)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                    })

                    with st.expander("🔍 Détails du processus"):
                        st.markdown("**Résultat RAG (Researcher)**")
                        st.info(final_state.get("research_result", "—"))
                        st.markdown("**Analyse (Analyst)**")
                        st.info(final_state.get("analysis_result", "—"))

            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")

# ── Sidebar ─────────────────────────────────────────
with st.sidebar:
    st.header("✈️ Assistant ATC")
    st.markdown("""
    Système multi-agent pour contrôleurs aériens:
    - 📚 **RAG multi-corpus** (procédures + urgences)
    - 🧠 **LangGraph** orchestration
    - 👤 **Human-in-the-loop** situations critiques
    """)

    st.divider()
    st.markdown("**Exemples de questions:**")
    st.code("Séparation minimale en approche ILS ?")
    st.code("Procédure MAYDAY complète ?")
    st.code("Que faire en cas de windshear ?")
    st.code("Squawk codes d'urgence ?")

    st.divider()
    if st.button("🔄 Nouvelle session", use_container_width=True):
        st.session_state.clear()
        st.rerun()