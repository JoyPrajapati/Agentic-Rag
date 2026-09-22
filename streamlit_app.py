"""Streamlit UI for the Enterprise Knowledge Assistant.

Run with: streamlit run streamlit_app.py
Or launch together with the API using: python run.py
"""

import uuid

import httpx
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Enterprise Knowledge Assistant")
st.caption("Agentic RAG over financial 10-K filings")

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.write(f"**Session ID:** `{st.session_state.user_id[:8]}...`")
    st.write(f"**API:** `{API_URL}`")

    # Health check
    try:
        health = httpx.get(f"{API_URL}/v1/health", timeout=5.0)
        if health.status_code == 200:
            st.success("✅ API is healthy")
        else:
            st.error(f"❌ API returned {health.status_code}")
    except Exception as e:
        st.error(f"❌ API unreachable: {e}")

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.user_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.caption(
        "**Architecture**\n\n"
        "Planner → Retriever → Quality Gate → Synthesizer → Critique\n\n"
        "Hybrid retrieval (dense + BM25) + cross-encoder reranking."
    )

# ------------------------------------------------------------------
# Conversation history
# ------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"📎 Sources: {', '.join(msg['sources'])}")
        if msg.get("confidence") is not None:
            st.caption(f"🎯 Confidence: {msg['confidence']:.2f}")

# ------------------------------------------------------------------
# Chat input
# ------------------------------------------------------------------
query = st.chat_input("Ask about the 10-K filings...")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Thinking through the agent pipeline..."):
            try:
                resp = httpx.post(
                    f"{API_URL}/v1/chat",
                    json={"query": query, "user_id": st.session_state.user_id},
                    timeout=180.0,  # 3 minutes — full pipeline can be slow
                )
                resp.raise_for_status()
                data = resp.json()

                answer = data.get("response", "No response")
                sources = [s["doc_id"] for s in data.get("sources", [])]
                confidence = data.get("confidence")

                st.markdown(answer)
                if sources:
                    st.caption(f"📎 Sources: {', '.join(sources)}")
                if confidence is not None:
                    st.caption(f"🎯 Confidence: {confidence:.2f}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "confidence": confidence,
                })

            except httpx.HTTPStatusError as e:
                st.error(f"API error {e.response.status_code}: {e.response.text}")
            except httpx.TimeoutException:
                st.error("⏱️ Request timed out. The full pipeline takes 30-90s.")
            except Exception as e:
                st.error(f"❌ Error: {e}")