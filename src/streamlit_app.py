import sys
from pathlib import Path

# Ensure src/ is on the path
sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st
from waystone_core import STATE, answer


# ------------------
# Page config
# ------------------
st.set_page_config(
    page_title="🍺 The Waystone Companion",
    layout="centered",
)

# ------------------
# Sidebar
# ------------------
with st.sidebar:
    st.title("🍺 Settings")

    STATE["chapter"] = st.number_input(
        "Spoiler boundary (chapter)",
        min_value=0,
        max_value=200,
        value=int(STATE.get("chapter", 9)),
        step=1,
    )

    STATE["model"] = st.selectbox(
        "Ollama model",
        options=["qwen2.5:14b", "llama3.1:8b", "mistral", "llama3.2:3b"],
        index=0,  # Makes qwen2.5:14b the default
    )

    STATE["debug"] = st.checkbox(
        "Debug",
        value=bool(STATE.get("debug", False)),
    )

    if st.button("Clear chat"):
        st.session_state.messages = []

# ------------------
# Header
# ------------------
st.title("🍺 The Waystone Companion")
st.caption(
    f"Book: {STATE['book']} · "
    f"Chapter limit: {STATE['chapter']} · "
    f"Model: {STATE['model']}"
)

# ------------------
# Session state
# ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ------------------
# Chat input
# ------------------
prompt = st.chat_input("Ask a question…")

if prompt:
    # User message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            out = answer(
                question=prompt,
                current_chapter=int(STATE["chapter"]),
                model=str(STATE["model"]),
            )

            # Main answer
            st.markdown(out["answer"])

            # Sources (collapsed)
            if out.get("sources"):
                with st.expander("Show sources"):
                    for s in out["sources"]:
                        st.markdown(
                            f"- **{s['chunk_id']}** "
                            f"(Chapter {s['chapter']}, score {s['score']:.3f})"
                        )

            # Debug (optional)
            if STATE["debug"] and out.get("debug"):
                st.subheader("Debug")
                st.json(out["debug"])

    # Persist assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": out["answer"]}
    )