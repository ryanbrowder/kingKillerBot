import sys
from pathlib import Path

# Ensure src/ is on the path so waystone_core can be imported
sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st
from waystone_core import STATE, retrieve_chunks


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

    k = st.slider(
        "Chunks (k)",
        min_value=3,
        max_value=20,
        value=8,
        step=1,
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
st.caption(f"Book: {STATE['book']} · Chapter limit: {STATE['chapter']}")

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
        with st.spinner("Searching the book…"):
            if STATE["debug"]:
                chunks, dbg = retrieve_chunks(
                    query=prompt,
                    max_chapter=int(STATE["chapter"]),
                    k=int(k),
                    debug=True,
                )
            else:
                chunks = retrieve_chunks(
                    query=prompt,
                    max_chapter=int(STATE["chapter"]),
                    k=int(k),
                    debug=False,
                )

            if not chunks:
                st.markdown(
                    "I didn’t find any passages within your spoiler boundary."
                )
            else:
                # --- Show a minimal, readable result ---
                st.markdown("**Relevant passages found.**")

                # Show top N passages inline
                TOP_N = 2
                for c in chunks[:TOP_N]:
                    st.markdown(
                        f"**{c.get('chunk_id','')}** "
                        f"(Chapter {c.get('chapter')}, Score {c.get('score'):.3f})"
                    )
                    st.markdown(c.get("text", ""))
                    st.divider()

                # Collapse the rest
                if len(chunks) > TOP_N:
                    with st.expander("Show all retrieved passages"):
                        for c in chunks[TOP_N:]:
                            st.markdown(
                                f"**{c.get('chunk_id','')}** "
                                f"(Chapter {c.get('chapter')}, Score {c.get('score'):.3f})"
                            )
                            st.markdown(c.get("text", ""))
                            st.divider()

            # Optional debug output
            if STATE["debug"]:
                st.subheader("Debug")
                st.json(dbg)

    # Lightweight assistant acknowledgement (keeps chat flow natural)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "I found relevant passages from the text. (LLM synthesis coming next.)",
        }
    )