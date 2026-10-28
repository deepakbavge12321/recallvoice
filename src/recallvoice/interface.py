import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import streamlit as st
from .configuration import load_settings
from .factory import build


@st.cache_resource
def services(settings):
    return build(settings)


def source_panel(outcome):
    if outcome.recognized:
        st.write("Recognized:", outcome.recognized)
        st.write("Submitted:", outcome.submitted)
    for notice in outcome.notices:
        st.warning(notice)
    with st.expander("Evidence supplied to the model", expanded=bool(outcome.sources)):
        for hit in outcome.sources:
            st.write(
                f"Source {hit.source_id} · conversation {hit.conversation} · {hit.speaker} · cosine {hit.similarity:.3f}"
            )
            st.write(hit.text)


def vocabulary_page(vocabulary):
    st.subheader("Recognition rules")
    st.caption("Only recorded/uploaded speech is corrected. Terms are literal phrases.")
    rules = vocabulary.rules()
    by_id = {row["id"]: row for row in rules}
    key = st.selectbox(
        "Recognition rule",
        [0, *by_id],
        format_func=lambda i: (
            "New rule" if i == 0 else f"{by_id[i]['heard']} → {by_id[i]['written']}"
        ),
    )
    row = by_id.get(key, {})
    with st.form(f"recognition-{key}"):
        heard = st.text_input("Recognized phrase", row.get("heard", ""))
        written = st.text_input("Canonical spelling", row.get("written", ""))
        if st.form_submit_button("Save recognition rule"):
            try:
                vocabulary.save_rule(heard, written, key or None)
                st.rerun()
            except ValueError as error:
                st.error(str(error))
    if key and st.button("Delete recognition rule"):
        vocabulary.delete_rule(key)
        st.rerun()
    st.subheader("Speech-only preferences")
    st.caption(
        "Optional plain-text respelling, applied only when enabled for a turn. No pronunciation guarantee."
    )
    terms = {row["id"]: row for row in vocabulary.pronunciations()}
    chosen = st.selectbox(
        "Speech preference",
        [0, *terms],
        format_func=lambda i: "New preference" if i == 0 else terms[i]["written"],
    )
    term = terms.get(chosen, {})
    with st.form(f"pronunciation-{chosen}"):
        canonical = st.text_input("Term in replies", term.get("written", ""))
        spoken = st.text_input("Spoken respelling", term.get("spoken", ""))
        if st.form_submit_button("Save speech preference"):
            try:
                vocabulary.save_pronunciation(canonical, spoken, chosen or None)
                st.rerun()
            except ValueError as error:
                st.error(str(error))
    if chosen and st.button("Delete speech preference"):
        vocabulary.delete_pronunciation(chosen)
        st.rerun()


def library_page(agent):
    st.subheader("Stored memory")
    question = st.text_input("Search all conversations")
    if st.button("Search sources") and question.strip():
        try:
            agent.memory.synchronize()
            st.session_state.library_hits = agent.memory.search(question)
        except Exception as error:
            st.error(str(error))
    for hit in st.session_state.get("library_hits", []):
        st.write(
            f"{hit.source_id} · {hit.conversation} · {hit.speaker} · {hit.similarity:.3f}"
        )
        st.write(hit.text)
        if st.button("Delete this source", key=f"delete-{hit.source_id}"):
            agent.repository.delete_source(hit.source_id)
            st.session_state.library_hits = []
            st.session_state.pop("last_turn", None)
            st.rerun()
    if st.button("Rebuild semantic index"):
        try:
            with st.spinner("Computing replacement vectors before committing..."):
                number = agent.memory.synchronize(rebuild=True)
            st.success(f"Rebuilt {number} source vectors")
        except Exception as error:
            st.error(str(error))
    if st.button("Discard vector index only"):
        agent.repository.clear_vectors()
        st.success("Vectors cleared; source conversations are retained")
    confirm = st.checkbox("I want to delete all stored conversations and attempts")
    if st.button("Delete all conversation data", disabled=not confirm):
        agent.repository.forget_conversations()
        st.session_state.clear()
        st.rerun()


def diagnostic_page(agent, project_root):
    st.subheader("Observed stage timings")
    st.caption(
        "Missing stages were not run. Errors are retained with their attempt IDs. Conversation deletion removes associated measurements."
    )
    measurements = agent.repository.observations()
    if measurements:
        st.dataframe(measurements, hide_index=True)
    else:
        st.info("No attempts recorded yet")
    st.subheader("Fixed real-model benchmark")
    st.caption(
        "Independent corpus: source-ID recall, not answer grading. Unit-test doubles are never used here."
    )
    if st.button("Measure source retrieval"):
        root = project_root
        try:
            with st.spinner("Running isolated benchmark process..."):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "recallvoice.benchmark",
                        "--dataset",
                        str(root / "evaluation/corpus.json"),
                        "--output",
                        str(root / "results/semantic.json"),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=360,
                )
            if completed.returncode not in (0, 2):
                raise RuntimeError(completed.stderr[-1200:])
            st.session_state.benchmark = json.loads(
                (root / "results/semantic.json").read_text(encoding="utf-8")
            )
        except Exception as error:
            st.error(str(error))
    report = st.session_state.get("benchmark")
    if report:
        st.metric("Alias fixture accuracy", f"{report['alias_accuracy']:.1%}")
        if report["status"] == "completed":
            st.dataframe(
                [
                    {
                        "Method": "Semantic sources",
                        "Recall@1": report["recall_at_1"],
                        "Recall@5": report["recall_at_5"],
                    },
                    {
                        "Method": "Current conversation only",
                        "Recall@1": report["current_session_only_recall"],
                        "Recall@5": report["current_session_only_recall"],
                    },
                ],
                hide_index=True,
            )
            st.write("Mean query time (ms):", round(report["retrieval_mean_ms"], 2))
        else:
            st.warning(report["error"])
        st.download_button(
            "Download measured JSON",
            json.dumps(report, indent=2),
            file_name="recallvoice-results.json",
        )


def main(project_root=None):
    project_root = Path(project_root or Path.cwd()).resolve()
    st.set_page_config(page_title="RecallVoice", page_icon="🎙", layout="wide")
    st.title("RecallVoice")
    try:
        settings = load_settings(
            os.environ.get("RV_CONFIG") or project_root / "settings.toml"
        )
        agent = services(settings)
    except Exception as error:
        st.error(f"Configuration/storage failure: {error}")
        st.stop()
    with st.sidebar:
        st.caption(f"{settings.chat_model} · Whisper {settings.whisper_device}")
        title = st.text_input("New conversation title", value="Conversation")
        if st.button("Create conversation"):
            st.session_state.active = agent.repository.new_conversation(title)
            st.session_state.pop("last_turn", None)
            st.rerun()
        choices = agent.repository.conversations()
        if not choices:
            st.session_state.active = agent.repository.new_conversation(title)
            choices = agent.repository.conversations()
        ids = [row["key"] for row in choices]
        labels = {row["key"]: row["title"] for row in choices}
        selected = st.selectbox(
            "Conversation",
            ids,
            index=ids.index(st.session_state.get("active"))
            if st.session_state.get("active") in ids
            else 0,
            format_func=labels.get,
        )
        st.session_state.active = selected
        speak = st.toggle("Generate spoken replies", value=True)
        hints = st.checkbox("Apply speech preferences")
        delete = st.checkbox("Confirm deletion of selected conversation")
        if st.button("Delete conversation", disabled=not delete):
            agent.repository.delete_conversation(selected)
            st.session_state.clear()
            st.rerun()
    conversation, vocabulary, library, diagnostics = st.tabs(
        ["Conversation", "Vocabulary", "Memory library", "Diagnostics"]
    )
    with conversation:
        for row in agent.repository.dialogue(selected, limit=500):
            with st.chat_message(row["speaker"]):
                st.write(row["body"])
        last = st.session_state.get("last_turn")
        if last and last[0] == selected:
            source_panel(last[1])
            if last[1].audio:
                st.caption(f"Spoken reply provider: {last[1].audio.provider}")
                st.audio(last[1].audio.data, format=last[1].audio.mime)
        with st.form("compose", clear_on_submit=True):
            typed = st.text_area("Message")
            recorded = st.audio_input("Record speech", sample_rate=16000)
            uploaded = st.file_uploader("Or upload a PCM WAV file", type=["wav"])
            send = st.form_submit_button("Submit turn")
        if send:
            provided = sum(
                [bool(typed.strip()), recorded is not None, uploaded is not None]
            )
            if provided != 1:
                st.error("Provide exactly one text, recording or WAV input")
            else:
                try:
                    with st.spinner("Processing turn..."):
                        audio = recorded or uploaded
                        outcome = agent.submit(
                            selected,
                            uuid.uuid4().hex,
                            text=typed if audio is None else None,
                            wav=audio.getvalue() if audio is not None else None,
                            speak=speak,
                            pronunciation=hints,
                        )
                    st.session_state.last_turn = (selected, outcome)
                    st.rerun()
                except Exception as error:
                    st.error(str(error))
    with vocabulary:
        vocabulary_page(agent.vocabulary)
    with library:
        library_page(agent)
    with diagnostics:
        diagnostic_page(agent, project_root)
