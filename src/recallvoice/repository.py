from contextlib import contextmanager
from pathlib import Path
import sqlite3
import uuid


class Repository:
    """One short-lived connection per operation; transactions never span model calls."""

    def __init__(self, filename: Path):
        self.filename = Path(filename)
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > 1:
                raise RuntimeError("Database schema is newer than this application")
            if version == 0:
                db.executescript(
                    "BEGIN IMMEDIATE;\n"
                    + Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
                    + "\nCOMMIT;"
                )

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.filename, timeout=20)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA journal_mode=WAL")
            with db:
                yield db
        finally:
            db.close()

    def new_conversation(self, title="Untitled conversation"):
        key = uuid.uuid4().hex
        with self.connection() as db:
            db.execute(
                "INSERT INTO conversation(key,title) VALUES(?,?)",
                (key, title.strip()[:120] or "Untitled conversation"),
            )
        return key

    def conversations(self):
        with self.connection() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM conversation ORDER BY opened DESC,key"
                )
            ]

    def begin(self, key, conversation, digest):
        if not key or len(key) > 128:
            raise ValueError("Submission ID must contain 1–128 characters")
        with self.connection() as db:
            prior = db.execute("SELECT * FROM attempt WHERE key=?", (key,)).fetchone()
            if prior:
                if (
                    prior["conversation_key"] != conversation
                    or prior["input_digest"] != digest
                ):
                    raise ValueError(
                        "Submission ID was already used for different input"
                    )
                if prior["state"] != "completed":
                    raise ValueError(
                        "This submission is running or failed; use a new submission ID"
                    )
                return dict(prior)
            db.execute(
                "INSERT INTO attempt(key,conversation_key,input_digest,state) VALUES(?,?,?,'running')",
                (key, conversation, digest),
            )
        return None

    def remember_input(self, key, raw, submitted):
        with self.connection() as db:
            db.execute(
                "UPDATE attempt SET recognized=?,submitted=? WHERE key=?",
                (raw, submitted, key),
            )

    def commit_dialogue(self, key, request, reply):
        with self.connection() as db:
            state = db.execute(
                "SELECT state FROM attempt WHERE key=?", (key,)
            ).fetchone()
            if not state or state["state"] != "running":
                raise ValueError("Attempt cannot be committed in its current state")
            db.executemany(
                "INSERT INTO utterance(attempt_key,speaker,body) VALUES(?,?,?)",
                [(key, "user", request), (key, "assistant", reply)],
            )
            db.execute("UPDATE attempt SET state='completed' WHERE key=?", (key,))

    def fail(self, key, error):
        with self.connection() as db:
            db.execute(
                "UPDATE attempt SET state='failed',error=? WHERE key=? AND state='running'",
                (str(error)[:2000], key),
            )

    def dialogue(self, conversation, limit=30):
        with self.connection() as db:
            rows = db.execute(
                "SELECT u.*,a.conversation_key,a.opened FROM utterance u JOIN attempt a ON a.key=u.attempt_key WHERE a.conversation_key=? ORDER BY u.id DESC LIMIT ?",
                (conversation, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def attempt_dialogue(self, key):
        with self.connection() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM utterance WHERE attempt_key=? ORDER BY id", (key,)
                )
            ]

    def sources(self):
        with self.connection() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT u.*,a.conversation_key FROM utterance u JOIN attempt a ON a.key=u.attempt_key ORDER BY u.id"
                )
            ]

    def vectors(self, encoder):
        with self.connection() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT v.*,u.body,u.speaker,a.conversation_key FROM vector v JOIN utterance u ON u.id=v.source JOIN attempt a ON a.key=u.attempt_key WHERE encoder=? ORDER BY source",
                    (encoder,),
                )
            ]

    def put_vectors(self, encoder, records, replace=False):
        with self.connection() as db:
            if replace:
                db.execute("DELETE FROM vector WHERE encoder=?", (encoder,))
            written = 0
            for source, dimensions, payload in records:
                cursor = db.execute(
                    "INSERT OR REPLACE INTO vector(source,encoder,dimensions,payload) SELECT id,?,?,? FROM utterance WHERE id=?",
                    (encoder, dimensions, payload, source),
                )
                written += cursor.rowcount
            return written

    def clear_vectors(self):
        with self.connection() as db:
            db.execute("DELETE FROM vector")

    def delete_source(self, source):
        with self.connection() as db:
            row = db.execute(
                "SELECT attempt_key,speaker FROM utterance WHERE id=?", (source,)
            ).fetchone()
            if row and row["speaker"] == "user":
                db.execute(
                    "UPDATE attempt SET recognized='',submitted='' WHERE key=?",
                    (row["attempt_key"],),
                )
            db.execute("DELETE FROM utterance WHERE id=?", (source,))

    def delete_conversation(self, key):
        with self.connection() as db:
            db.execute("DELETE FROM conversation WHERE key=?", (key,))

    def forget_conversations(self):
        with self.connection() as db:
            db.execute("DELETE FROM conversation")

    def observe(self, attempt, stage, duration, outcome, detail=""):
        with self.connection() as db:
            db.execute(
                "INSERT INTO observation(attempt_key,stage,elapsed_ms,outcome,detail) VALUES(?,?,?,?,?)",
                (attempt, stage, duration, outcome, detail[:1000]),
            )

    def observations(self, attempt=None):
        with self.connection() as db:
            sql = "SELECT * FROM observation"
            params = ()
            if attempt:
                sql += " WHERE attempt_key=?"
                params = (attempt,)
            return [
                dict(row)
                for row in db.execute(sql + " ORDER BY id DESC LIMIT 300", params)
            ]
