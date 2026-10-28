CREATE TABLE conversation (
    key TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    opened TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE attempt (
    key TEXT PRIMARY KEY,
    conversation_key TEXT NOT NULL REFERENCES conversation(key) ON DELETE CASCADE,
    input_digest TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('running','failed','completed')),
    recognized TEXT NOT NULL DEFAULT '',
    submitted TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    opened TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE utterance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_key TEXT NOT NULL REFERENCES attempt(key) ON DELETE CASCADE,
    speaker TEXT NOT NULL CHECK(speaker IN ('user','assistant')),
    body TEXT NOT NULL,
    UNIQUE(attempt_key,speaker)
);
CREATE TABLE vector (
    source INTEGER NOT NULL REFERENCES utterance(id) ON DELETE CASCADE,
    encoder TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK(dimensions>0),
    payload BLOB NOT NULL,
    PRIMARY KEY(source,encoder)
);
CREATE TABLE recognition_rule (
    id INTEGER PRIMARY KEY,
    heard TEXT NOT NULL COLLATE NOCASE UNIQUE,
    written TEXT NOT NULL
);
CREATE TABLE spoken_term (
    id INTEGER PRIMARY KEY,
    written TEXT NOT NULL COLLATE NOCASE UNIQUE,
    spoken TEXT NOT NULL
);
CREATE TABLE observation (
    id INTEGER PRIMARY KEY,
    attempt_key TEXT NOT NULL REFERENCES attempt(key) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    elapsed_ms REAL NOT NULL CHECK(elapsed_ms>=0),
    outcome TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT ''
);
CREATE INDEX utterance_attempt ON utterance(attempt_key,id);
CREATE INDEX attempt_conversation ON attempt(conversation_key,opened);
PRAGMA user_version=1;
