import re
import unicodedata
from .repository import Repository


def cleaned(value):
    value = unicodedata.normalize("NFC", value).strip()
    if (
        not value
        or len(value) > 100
        or any(unicodedata.category(c).startswith("C") for c in value)
    ):
        raise ValueError("Terms must be 1–100 printable characters")
    return value


def substitute(text, mapping):
    """Literal phrases, longest alternatives first, one pass, Unicode adjacency guards."""
    if not mapping:
        return text
    choices = {k.casefold(): v for k, v in mapping.items()}
    alternatives = sorted(choices, key=lambda term: (-len(term), term))
    expression = r"(?<!\w)(?:" + "|".join(map(re.escape, alternatives)) + r")(?!\w)"

    def replacement(match):
        # Python's \w excludes combining marks and some connector characters.
        adjacent = (
            text[max(0, match.start() - 1) : match.start()]
            + text[match.end() : match.end() + 1]
        )
        if any(
            unicodedata.category(char).startswith("M")
            or unicodedata.category(char) == "Pc"
            or char in {"\u200c", "\u200d"}
            for char in adjacent
        ):
            return match[0]
        return choices.get(match[0].casefold(), match[0])

    return re.sub(expression, replacement, text, flags=re.IGNORECASE)


class Vocabulary:
    def __init__(self, repository: Repository):
        self.repository = repository

    def rules(self):
        with self.repository.connection() as db:
            return [
                dict(row)
                for row in db.execute("SELECT * FROM recognition_rule ORDER BY id")
            ]

    def pronunciations(self):
        with self.repository.connection() as db:
            return [
                dict(row) for row in db.execute("SELECT * FROM spoken_term ORDER BY id")
            ]

    def save_rule(self, heard, written, identity=None):
        heard, written = cleaned(heard), cleaned(written)
        with self.repository.connection() as db:
            others = db.execute(
                "SELECT * FROM recognition_rule WHERE id != ?", (identity or -1,)
            ).fetchall()
            for row in others:
                if row["heard"].casefold() == heard.casefold():
                    raise ValueError("Recognized phrase already has a rule")
                if (
                    heard.casefold() == row["written"].casefold()
                    and heard.casefold() != written.casefold()
                ) or (
                    written.casefold() == row["heard"].casefold()
                    and row["heard"].casefold() != row["written"].casefold()
                ):
                    raise ValueError("Rules cannot form recognition chains or cycles")
            if identity is None:
                return db.execute(
                    "INSERT INTO recognition_rule(heard,written) VALUES(?,?)",
                    (heard, written),
                ).lastrowid
            cursor = db.execute(
                "UPDATE recognition_rule SET heard=?,written=? WHERE id=?",
                (heard, written, identity),
            )
            if not cursor.rowcount:
                raise ValueError("Rule no longer exists")
            return identity

    def save_pronunciation(self, written, spoken, identity=None):
        written, spoken = cleaned(written), cleaned(spoken)
        with self.repository.connection() as db:
            for row in db.execute(
                "SELECT * FROM spoken_term WHERE id != ?", (identity or -1,)
            ):
                if row["written"].casefold() == written.casefold():
                    raise ValueError("Term already has a speech preference")
            if identity is None:
                return db.execute(
                    "INSERT INTO spoken_term(written,spoken) VALUES(?,?)",
                    (written, spoken),
                ).lastrowid
            if not db.execute(
                "UPDATE spoken_term SET written=?,spoken=? WHERE id=?",
                (written, spoken, identity),
            ).rowcount:
                raise ValueError("Speech preference no longer exists")
            return identity

    def delete_rule(self, identity):
        with self.repository.connection() as db:
            db.execute("DELETE FROM recognition_rule WHERE id=?", (identity,))

    def delete_pronunciation(self, identity):
        with self.repository.connection() as db:
            db.execute("DELETE FROM spoken_term WHERE id=?", (identity,))

    def correct(self, transcript):
        return substitute(
            transcript, {row["heard"]: row["written"] for row in self.rules()}
        )

    def for_speech(self, answer):
        return substitute(
            answer, {row["written"]: row["spoken"] for row in self.pronunciations()}
        )
