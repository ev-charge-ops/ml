from dataclasses import dataclass


@dataclass(frozen=True)
class RawFile:
    name: str
    url: str
    md5: str


NORWAY_RECORD = "https://zenodo.org/api/records/13896176/files"
TURKU_RECORD = "https://zenodo.org/api/records/5721233/files"

NORWAY_SESSIONS = RawFile(
    name="norway_charging_reports.csv",
    url=f"{NORWAY_RECORD}/Dataset1_charging_reports.csv/content",
    md5="8bb7463d760cc846271b1ed9bdf808fd",
)
NORWAY_SESSION_PREDICTIONS = RawFile(
    name="norway_session_predictions.csv",
    url=f"{NORWAY_RECORD}/Dataset3_session_predictions.csv/content",
    md5="df82be864494c37f727e19215c62c235",
)
TURKU_SESSIONS = RawFile(
    name="turku_public_charging_2019.csv",
    url=f"{TURKU_RECORD}/Lataustapahtumat,%20julkiset%20latauslaitteet%202019.csv/content",
    md5="0e6a0741dca0e0631813e69c6b55cecd",
)

RAW_FILES = (NORWAY_SESSIONS, NORWAY_SESSION_PREDICTIONS, TURKU_SESSIONS)
