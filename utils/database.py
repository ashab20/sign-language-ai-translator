import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

_sqlite_rel = os.getenv("SQLITE_PATH", "data/sign_ai.db")
_sqlite_path = Path(_sqlite_rel)
if not _sqlite_path.is_absolute():
    _sqlite_path = ROOT / _sqlite_path
_sqlite_path.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{_sqlite_path}"
engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine)
Session = scoped_session(SessionLocal)
print(f"SQLite ready at {_sqlite_path}")

Base = declarative_base()

from model.gesture_data import GestureData  # noqa: E402

Base.metadata.create_all(bind=engine)
print("Table 'gestures' ready (if not exists).")
