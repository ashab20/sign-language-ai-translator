"""SQLAlchemy database layer for storing gesture landmark samples."""

import json
import os

from sqlalchemy import Column, Integer, String, Text, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import DB_PATH


class Base(DeclarativeBase):
    pass


class GestureSample(Base):
    __tablename__ = "gesture_samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    label = Column(String(50), nullable=False, index=True)
    landmarks = Column(Text, nullable=False)

    def get_landmarks_array(self) -> list[float]:
        return json.loads(self.landmarks)


_engine = None
_SessionFactory = None


def _get_engine():
    global _engine
    if _engine is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
        Base.metadata.create_all(_engine)
    return _engine


def get_session() -> Session:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=_get_engine())
    return _SessionFactory()


def add_sample(label: str, landmarks: list[float]):
    session = get_session()
    try:
        sample = GestureSample(label=label.upper().strip(), landmarks=json.dumps(landmarks))
        session.add(sample)
        session.commit()
    finally:
        session.close()


def add_samples_bulk(label: str, landmarks_list: list[list[float]]):
    session = get_session()
    try:
        clean_label = label.upper().strip()
        objects = [
            GestureSample(label=clean_label, landmarks=json.dumps(lm))
            for lm in landmarks_list
        ]
        session.add_all(objects)
        session.commit()
    finally:
        session.close()


def get_all_samples() -> list[tuple[str, list[float]]]:
    session = get_session()
    try:
        rows = session.query(GestureSample).all()
        return [(row.label, row.get_landmarks_array()) for row in rows]
    finally:
        session.close()


def get_label_counts() -> dict[str, int]:
    session = get_session()
    try:
        results = (
            session.query(GestureSample.label, func.count(GestureSample.id))
            .group_by(GestureSample.label)
            .order_by(GestureSample.label)
            .all()
        )
        return {label: count for label, count in results}
    finally:
        session.close()


def delete_label(label: str) -> int:
    session = get_session()
    try:
        count = (
            session.query(GestureSample)
            .filter(GestureSample.label == label.upper().strip())
            .delete()
        )
        session.commit()
        return count
    finally:
        session.close()


def delete_all_samples() -> int:
    session = get_session()
    try:
        count = session.query(GestureSample).delete()
        session.commit()
        return count
    finally:
        session.close()


def get_total_sample_count() -> int:
    session = get_session()
    try:
        return session.query(func.count(GestureSample.id)).scalar() or 0
    finally:
        session.close()
