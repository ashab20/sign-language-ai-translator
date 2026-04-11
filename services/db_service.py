from utils.database import Base, engine
from model.gesture_data import GestureData

def initialize_database():
    """
    Ensure the database and tables are created.
    Call this once at startup.
    """
    print("🔧 Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created (if not already present).")
