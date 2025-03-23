from app.models.models import Base
from sqlalchemy import create_engine

# Create database engine
engine = create_engine('sqlite:///app/database/app.db')

def init_db():
    Base.metadata.create_all(engine)

if __name__ == '__main__':
    init_db()
    print("Database initialized successfully!") 
