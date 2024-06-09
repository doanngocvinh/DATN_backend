from sqlalchemy import Column, String, ARRAY
from sqlalchemy.dialects.postgresql import UUID
import uuid
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class UserModel(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, nullable=False, unique=True)
    password = Column(String, nullable=True)
    projects = Column(ARRAY(String), nullable=True)  # Storing list of strings
