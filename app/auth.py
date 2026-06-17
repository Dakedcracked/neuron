import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base
import bcrypt

from app.database import Base, get_db, engine

# ── JWT Config ──────────────────────────────────────────────────────────────
from app.config import JWT_SECRET, NEURON_ADMIN_USERNAME, NEURON_ADMIN_PASSWORD

SECRET_KEY = JWT_SECRET
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 8

# ── Password Hashing ─────────────────────────────────────────────────────────
# Replaced passlib with direct bcrypt to avoid ValueError on large inputs

# ── Bearer Token Extractor ───────────────────────────────────────────────────
bearer_scheme = HTTPBearer(auto_error=False)


# ── User Model (ORM) ─────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="radiologist")  # radiologist | admin
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ── Helpers ──────────────────────────────────────────────────────────────────
def verify_password(plain: str, hashed: str) -> bool:
    try:
        # bcrypt limits passwords to 72 bytes. Slice to 72 to prevent ValueError.
        return bcrypt.checkpw(plain.encode("utf-8")[:72], hashed.encode("utf-8"))
    except Exception:
        return False


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def seed_default_admin(db: Session):
    """
    Creates the default admin user on first startup if no users exist.
    Loads username and password from configuration properties.
    """
    existing = db.query(User).first()
    if not existing:
        admin = User(
            username=NEURON_ADMIN_USERNAME,
            hashed_password=hash_password(NEURON_ADMIN_PASSWORD),
            role="admin",
            is_active=True,
        )
        db.add(admin)
        db.commit()
        print(f"✓ Default admin user '{NEURON_ADMIN_USERNAME}' seeded successfully.")


# ── Auth Dependency ───────────────────────────────────────────────────────────
def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None:
        raise credentials_exception

    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None or not user.is_active:
        raise credentials_exception
    return user


def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return current_user
