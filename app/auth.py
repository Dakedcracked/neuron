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
import secrets
NEURON_SECRET_KEY = os.environ.get("NEURON_SECRET_KEY")
if not NEURON_SECRET_KEY:
    SECRET_KEY = secrets.token_hex(32)
    print("⚠️ WARNING: NEURON_SECRET_KEY is not set. A random session-based key has been generated. Logins will expire on restart.")
else:
    SECRET_KEY = NEURON_SECRET_KEY

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
    Loads username and password from environment variables if present.
    """
    existing = db.query(User).first()
    if not existing:
        admin_username = os.environ.get("NEURON_ADMIN_USERNAME", "admin")
        admin_password = os.environ.get("NEURON_ADMIN_PASSWORD")
        
        if not admin_password:
            admin_password = "neuron2026"
            print("⚠️ WARNING: NEURON_ADMIN_PASSWORD is not set. Seeding default admin user with password 'neuron2026'.")
        else:
            print("✓ Seeding default admin user with password from environment.")
            
        admin = User(
            username=admin_username,
            hashed_password=hash_password(admin_password),
            role="admin",
            is_active=True,
        )
        db.add(admin)
        db.commit()
        print(f"✓ Default admin user '{admin_username}' seeded successfully.")


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
