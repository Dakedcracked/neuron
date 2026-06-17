import uuid
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, func, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import DATABASE_URL

# Hardened PostgreSQL engine with enterprise-grade connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_recycle=1800,
    pool_pre_ping=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── ORM Models ────────────────────────────────────────────────────────────────

class ClinicTenant(Base):
    __tablename__ = "clinic_tenants"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class Scan(Base):
    __tablename__ = "scans"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_hash = Column(String, nullable=False, index=True)
    scan_type = Column(String, nullable=False)             # XRAY, CT, MRI
    pathology_detected = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    inference_latency = Column(Float, default=0.0)         # Latency in ms
    pytorch_executed = Column(String, default="false")
    status = Column(String, default="pending")             # pending, completed, failed
    img_base64 = Column(String, nullable=True)             # Visualization mask
    predictions = Column(String, nullable=True)            # JSON string of class probs
    bbox = Column(String, nullable=True)                   # JSON string of bounding box
    priority = Column(String, default="normal")             # normal, high, critical
    s3_url = Column(String, nullable=True)                 # Cloud storage link
    tenant_id = Column(String, ForeignKey("clinic_tenants.id"), nullable=True, index=True)


class ClinicSettings(Base):
    __tablename__ = "clinic_settings"

    key = Column(String, primary_key=True)
    value = Column(String, nullable=False)


# ── Lifecycle ─────────────────────────────────────────────────────────────────

def init_db():
    Base.metadata.create_all(bind=engine)
    _seed_default_settings()


def _seed_default_settings():
    db = SessionLocal()
    try:
      # Seed default tenant
      default_tenant = db.query(ClinicTenant).filter_by(name="Default National Clinic").first()
      if not default_tenant:
          default_tenant = ClinicTenant(id="default-tenant-uuid", name="Default National Clinic")
          db.add(default_tenant)
          db.commit()
          
          # Associate any orphaned scans to the default tenant
          db.query(Scan).filter(Scan.tenant_id == None).update({Scan.tenant_id: "default-tenant-uuid"})
          db.commit()

      defaults = {
          "clinic_name": "Neuron AI Diagnostic Centre",
          "station_id": "B2B-HYD-94",
      }
      for key, value in defaults.items():
          existing = db.query(ClinicSettings).filter_by(key=key).first()
          if not existing:
              db.add(ClinicSettings(key=key, value=value))
      db.commit()
    finally:
      db.close()


def get_db():
    with SessionLocal() as db:
        yield db


# ── Write Operations ──────────────────────────────────────────────────────────

def create_pending_scan(db, patient_hash: str, scan_type: str, s3_url: str, tenant_id: str = None):
    scan = Scan(
        patient_hash=patient_hash,
        scan_type=scan_type,
        pathology_detected="Pending",
        confidence_score=0.0,
        s3_url=s3_url,
        status="pending",
        timestamp=datetime.now(timezone.utc),
        tenant_id=tenant_id
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)
    return scan


def update_scan_result(db, scan_id: str, pathology: str, confidence: float, latency: float, pytorch_exec: bool, img_base64: str, predictions: str, bbox: str, priority: str = "normal"):
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if scan:
        scan.pathology_detected = pathology
        scan.confidence_score = confidence
        scan.inference_latency = latency
        scan.pytorch_executed = str(pytorch_exec).lower()
        scan.img_base64 = img_base64
        scan.predictions = predictions
        scan.bbox = bbox
        scan.priority = priority
        scan.status = "completed"
        db.commit()


def update_clinic_setting(db, key: str, value: str):
    setting = db.query(ClinicSettings).filter_by(key=key).first()
    if setting:
        setting.value = value
        db.commit()
    else:
        db.add(ClinicSettings(key=key, value=value))
        db.commit()


def get_clinic_settings(db) -> dict:
    rows = db.query(ClinicSettings).all()
    return {r.key: r.value for r in rows}


# ── Read Operations ───────────────────────────────────────────────────────────

def get_scan_history(db, page: int = 1, page_size: int = 20, scan_type: str = None, pathology: str = None, tenant_id: str = None):
    """Paginated scan history with latency statistics for accuracy auditing."""
    query = db.query(Scan)
    if tenant_id:
        query = query.filter(Scan.tenant_id == tenant_id)
    if scan_type:
        query = query.filter(Scan.scan_type == scan_type.upper())
    if pathology:
        query = query.filter(Scan.pathology_detected.ilike(f"%{pathology}%"))

    query = query.order_by(Scan.timestamp.desc())
    total = query.count()
    scans = query.offset((page - 1) * page_size).limit(page_size).all()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
        "scans": [
            {
                "id": s.id,
                "patient_hash": s.patient_hash,
                "scan_type": s.scan_type,
                "pathology_detected": s.pathology_detected,
                "confidence_score": s.confidence_score,
                "timestamp": s.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "inference_latency": s.inference_latency,
                "pytorch_executed": s.pytorch_executed,
            }
            for s in scans
        ],
    }


def get_dashboard_metrics(db, tenant_id: str = None):
    total_scans_query = db.query(func.count(Scan.id))
    if tenant_id:
        total_scans_query = total_scans_query.filter(Scan.tenant_id == tenant_id)
    total_scans = total_scans_query.scalar() or 0

    # Scan counts per modality
    modality_counts_query = db.query(Scan.scan_type, func.count(Scan.id))
    if tenant_id:
        modality_counts_query = modality_counts_query.filter(Scan.tenant_id == tenant_id)
    modality_counts = modality_counts_query.group_by(Scan.scan_type).all()
    modality_map = {row[0]: row[1] for row in modality_counts}

    # Average latency per modality for model optimization audits
    modality_latencies_query = db.query(Scan.scan_type, func.avg(Scan.inference_latency))
    if tenant_id:
        modality_latencies_query = modality_latencies_query.filter(Scan.tenant_id == tenant_id)
    modality_latencies = modality_latencies_query.group_by(Scan.scan_type).all()
    modality_latency_map = {row[0]: round(float(row[1]), 2) if row[1] is not None else 0.0 for row in modality_latencies}

    # Pathology counts
    pathology_counts_query = db.query(Scan.pathology_detected, func.count(Scan.id))
    if tenant_id:
        pathology_counts_query = pathology_counts_query.filter(Scan.tenant_id == tenant_id)
    pathology_counts = pathology_counts_query.group_by(Scan.pathology_detected).all()
    pathology_map = {row[0]: row[1] for row in pathology_counts}

    # Compute Positive rate (abnormals / eligible)
    eligible_scans_query = db.query(func.count(Scan.id)).filter(Scan.pathology_detected != "Inconclusive")
    if tenant_id:
        eligible_scans_query = eligible_scans_query.filter(Scan.tenant_id == tenant_id)
    eligible_scans = eligible_scans_query.scalar() or 0

    abnormal_count_query = db.query(func.count(Scan.id)).filter(
        Scan.pathology_detected != "Normal",
        Scan.pathology_detected != "Inconclusive",
    )
    if tenant_id:
        abnormal_count_query = abnormal_count_query.filter(Scan.tenant_id == tenant_id)
    abnormal_count = abnormal_count_query.scalar() or 0
    positive_rate = round((abnormal_count / eligible_scans * 100), 1) if eligible_scans > 0 else 0.0

    recent_scans_query = db.query(Scan)
    if tenant_id:
        recent_scans_query = recent_scans_query.filter(Scan.tenant_id == tenant_id)
    recent_scans = recent_scans_query.order_by(Scan.timestamp.desc()).limit(10).all()

    recent_list = [
        {
            "id": r.id,
            "patient_hash": r.patient_hash,
            "scan_type": r.scan_type,
            "pathology_detected": r.pathology_detected,
            "confidence_score": r.confidence_score,
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "inference_latency": r.inference_latency,
            "pytorch_executed": r.pytorch_executed,
        }
        for r in recent_scans
    ]

    settings = get_clinic_settings(db)

    return {
        "total_scans": total_scans,
        "positive_rate": positive_rate,
        "modality_counts": {
            "XRAY": modality_map.get("XRAY", 0),
            "CT": modality_map.get("CT", 0),
            "MRI": modality_map.get("MRI", 0),
        },
        "modality_latencies": {
            "XRAY": modality_latency_map.get("XRAY", 0.0),
            "CT": modality_latency_map.get("CT", 0.0),
            "MRI": modality_latency_map.get("MRI", 0.0),
        },
        "pathology_counts": pathology_map,
        "recent_scans": recent_list,
        "settings": settings,
    }
