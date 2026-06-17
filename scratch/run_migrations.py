import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load env vars
load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
print(f"Connecting to database: {DATABASE_URL}")

engine = create_engine(DATABASE_URL)

with engine.begin() as conn:
    # 1. Create clinic_tenants table if it doesn't exist
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS clinic_tenants (
        id VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL UNIQUE,
        created_at TIMESTAMP WITHOUT TIME ZONE
    );
    """))
    print("clinic_tenants table verified/created.")

    # 2. Add tenant_id to scans if not exists
    try:
        conn.execute(text("ALTER TABLE scans ADD COLUMN tenant_id VARCHAR;"))
        print("tenant_id column added to scans.")
    except Exception as e:
        print(f"scans.tenant_id check: {e}")
        
    # Add foreign key constraint to scans.tenant_id if not exists
    try:
        conn.execute(text("ALTER TABLE scans ADD CONSTRAINT fk_scans_tenant FOREIGN KEY (tenant_id) REFERENCES clinic_tenants(id);"))
        print("Foreign key constraint added to scans.tenant_id.")
    except Exception as e:
        print(f"scans foreign key constraint check: {e}")

    # 3. Add tenant_id to users if not exists
    try:
        conn.execute(text("ALTER TABLE users ADD COLUMN tenant_id VARCHAR;"))
        print("tenant_id column added to users.")
    except Exception as e:
        print(f"users.tenant_id check: {e}")
        
    # Add foreign key constraint to users.tenant_id if not exists
    try:
        conn.execute(text("ALTER TABLE users ADD CONSTRAINT fk_users_tenant FOREIGN KEY (tenant_id) REFERENCES clinic_tenants(id);"))
        print("Foreign key constraint added to users.tenant_id.")
    except Exception as e:
        print(f"users foreign key constraint check: {e}")
        
    print("Database migrations applied successfully.")
