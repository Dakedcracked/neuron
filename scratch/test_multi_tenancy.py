import sys
import os
import unittest
from pathlib import Path

# Add project root to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Force cwd
os.chdir(str(project_root))

from fastapi.testclient import TestClient
from app.main import app
from app.database import engine, Base, SessionLocal, ClinicTenant, Scan
from app.auth import User, hash_password

class TestMultiTenancy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure tables exist
        Base.metadata.create_all(bind=engine)
        cls.db = SessionLocal()
        
        # Clean up existing test data to ensure clean state
        cls.db.query(User).filter(User.username.in_(["user_tenant_a", "user_tenant_b"])).delete(synchronize_session=False)
        cls.db.query(ClinicTenant).filter(ClinicTenant.id.in_(["tenant-a-uuid", "tenant-b-uuid"])).delete(synchronize_session=False)
        cls.db.commit()

        # 1. Create Tenants
        cls.tenant_a = ClinicTenant(id="tenant-a-uuid", name="Hospital Tenant A")
        cls.tenant_b = ClinicTenant(id="tenant-b-uuid", name="Hospital Tenant B")
        cls.db.add_all([cls.tenant_a, cls.tenant_b])
        cls.db.commit()

        # 2. Create Users associated with tenants
        cls.user_a = User(
            id="user-a-uuid",
            username="user_tenant_a",
            hashed_password=hash_password("password123"),
            role="radiologist",
            is_active=True,
            tenant_id="tenant-a-uuid"
        )
        cls.user_b = User(
            id="user-b-uuid",
            username="user_tenant_b",
            hashed_password=hash_password("password123"),
            role="radiologist",
            is_active=True,
            tenant_id="tenant-b-uuid"
        )
        cls.db.add_all([cls.user_a, cls.user_b])
        cls.db.commit()

        # 3. Create Scans associated with tenants
        cls.scan_a = Scan(
            id="scan-a-uuid",
            patient_hash="hash_a",
            scan_type="MRI",
            pathology_detected="Normal",
            confidence_score=0.95,
            status="completed",
            tenant_id="tenant-a-uuid"
        )
        cls.scan_b = Scan(
            id="scan-b-uuid",
            patient_hash="hash_b",
            scan_type="CT",
            pathology_detected="Stroke",
            confidence_score=0.88,
            status="completed",
            tenant_id="tenant-b-uuid"
        )
        cls.db.add_all([cls.scan_a, cls.scan_b])
        cls.db.commit()

        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        # Clean up database records
        cls.db.query(Scan).filter(Scan.id.in_(["scan-a-uuid", "scan-b-uuid"])).delete(synchronize_session=False)
        cls.db.query(User).filter(User.username.in_(["user_tenant_a", "user_tenant_b"])).delete(synchronize_session=False)
        cls.db.query(ClinicTenant).filter(ClinicTenant.id.in_(["tenant-a-uuid", "tenant-b-uuid"])).delete(synchronize_session=False)
        cls.db.commit()
        cls.db.close()

    def _get_token(self, username, password):
        response = self.client.post("/api/auth/login", data={"username": username, "password": password})
        self.assertEqual(response.status_code, 200, f"Login failed: {response.text}")
        return response.json()["access_token"]

    def test_tenant_a_isolation(self):
        token = self._get_token("user_tenant_a", "password123")
        headers = {"Authorization": f"Bearer {token}"}

        # 1. Retrieve Scan A (should succeed)
        response = self.client.get("/api/scans/scan-a-uuid", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["patient_hash"], "hash_a")

        # 2. Retrieve Scan B (should fail with 404 Not Found)
        response = self.client.get("/api/scans/scan-b-uuid", headers=headers)
        self.assertEqual(response.status_code, 404)

        # 3. Check dashboard metrics (should only count scan A)
        response = self.client.get("/api/dashboard-metrics", headers=headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_scans"], 1)
        self.assertEqual(data["modality_counts"]["MRI"], 1)
        self.assertEqual(data["modality_counts"]["CT"], 0)

        # 4. Check scan history list (should only contain scan A)
        response = self.client.get("/api/scans", headers=headers)
        self.assertEqual(response.status_code, 200)
        scans = response.json()["scans"]
        scan_ids = [s["id"] for s in scans]
        self.assertIn("scan-a-uuid", scan_ids)
        self.assertNotIn("scan-b-uuid", scan_ids)

    def test_tenant_b_isolation(self):
        token = self._get_token("user_tenant_b", "password123")
        headers = {"Authorization": f"Bearer {token}"}

        # 1. Retrieve Scan B (should succeed)
        response = self.client.get("/api/scans/scan-b-uuid", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["patient_hash"], "hash_b")

        # 2. Retrieve Scan A (should fail with 404 Not Found)
        response = self.client.get("/api/scans/scan-a-uuid", headers=headers)
        self.assertEqual(response.status_code, 404)

        # 3. Check dashboard metrics (should only count scan B)
        response = self.client.get("/api/dashboard-metrics", headers=headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_scans"], 1)
        self.assertEqual(data["modality_counts"]["MRI"], 0)
        self.assertEqual(data["modality_counts"]["CT"], 1)

        # 4. Check scan history list (should only contain scan B)
        response = self.client.get("/api/scans", headers=headers)
        self.assertEqual(response.status_code, 200)
        scans = response.json()["scans"]
        scan_ids = [s["id"] for s in scans]
        self.assertIn("scan-b-uuid", scan_ids)
        self.assertNotIn("scan-a-uuid", scan_ids)

if __name__ == "__main__":
    unittest.main()
