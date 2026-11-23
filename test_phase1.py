import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.main import app
from app.database.session import engine
from sqlalchemy import inspect

print("=" * 60)
print("1. Verifying Database Tables Creation")
print("=" * 60)

inspector = inspect(engine)
tables = inspector.get_table_names()
print(f"Created Tables in Database ({len(tables)}):")
for t in sorted(tables):
    print(f"  - {t}")

expected_tables = {
    "zones",
    "cameras",
    "persons",
    "person_photos",
    "person_embeddings",
    "roles",
    "person_zone_permissions",
    "security_events"
}

missing = expected_tables - set(tables)
if missing:
    print(f"FAILED: Missing tables: {missing}")
    sys.exit(1)
else:
    print("\nSUCCESS: All expected database tables created successfully!")

print("\n" + "=" * 60)
print("2. Verifying FastAPI Endpoints in Router")
print("=" * 60)
routes = [route.path for route in app.routes]
expected_routes = [
    "/api/v1/zones",
    "/api/v1/cameras",
    "/api/v1/persons/enroll",
    "/api/v1/persons",
    "/health"
]

for r in expected_routes:
    found = any(r in route for route in routes)
    print(f"Route {r}: {'FOUND' if found else 'MISSING'}")
    assert found, f"Route {r} not found in app routes"

print("\nSUCCESS: All Phase 1 routes registered!")
