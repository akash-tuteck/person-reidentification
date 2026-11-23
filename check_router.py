import sys
sys.path.insert(0, r"C:\person-reidentification")
from app.routers.endpoints.access_router import router
print("Router imported OK")
print("Enroll endpoint parameters:")
for route in router.routes:
    if hasattr(route, 'path') and 'enroll' in route.path:
        print(f"  Path: {route.path}")
        for dep in route.dependencies:
            print(f"  Dep: {dep}")
