import json

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.employee_registration_service import (
    EmployeeRegistrationService,
    EMPLOYEE_REGISTRY_DIR,
)

router = APIRouter()


@router.post("/register", status_code=201)
async def register_employee(
    employee_id: str = Form(...),
    employee_name: str = Form(...),
    files: list[UploadFile] = File(..., description="Upload 1 to 5 JPG, PNG or WEBP reference photos"),
):
    """Register an employee and create normalized 512-D OSNet reference embeddings."""
    try:
        result = EmployeeRegistrationService.register_employee(
            employee_id=employee_id,
            employee_name=employee_name,
            files=files,
        )
        return {
            "success": True,
            "message": "Employee registered successfully",
            "data": result,
        }
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail=str(error))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))
    except Exception as error:
        print("Employee registration error:", error)
        raise HTTPException(
            status_code=500,
            detail="Employee registration failed. Check the API server logs.",
        )
    finally:
        for upload in files:
            await upload.close()


@router.get("")
def list_employees():
    """List registered employees without exposing image or embedding paths."""
    if not EMPLOYEE_REGISTRY_DIR.exists():
        return {"success": True, "total_employees": 0, "employees": []}

    employees = []
    for metadata_path in sorted(EMPLOYEE_REGISTRY_DIR.glob("*/employee.json")):
        try:
            with metadata_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)
            employees.append({
                "employee_id": metadata.get("employee_id"),
                "employee_name": metadata.get("employee_name"),
                "total_reference_photos": metadata.get("total_reference_photos", 0),
                "embedding_dimension": metadata.get("embedding_dimension"),
            })
        except (OSError, json.JSONDecodeError) as error:
            print(f"Skipping invalid employee metadata {metadata_path}: {error}")

    return {
        "success": True,
        "total_employees": len(employees),
        "employees": employees,
    }


@router.get("/{employee_id}")
def get_employee(employee_id: str):
    """Get registration metadata for one employee."""
    if not employee_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in employee_id):
        raise HTTPException(status_code=400, detail="Invalid employee ID")

    metadata_path = EMPLOYEE_REGISTRY_DIR / employee_id / "employee.json"
    if not metadata_path.is_file():
        raise HTTPException(status_code=404, detail="Employee not found")

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=500, detail="Employee metadata is unreadable")

    return {"success": True, "data": metadata}
