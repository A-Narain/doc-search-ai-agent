ALLOWED_SCHEMA = {
    "employees": {
        "description": "Company employees and their roles",
        "columns": {
            "employee_id":    "Unique employee identifier",
            "first_name":     "Employee's first name",
            "last_name":      "Employee's last name",
            "department_id":  "Links to departments table",
            "job_title":      "Employee's job title",
            "hire_date":      "Date employee was hired",
        }
    },
    "departments": {
        "description": "Company departments",
        "columns": {
            "department_id":   "Unique department identifier",
            "department_name": "Name of the department",
        }
    },
    "projects": {
        "description": "Projects run by each department",
        "columns": {
            "project_id":     "Unique project identifier",
            "project_name":   "Name of the project",
            "department_id":  "Links to departments table",
            "status":         "Project status: active or completed",
        }
    }
}

def get_schema_description():
    """
    Converts the allowlist into a text block the LLM can read
    to understand what tables/columns it's allowed to query.
    """
    lines = []
    for table, info in ALLOWED_SCHEMA.items():
        lines.append(f"Table: {table} — {info['description']}")
        for col, desc in info["columns"].items():
            lines.append(f"  - {col}: {desc}")
        lines.append("")
    return "\n".join(lines)


def get_allowed_tables():
    return set(ALLOWED_SCHEMA.keys())


def get_allowed_columns(table_name):
    if table_name not in ALLOWED_SCHEMA:
        return set()
    return set(ALLOWED_SCHEMA[table_name]["columns"].keys())