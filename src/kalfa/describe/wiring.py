def wiring_section(prepared, style, width, probe=None):
    if not prepared.implicit:
        return [style.dim("  no implicit bindings")]
    return [f"  {path}: {param} ← {key}" for path, param, key in prepared.implicit]
