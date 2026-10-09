"""Only the shared FastGRPO engine and OPD Reflex are supported."""


METHOD_TO_REFLEX_MODE = {
    "fastgrpo": "off",
    "opd_reflex": "opd_reflex",
}


def resolve_method(method):
    normalized = str(method).strip().lower()
    if normalized not in METHOD_TO_REFLEX_MODE:
        choices = ", ".join(sorted(METHOD_TO_REFLEX_MODE))
        raise ValueError(f"METHOD must be one of: {choices}")
    return normalized, METHOD_TO_REFLEX_MODE[normalized]
