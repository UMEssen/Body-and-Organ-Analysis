def convert_name(name: str) -> str:
    return "".join(s.capitalize() for s in name.split("_"))
