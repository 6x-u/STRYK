__version__ = "2.0.0"
__build__ = "fc27"
__api__ = 2

def version_string() -> str:
    return f"STRYK {__version__}+{__build__} api={__api__}"
