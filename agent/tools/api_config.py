REQUIRED_API_FIELDS = ("provider", "base_url", "model_name")


def validate_api_config(api_config: dict) -> dict:
    """Validate the single API configuration shared by every agent role."""
    if not isinstance(api_config, dict):
        raise TypeError("API configuration must contain one JSON object.")

    missing = [
        field for field in REQUIRED_API_FIELDS
        if not isinstance(api_config.get(field), str) or not api_config[field].strip()
    ]
    if missing:
        raise ValueError(
            f"API configuration is missing non-empty fields: {missing}. "
            f"Required fields: {REQUIRED_API_FIELDS}."
        )
    if api_config["provider"].strip().lower() != "openai":
        raise ValueError("The release supports one shared OpenAI-compatible API only.")
    return api_config
