class GenerationExhaustedError(Exception):
    """Raised when the LLM fails to provide a valid response after the maximum number of retries."""
    pass

class DiscoveryFailedError(Exception):
    """Raised when no suitable semantic context (chunks) could be found during the discovery phase."""
    pass