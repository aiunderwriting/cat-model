"""Stable, actionable domain error identities for replayable elicitation feedback."""


class EvidenceValidationError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        location: tuple[str | int, ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.location = location

    def __reduce__(self):
        # Rebuilt from its fields, so it crosses a process boundary intact.
        return _evidence_validation_error, (self.code, str(self), self.location)


def _evidence_validation_error(code, message, location) -> EvidenceValidationError:
    return EvidenceValidationError(code, message, location=location)
