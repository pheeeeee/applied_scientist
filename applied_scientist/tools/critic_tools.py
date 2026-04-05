from __future__ import annotations

import json

from applied_scientist.tools.base import Tool


class SubmitSpecReview(Tool):
    name = "submit_review"
    description = "Submit your review decision for an experiment spec."
    parameters = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["approved", "rejected"]},
            "score": {"type": "number", "description": "Ranking score (only if approved)"},
            "feedback": {"type": "string", "description": "Specific feedback or edits"},
            "control_variant_needed": {"type": "boolean"},
            "control_description": {"type": "string",
                                    "description": "Control variant description (if needed)"},
        },
        "required": ["verdict", "feedback"],
    }

    def execute(self, **kwargs) -> str:
        return json.dumps(kwargs)


class SubmitCodeReview(Tool):
    """Structured code review tool. Replaces free-text APPROVED/REJECTED parsing."""
    name = "submit_code_review"
    description = "Submit your review decision for a Builder code change."
    parameters = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["approved", "rejected"]},
            "feedback": {"type": "string", "description": "Specific feedback or required fix"},
        },
        "required": ["verdict", "feedback"],
    }

    def execute(self, **kwargs) -> str:
        return json.dumps(kwargs)


class SubmitInsightReview(Tool):
    name = "submit_insight_review"
    description = "Submit your review decision for a Builder insight."
    parameters = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["approved", "rejected"]},
            "edits": {"type": "string",
                      "description": "Edited insight text (if approved with changes)"},
            "feedback": {"type": "string", "description": "Feedback (if rejected)"},
        },
        "required": ["verdict"],
    }

    def execute(self, **kwargs) -> str:
        return json.dumps(kwargs)


class SubmitSuggestionTriage(Tool):
    name = "submit_triage"
    description = "Triage a Builder improvement suggestion."
    parameters = {
        "type": "object",
        "properties": {
            "level": {"type": "string", "enum": ["minor", "moderate", "major"]},
            "justification": {"type": "string"},
            "action": {"type": "string", "description": "What should happen next"},
        },
        "required": ["level", "justification"],
    }

    def execute(self, **kwargs) -> str:
        return json.dumps(kwargs)
