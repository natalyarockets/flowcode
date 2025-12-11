"""Pydantic schemas for validating FlowGraph review responses."""

import json
from typing import Dict, Optional, Literal, Any

from pydantic import BaseModel, root_validator


class FlowGraphNode(BaseModel):
    id: str
    shape: str
    text: str
    out: Optional[str]
    out_yes: Optional[str]
    out_no: Optional[str]


class FlowGraphReview(BaseModel):
    orientation: Literal["top-down", "left-right"]
    start_node: Optional[str]
    nodes: Dict[str, FlowGraphNode]

    @root_validator(pre=True)
    def ensure_nodes_are_dict(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        nodes = values.get("nodes")
        if not isinstance(nodes, dict):
            raise ValueError("nodes must be an object")
        return values

    def canonical_json(self) -> str:
        return self.model_dump_json(indent=2, ensure_ascii=False)


def validate_review_json(payload: Any) -> str:
    if isinstance(payload, str):
        data = json.loads(payload)
    else:
        data = payload
    review = FlowGraphReview(**data)
    return review.canonical_json()
