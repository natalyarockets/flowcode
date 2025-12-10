from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Node:
    id: str
    approx_position: Dict[str, int]
    inferred_shape: str
    text_summary: str
    role: Optional[str] = None


@dataclass
class Edge:
    from_id: str
    to_id: Optional[str]
    label: Optional[str] = None
    possible_labels: Optional[List[str]] = None


@dataclass
class SemanticSkeleton:
    layout: Dict
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


