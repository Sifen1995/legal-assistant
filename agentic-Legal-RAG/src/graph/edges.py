from __future__ import annotations

from typing import Callable, Optional

from .nodes import Node
from .state import LegalState


Condition = Callable[[LegalState], bool]


class GraphEdge:
    def __init__(self, source: Node, target: Node, condition: Optional[Condition] = None):
        self.source = source
        self.target = target
        self.condition = condition

    def is_active(self, state: LegalState) -> bool:
        if self.condition is None:
            return True
        return self.condition(state)


class ConditionalEdge(GraphEdge):
    def __init__(self, source: Node, target: Node, condition: Condition):
        super().__init__(source, target, condition)
