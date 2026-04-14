"""
Room graph builder - creates graph representation of floor plan requirements
"""

from dataclasses import dataclass, field
from typing import Optional
import networkx as nx

from app.models.schemas import FloorPlanRequest
from app.models.room_types import RoomType, ROOM_TYPES, get_room_config
from app.core.vastu_rules import get_preferred_zones, get_avoided_zones, ADJACENCY_RULES


@dataclass
class RoomNode:
    """Represents a room in the graph"""
    id: str
    room_type: str
    min_area: float
    max_area: float
    min_width: float
    min_height: float
    preferred_zones: list[str] = field(default_factory=list)
    avoid_zones: list[str] = field(default_factory=list)
    priority: int = 0
    is_required: bool = True


class RoomGraphBuilder:
    """Builds a graph representation of room requirements"""

    def __init__(self):
        self.graph: Optional[nx.Graph] = None
        self.nodes: list[RoomNode] = []

    def build_from_request(self, request: FloorPlanRequest) -> nx.Graph:
        """
        Build room graph from floor plan request

        Args:
            request: User's floor plan requirements

        Returns:
            NetworkX graph with rooms as nodes and adjacency requirements as edges
        """
        self.graph = nx.Graph()
        self.nodes = []

        # Add required rooms based on request
        self._add_bedrooms(request)
        self._add_bathrooms(request)

        if request.include_living_room:
            self._add_room(RoomType.LIVING_ROOM, request.facing_direction)

        if request.include_kitchen:
            self._add_room(RoomType.KITCHEN, request.facing_direction)

        if request.include_dining:
            self._add_room(RoomType.DINING, request.facing_direction)

        if request.include_puja_room:
            self._add_room(RoomType.PUJA_ROOM, request.facing_direction)

        if request.include_parking:
            self._add_room(RoomType.PARKING, request.facing_direction)

        if request.include_balcony:
            self._add_room(RoomType.BALCONY, request.facing_direction)

        if request.include_staircase:
            self._add_room(RoomType.STAIRCASE, request.facing_direction)

        if request.include_ots:
            self._add_room(RoomType.OTS, request.facing_direction)

        # Add adjacency edges
        self._add_adjacency_edges()

        return self.graph

    def _add_room(self, room_type: str, facing: str, room_id: Optional[str] = None) -> RoomNode:
        """Add a room node to the graph"""
        config = get_room_config(room_type)

        if room_id is None:
            # Generate unique ID
            count = sum(1 for n in self.nodes if n.room_type == room_type)
            room_id = f"{room_type}_{count + 1}" if count > 0 else room_type

        node = RoomNode(
            id=room_id,
            room_type=room_type,
            min_area=config.min_area,
            max_area=config.max_area,
            min_width=config.min_width,
            min_height=config.min_height,
            preferred_zones=get_preferred_zones(room_type, facing),
            avoid_zones=get_avoided_zones(room_type),
            priority=config.priority,
        )

        self.nodes.append(node)
        self.graph.add_node(
            node.id,
            room_type=node.room_type,
            min_area=node.min_area,
            max_area=node.max_area,
            min_width=node.min_width,
            min_height=node.min_height,
            preferred_zones=node.preferred_zones,
            avoid_zones=node.avoid_zones,
            priority=node.priority,
        )

        return node

    def _add_bedrooms(self, request: FloorPlanRequest):
        """Add bedroom nodes"""
        # First bedroom is master bedroom
        self._add_room(RoomType.MASTER_BEDROOM, request.facing_direction)

        # Additional bedrooms
        for i in range(1, request.num_bedrooms):
            self._add_room(
                RoomType.BEDROOM,
                request.facing_direction,
                f"bedroom_{i + 1}"
            )

    def _add_bathrooms(self, request: FloorPlanRequest):
        """Add bathroom nodes"""
        for i in range(request.num_bathrooms):
            room_id = "bathroom" if i == 0 else f"bathroom_{i + 1}"
            self._add_room(RoomType.BATHROOM, request.facing_direction, room_id)

    def _add_adjacency_edges(self):
        """Add edges for adjacency requirements"""
        # Preferred adjacencies (weight = positive)
        for room1, room2 in ADJACENCY_RULES["preferred"]:
            nodes1 = [n for n in self.nodes if n.room_type == room1]
            nodes2 = [n for n in self.nodes if n.room_type == room2]

            for n1 in nodes1:
                for n2 in nodes2:
                    self.graph.add_edge(n1.id, n2.id, weight=1, preference="preferred")

        # Attach first bathroom to master bedroom
        master = next((n for n in self.nodes if n.room_type == RoomType.MASTER_BEDROOM), None)
        bathroom = next((n for n in self.nodes if n.room_type == RoomType.BATHROOM), None)
        if master and bathroom:
            self.graph.add_edge(master.id, bathroom.id, weight=2, preference="attached")

        # Living room should connect to entrance
        living = next((n for n in self.nodes if n.room_type == RoomType.LIVING_ROOM), None)
        if living:
            self.graph.nodes[living.id]["connects_entrance"] = True

    def get_nodes_by_priority(self) -> list[RoomNode]:
        """Get nodes sorted by placement priority (highest first)"""
        return sorted(self.nodes, key=lambda n: n.priority, reverse=True)

    def get_total_min_area(self) -> float:
        """Get total minimum area required for all rooms"""
        return sum(n.min_area for n in self.nodes)

    def validate_against_plot(self, plot_area: float) -> tuple[bool, str]:
        """
        Validate if rooms can fit in plot

        Returns:
            (is_valid, message)
        """
        total_min = self.get_total_min_area()

        # Account for walls and passages (~15% overhead)
        required_area = total_min * 1.15

        if required_area > plot_area:
            return False, f"Plot area ({plot_area} sq ft) is too small for requested rooms (need ~{required_area:.0f} sq ft)"

        if required_area > plot_area * 0.85:
            return True, "Warning: Plot will be tightly packed. Consider reducing rooms."

        return True, "Plot size is adequate for requested rooms."
