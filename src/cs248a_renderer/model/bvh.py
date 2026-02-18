import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable
import numpy as np
import slangpy as spy

from cs248a_renderer.model.bounding_box import BoundingBox3D
from cs248a_renderer.model.primitive import Primitive
from tqdm import tqdm


logger = logging.getLogger(__name__)


@dataclass
class BVHNode:
    # The bounding box of this node.
    bound: BoundingBox3D = field(default_factory=BoundingBox3D)
    # The index of the left child node, or -1 if this is a leaf node.
    left: int = -1
    # The index of the right child node, or -1 if this is a leaf node.
    right: int = -1
    # The starting index of the primitives in the primitives array.
    prim_left: int = 0
    # The ending index (exclusive) of the primitives in the primitives array.
    prim_right: int = 0
    # The depth of this node in the BVH tree.
    depth: int = 0

    def get_this(self) -> Dict:
        return {
            "bound": self.bound.get_this(),
            "left": self.left,
            "right": self.right,
            "primLeft": self.prim_left,
            "primRight": self.prim_right,
            "depth": self.depth,
        }

    @property
    def is_leaf(self) -> bool:
        """Checks if this node is a leaf node."""
        return self.left == -1 and self.right == -1


class BVH:
    def __init__(
        self,
        primitives: List[Primitive],
        max_nodes: int,
        min_prim_per_node: int = 1,
        num_thresholds: int = 16,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Builds the BVH from the given list of primitives. The build algorithm should
        reorder the primitives in-place to align with the BVH node structure.
        The algorithm will start from the root node and recursively partition the primitives
        into child nodes until the maximum number of nodes is reached or the primitives
        cannot be further subdivided.
        At each node, the splitting axis and threshold should be chosen using the Surface Area Heuristic (SAH)
        to minimize the expected cost of traversing the BVH during ray intersection tests.

        :param primitives: the list of primitives to build the BVH from
        :type primitives: List[Primitive]
        :param max_nodes: the maximum number of nodes in the BVH
        :type max_nodes: int
        :param min_prim_per_node: the minimum number of primitives per leaf node
        :type min_prim_per_node: int
        :param num_thresholds: the number of thresholds per axis to consider when splitting
        :type num_thresholds: int
        """
        self.nodes: List[BVHNode] = []

        # TODO: Student implementation starts here.
        root_node = BVHNode()
        # Compute bounding box by taking union of all primitives
        root_node.bound = primitives[0].bounding_box
        for prim in primitives[1:]:
            root_node.bound = BoundingBox3D.union(root_node.bound, prim.bounding_box)
        root_node.prim_left = 0
        root_node.prim_right = len(primitives)
        self.nodes.append(root_node)
        node_queue: List[Tuple[int, int, int]] = [(0, 0, len(primitives))]  # (node_index, prim_left, prim_right)

        while node_queue and len(self.nodes) < max_nodes:
            node_index, prim_left, prim_right = node_queue.pop(0)
            node = self.nodes[node_index]
            num_prims = prim_right - prim_left

            if num_prims <= min_prim_per_node:
                continue  # Leaf node

            best_cost = float('inf')
            best_axis = -1
            best_threshold = -1
            best_left_indices: List[int] = []
            best_right_indices: List[int] = []
            best_left_bound: BoundingBox3D | None = None
            best_right_bound: BoundingBox3D | None = None

            for axis in range(3):
                axis_min = node.bound.min[axis]
                axis_max = node.bound.max[axis]
                if axis_max - axis_min < 1e-5:
                    continue  # Avoid division by zero

                for t in range(1, num_thresholds):
                    threshold = axis_min + (axis_max - axis_min) * t / num_thresholds
                    left_indices: List[int] = []
                    right_indices: List[int] = []

                    for i in range(prim_left, prim_right):
                        prim = primitives[i]
                        prim_center = (prim.bounding_box.min + prim.bounding_box.max) * 0.5
                        if prim_center[axis] < threshold:
                            left_indices.append(i)
                        else:
                            right_indices.append(i)

                    if not left_indices or not right_indices:
                        continue  # Invalid split

                    left_bound = primitives[left_indices[0]].bounding_box
                    for idx in left_indices[1:]:
                        left_bound = BoundingBox3D.union(left_bound, primitives[idx].bounding_box)

                    right_bound = primitives[right_indices[0]].bounding_box
                    for idx in right_indices[1:]:
                        right_bound = BoundingBox3D.union(right_bound, primitives[idx].bounding_box)

                    cost = (left_bound.area * len(left_indices) + right_bound.area * len(right_indices))

                    if cost < best_cost:
                        best_cost = cost
                        best_axis = axis
                        best_threshold = threshold
                        best_left_indices = left_indices
                        best_right_indices = right_indices
                        best_left_bound = left_bound
                        best_right_bound = right_bound

            if best_axis == -1:
                continue  # No valid split found

            # Reorder primitives based on the best split
            new_order = best_left_indices + best_right_indices
            primitives[prim_left:prim_right] = [primitives[i] for i in new_order]

            # Create left node
            left_node = BVHNode()
            left_node.bound = best_left_bound
            left_node.prim_left = prim_left
            left_node.prim_right = prim_left + len(best_left_indices)
            left_node.depth = node.depth + 1
            self.nodes.append(left_node)
            left_index = len(self.nodes) - 1
            node.left = left_index
            node_queue.append((left_index, left_node.prim_left, left_node.prim_right))
            
            # Create right node
            right_node = BVHNode()
            right_node.bound = best_right_bound
            right_node.prim_left = left_node.prim_right
            right_node.prim_right = prim_right
            right_node.depth = node.depth + 1
            self.nodes.append(right_node)
            right_index = len(self.nodes) - 1
            node.right = right_index
            node_queue.append((right_index, right_node.prim_left, right_node.prim_right))

            if on_progress:
                on_progress(len(self.nodes), max_nodes)

        # TODO: Student implementation ends here.


def create_bvh_node_buf(module: spy.Module, bvh_nodes: List[BVHNode]) -> spy.NDBuffer:
    device = module.device
    node_buf = spy.NDBuffer(
        device=device, dtype=module.BVHNode.as_struct(), shape=(max(len(bvh_nodes), 1),)
    )
    cursor = node_buf.cursor()
    for idx, node in enumerate(bvh_nodes):
        cursor[idx].write(node.get_this())
    cursor.apply()
    return node_buf
