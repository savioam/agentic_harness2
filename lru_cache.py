"""LRU cache with O(1) get and put.

Design
------
Two structures work together:

* A hash map (dict) from key -> node, giving O(1) lookup.
* A doubly linked list ordered by recency, giving O(1) removal of any
  node (given a reference to it) and O(1) insertion at either end.

The node right after the head sentinel is the most recently used (MRU);
the node right before the tail sentinel is the least recently used (LRU).
Sentinel nodes remove all "is this the first/last node?" edge cases.
"""

from typing import Any, Dict, Optional


class _Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: Any = None, value: Any = None) -> None:
        self.key = key
        self.value = value
        self.prev: Optional["_Node"] = None
        self.next: Optional["_Node"] = None


class LRUCache:
    """Fixed-capacity cache that evicts the least recently used entry."""

    def __init__(self, capacity: int) -> None:
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._map: Dict[Any, _Node] = {}
        # Sentinels: head <-> ... <-> tail
        self._head = _Node()
        self._tail = _Node()
        self._head.next = self._tail
        self._tail.prev = self._head

    # -- linked list helpers (all O(1)) ------------------------------------
    def _remove(self, node: _Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_front(self, node: _Node) -> None:
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    # -- public API ---------------------------------------------------------
    def get(self, key: Any) -> Any:
        """Return the value for key, or -1 if absent. Marks key as MRU."""
        node = self._map.get(key)
        if node is None:
            return -1
        self._remove(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: Any, value: Any) -> None:
        """Insert or update key. Evicts the LRU entry if over capacity."""
        node = self._map.get(key)
        if node is not None:
            node.value = value
            self._remove(node)
            self._add_to_front(node)
            return

        if len(self._map) >= self.capacity:
            lru = self._tail.prev
            self._remove(lru)
            del self._map[lru.key]

        node = _Node(key, value)
        self._map[key] = node
        self._add_to_front(node)

    def __len__(self) -> int:
        return len(self._map)

    def __contains__(self, key: Any) -> bool:
        """Membership test that does not affect recency."""
        return key in self._map


if __name__ == "__main__":
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1      # order: 1 (MRU), 2
    cache.put(3, 3)               # evicts 2
    assert cache.get(2) == -1
    cache.put(4, 4)               # evicts 1
    assert cache.get(1) == -1
    assert cache.get(3) == 3
    assert cache.get(4) == 4
    cache.put(3, 30)              # update existing, no eviction
    assert len(cache) == 2 and cache.get(3) == 30
    print("All checks passed.")
