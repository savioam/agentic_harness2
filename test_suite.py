from lru_cache import LRUCache


def test_lru_cache_basic_operations():
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1
    cache.put(3, 3)  # Evicts key 2
    assert cache.get(2) == -1
    cache.put(4, 4)  # Evicts key 1
    assert cache.get(1) == -1
    assert cache.get(3) == 3
    assert cache.get(4) == 4


def test_lru_cache_update_existing_key():
    cache = LRUCache(2)
    cache.put(1, 10)
    cache.put(1, 20)
    assert cache.get(1) == 20
    cache.put(2, 30)
    cache.put(3, 40)  # Evicts key 1 because key 1 was accessed before key 2
    assert cache.get(1) == -1 or cache.get(2) == -1


def test_lru_cache_recency_on_get():
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    # Accessing key 1 makes it most recently used, so key 2 should be evicted next
    assert cache.get(1) == 1
    cache.put(3, 3)
    assert cache.get(2) == -1
    assert cache.get(1) == 1
    assert cache.get(3) == 3
