"""Tests for Party name and ID tracking functionality."""
import pytest
import poolparty as pp


class TestPoolNameTracking:
    """Tests for pool name tracking in Party."""
    
    def test_pool_registered_by_id(self):
        """Pool is registered in _pools_by_id list."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGT")
            assert party._pools_by_id[0] is pool
    
    def test_pool_registered_by_name(self):
        """Pool is registered in _pools_by_name dict."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGT")
            assert party._pools_by_name[pool.name] is pool
    
    def test_get_pool_by_id(self):
        """Can retrieve pool by ID."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGT")
            assert party.get_pool_by_id(pool._id) is pool
    
    def test_get_pool_by_name(self):
        """Can retrieve pool by name."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGT").named("my_pool")
            assert party.get_pool_by_name("my_pool") is pool
    
    def test_pool_name_uniqueness(self):
        """Pool names must be unique."""
        with pp.Party() as party:
            pp.from_seq("ACGT").named("unique_name")
            with pytest.raises(ValueError, match="Pool name 'unique_name' already exists"):
                pp.from_seq("TGCA").named("unique_name")
    
    def test_pool_rename_updates_tracking(self):
        """Renaming a pool updates the party's name dict."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGT").named("old_name")
            assert party.get_pool_by_name("old_name") is pool
            pool.name = "new_name"
            assert party.get_pool_by_name("new_name") is pool
            with pytest.raises(KeyError):
                party.get_pool_by_name("old_name")
    
    def test_pool_rename_to_same_name_allowed(self):
        """Renaming a pool to its current name is allowed."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGT").named("same_name")
            pool.name = "same_name"  # Should not raise
            assert pool.name == "same_name"
    
    def test_pool_rename_to_duplicate_raises(self):
        """Renaming a pool to another pool's name raises error."""
        with pp.Party() as party:
            pool1 = pp.from_seq("ACGT").named("name1")
            pool2 = pp.from_seq("TGCA").named("name2")
            with pytest.raises(ValueError, match="Pool name 'name1' already exists"):
                pool2.name = "name1"
    
    def test_multiple_pools_tracked(self):
        """Multiple pools are tracked correctly."""
        with pp.Party() as party:
            pool1 = pp.from_seq("ACGT").named("pool_a")
            pool2 = pp.from_seq("TGCA").named("pool_b")
            pool3 = pp.from_seq("GGGG").named("pool_c")
            assert len(party._pools_by_id) == 3
            assert len(party._pools_by_name) == 3
            assert party.get_pool_by_id(0) is pool1
            assert party.get_pool_by_id(1) is pool2
            assert party.get_pool_by_id(2) is pool3


class TestOperationNameTracking:
    """Tests for operation name tracking in Party."""
    
    def test_op_registered_by_id(self):
        """Operation is registered in _ops_by_id list."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGT")
            op = pool.operation
            assert party._ops_by_id[0] is op
    
    def test_op_registered_by_name(self):
        """Operation is registered in _ops_by_name dict."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGT")
            op = pool.operation
            assert party._ops_by_name[op.name] is op
    
    def test_get_op_by_id(self):
        """Can retrieve operation by ID."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGT")
            op = pool.operation
            assert party.get_op_by_id(op.id) is op
    
    def test_get_op_by_name(self):
        """Can retrieve operation by name."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGT")
            op = pool.operation
            assert party.get_op_by_name(op.name) is op
    
    def test_op_name_uniqueness(self):
        """Operation names must be unique."""
        with pp.Party() as party:
            pool1 = pp.from_seq("ACGT")
            pool1.operation.name = "unique_op"
            pool2 = pp.from_seq("TGCA")
            with pytest.raises(ValueError, match="Operation name 'unique_op' already exists"):
                pool2.operation.name = "unique_op"
    
    def test_op_rename_updates_tracking(self):
        """Renaming an operation updates the party's name dict."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGT")
            op = pool.operation
            old_name = op.name
            op.name = "new_op_name"
            assert party.get_op_by_name("new_op_name") is op
            with pytest.raises(KeyError):
                party.get_op_by_name(old_name)
    
    def test_op_rename_to_same_name_allowed(self):
        """Renaming an operation to its current name is allowed."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGT")
            op = pool.operation
            original_name = op.name
            op.name = original_name  # Should not raise
            assert op.name == original_name
    
    def test_op_rename_updates_counter_name(self):
        """Renaming an operation updates its counter's name."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGT")
            op = pool.operation
            op.name = "renamed_op"
            assert op.counter.name == "renamed_op.state"


class TestSeparateNamespaces:
    """Tests that pool and operation names are in separate namespaces."""
    
    def test_pool_and_op_can_share_name(self):
        """A pool and operation can have the same name."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGT").named("shared_name")
            pool.operation.name = "shared_name"  # Should not raise
            assert pool.name == "shared_name"
            assert pool.operation.name == "shared_name"
    
    def test_get_pool_vs_op_by_shared_name(self):
        """Separate lookups for pools and operations with same name."""
        with pp.Party() as party:
            pool = pp.from_seq("ACGT").named("shared_name")
            pool.operation.name = "shared_name"
            assert party.get_pool_by_name("shared_name") is pool
            assert party.get_op_by_name("shared_name") is pool.operation
            assert party.get_pool_by_name("shared_name") is not party.get_op_by_name("shared_name")


class TestRetrievalErrors:
    """Tests for error handling in retrieval methods."""
    
    def test_get_pool_by_invalid_id(self):
        """Getting pool by invalid ID raises IndexError."""
        with pp.Party() as party:
            pp.from_seq("ACGT")
            with pytest.raises(IndexError):
                party.get_pool_by_id(999)
    
    def test_get_pool_by_invalid_name(self):
        """Getting pool by invalid name raises KeyError."""
        with pp.Party() as party:
            pp.from_seq("ACGT")
            with pytest.raises(KeyError):
                party.get_pool_by_name("nonexistent")
    
    def test_get_op_by_invalid_id(self):
        """Getting operation by invalid ID raises IndexError."""
        with pp.Party() as party:
            pp.from_seq("ACGT")
            with pytest.raises(IndexError):
                party.get_op_by_id(999)
    
    def test_get_op_by_invalid_name(self):
        """Getting operation by invalid name raises KeyError."""
        with pp.Party() as party:
            pp.from_seq("ACGT")
            with pytest.raises(KeyError):
                party.get_op_by_name("nonexistent")

