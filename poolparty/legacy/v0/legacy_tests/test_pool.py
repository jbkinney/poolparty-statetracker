"""Tests for the base Pool class."""

import pytest
from poolparty import Pool, visualize_computation_graph
from io import StringIO
import sys


def test_pool_basic_creation():
    """Test basic Pool creation with a sequence."""
    pool = Pool(seqs=["ACGTACGTACGTACGT"])
    assert pool.seq == "ACGTACGTACGTACGT"
    assert pool.num_states == 1
    assert pool.is_sequential_compatible()


def test_pool_concatenation_with_string():
    """Test Pool concatenation with string."""
    pool = Pool(seqs=["ACGTACGTACGTACGT"])
    combined = pool + "TGCATGCATGCA"
    assert combined.seq == "ACGTACGTACGTACGTTGCATGCATGCA"


def test_pool_concatenation_with_pool():
    """Test Pool concatenation with another Pool."""
    pool1 = Pool(seqs=["ACGTACGTACGTACGT"])
    pool2 = Pool(seqs=["TGCATGCATGCATGCA"])
    combined = pool1 + pool2
    assert combined.seq == "ACGTACGTACGTACGTTGCATGCATGCATGCA"


def test_pool_repetition():
    """Test Pool repetition operator."""
    pool = Pool(seqs=["ACGTACGT"])
    repeated = pool * 3
    assert repeated.seq == "ACGTACGTACGTACGTACGTACGT"


def test_pool_slicing():
    """Test Pool slicing."""
    pool = Pool(seqs=["ACGTACGTACGTACGT"])
    sliced = pool[4:12]
    assert sliced.seq == "ACGTACGT"


def test_pool_length():
    """Test Pool sequence length properties."""
    pool = Pool(seqs=["ACGTACGTACGTACGT"])
    assert pool.seq_length == 16


def test_pool_string_representation():
    """Test Pool __str__ and __repr__."""
    pool = Pool(seqs=["ACGTACGTACGTACGT"])
    assert str(pool) == "ACGTACGTACGTACGT"
    assert "Pool" in repr(pool)


def test_pool_state_management():
    """Test Pool state setting."""
    pool = Pool(seqs=["ACGTACGTACGTACGT"], mode='sequential')
    pool.set_sequential_op_states(0)
    assert pool.get_state() == 0


def test_pool_radd():
    """Test Pool reverse addition."""
    pool = Pool(seqs=["ACGTACGTACGTACGT"])
    combined = "TTTTTTTTTTTT" + pool
    assert combined.seq == "TTTTTTTTTTTTACGTACGTACGTACGT"


def test_pool_rmul():
    """Test Pool reverse multiplication."""
    pool = Pool(seqs=["ACGTACGT"])
    repeated = 2 * pool
    assert repeated.seq == "ACGTACGTACGTACGT"


def test_generate_seqs_with_computation_graph_simple():
    """Test generate_seqs with return_computation_graph for a simple Pool."""
    pool = Pool(seqs=["ACGTACGTACGTACGT"])
    result = pool.generate_library(num_seqs=3, return_computation_graph=True)
    
    # Check structure
    assert "sequences" in result
    assert "graph" in result
    assert "node_sequences" in result
    
    # Check sequences
    assert result["sequences"] == ["ACGTACGTACGTACGT"] * 3
    
    # Check graph structure
    assert "nodes" in result["graph"]
    assert len(result["graph"]["nodes"]) == 1
    assert result["graph"]["nodes"][0]["type"] == "Pool"
    assert result["graph"]["nodes"][0]["node_id"] == 0
    assert result["graph"]["nodes"][0]["op"] is None
    
    # Check node_sequences
    assert "0" in result["node_sequences"]
    assert result["node_sequences"]["0"] == ["ACGTACGTACGTACGT"] * 3


def test_generate_seqs_with_computation_graph_concatenation():
    """Test generate_seqs with return_computation_graph for concatenated pools."""
    pool1 = Pool(seqs=["AAAAAAAAAAAAAAAA"])
    pool2 = Pool(seqs=["TTTTTTTTTTTTTTTT"])
    combined = pool1 + pool2
    
    result = combined.generate_seqs(num_seqs=2, return_computation_graph=True)
    
    # Check sequences
    assert result["sequences"] == ["AAAAAAAAAAAAAAAATTTTTTTTTTTTTTTT"] * 2
    
    # Check graph structure
    assert len(result["graph"]["nodes"]) == 3
    
    root_node = result["graph"]["nodes"][0]
    assert root_node["type"] == "Pool"
    assert root_node["op"] == "+"
    assert len(root_node["parent_ids"]) == 2


def test_generate_seqs_with_computation_graph_with_literals():
    """Test generate_seqs with return_computation_graph including string literals."""
    pool = Pool(seqs=["AAAAAAAAAAAAAAAA"])
    combined = "PREFIX." + pool + ".SUFFIX"
    
    result = combined.generate_seqs(num_seqs=2, return_computation_graph=True)
    
    # Check sequences
    assert result["sequences"] == ["PREFIX.AAAAAAAAAAAAAAAA.SUFFIX"] * 2
    
    # Check that literal nodes exist
    literal_nodes = [n for n in result["graph"]["nodes"] if n["type"] == "literal"]
    assert len(literal_nodes) == 2


def test_generate_seqs_with_computation_graph_random_mutations():
    """Test generate_seqs with return_computation_graph for RandomMutationPool."""
    from poolparty import RandomMutationPool
    
    base_pool = Pool(seqs=["AAAAAAAAAAAAAAAA"])
    mut_pool = RandomMutationPool(base_pool, alphabet=['A', 'C', 'G', 'T'], mutation_rate=0.5)
    
    result = mut_pool.generate_seqs(num_seqs=5, return_computation_graph=True)
    
    # Check structure
    assert len(result["sequences"]) == 5
    assert "graph" in result
    
    root_node = result["graph"]["nodes"][0]
    assert root_node["op"] == "mutate"
    assert root_node["num_states"] == "infinite"


def test_generate_seqs_with_computation_graph_sequential_mode():
    """Test generate_seqs with return_computation_graph in sequential mode."""
    pool = Pool(seqs=["AAAAAAAAAAAAAAAA", "TTTTTTTTTTTTTTTT", "CCCCCCCCCCCCCCCC"])
    pool.set_mode('sequential')
    
    result = pool.generate_library(num_seqs=3, return_computation_graph=True)
    
    # Check sequences iterate through the list
    assert result["sequences"] == ["AAAAAAAAAAAAAAAA", "TTTTTTTTTTTTTTTT", "CCCCCCCCCCCCCCCC"]
    assert result["node_sequences"]["0"] == ["AAAAAAAAAAAAAAAA", "TTTTTTTTTTTTTTTT", "CCCCCCCCCCCCCCCC"]


def test_generate_seqs_with_computation_graph_complex():
    """Test generate_seqs with return_computation_graph for complex nested structure."""
    pool1 = Pool(seqs=["AAAAA"])
    pool2 = Pool(seqs=["BBBBB"])
    pool3 = Pool(seqs=["CCCCC"])
    
    intermediate = pool1 + pool2
    combined = intermediate + pool3
    
    result = combined.generate_seqs(num_seqs=2, return_computation_graph=True)
    
    # Check sequences
    assert result["sequences"] == ["AAAAABBBBBCCCCC"] * 2
    
    # Check graph has all nodes
    assert len(result["graph"]["nodes"]) == 5


def test_visualize_computation_graph_basic():
    """Test visualize_computation_graph function with basic pool."""
    pool = Pool(seqs=["ACGTACGTACGTACGT"])
    result = pool.generate_library(num_seqs=1, return_computation_graph=True)
    
    captured_output = StringIO()
    sys.stdout = captured_output
    visualize_computation_graph(result["graph"], result["node_sequences"])
    sys.stdout = sys.__stdout__
    
    output = captured_output.getvalue()
    assert "node_id: 0" in output


def test_visualize_computation_graph_with_concatenation():
    """Test visualize_computation_graph function with concatenated pools."""
    pool1 = Pool(seqs=["AAAAAAAAAAAAAAAA"])
    pool2 = Pool(seqs=["TTTTTTTTTTTTTTTT"])
    combined = pool1 + pool2
    
    result = combined.generate_seqs(num_seqs=1, return_computation_graph=True)
    
    captured_output = StringIO()
    sys.stdout = captured_output
    visualize_computation_graph(result["graph"], result["node_sequences"])
    sys.stdout = sys.__stdout__
    
    output = captured_output.getvalue()
    assert "node_id: 0" in output
    assert "op: +" in output


def test_visualize_computation_graph_without_sequences():
    """Test visualize_computation_graph function without node_sequences."""
    pool = Pool(seqs=["ACGTACGTACGTACGT"])
    result = pool.generate_library(num_seqs=1, return_computation_graph=True)
    
    captured_output = StringIO()
    sys.stdout = captured_output
    visualize_computation_graph(result["graph"])
    sys.stdout = sys.__stdout__
    
    output = captured_output.getvalue()
    assert "node_id: 0" in output
    assert "op: input" in output


def test_generate_seqs_without_computation_graph():
    """Test that generate_seqs still works without return_computation_graph."""
    pool = Pool(seqs=["ACGTACGTACGTACGT"])
    result = pool.generate_library(num_seqs=3, return_computation_graph=False)
    
    assert isinstance(result, list)
    assert result == ["ACGTACGTACGTACGT"] * 3


def test_generate_seqs_computation_graph_with_num_complete_iterations():
    """Test generate_seqs with return_computation_graph using num_complete_iterations."""
    pool = Pool(seqs=["AAAAAAAAAAAAAAAA", "BBBBBBBBBBBBBBBB"])
    pool.set_mode('sequential')
    
    result = pool.generate_library(num_complete_iterations=2, return_computation_graph=True)
    
    assert result["sequences"] == ["AAAAAAAAAAAAAAAA", "BBBBBBBBBBBBBBBB"] * 2


def test_visualize_computation_graph_show_first_only():
    """Test visualize_computation_graph with show_first_only parameter."""
    pool = Pool(seqs=["AAAAAAAAAAAAAAAA", "TTTTTTTTTTTTTTTT", "CCCCCCCCCCCCCCCC"])
    pool.set_mode('sequential')
    combined = "PREFIX." + pool
    
    result = combined.generate_seqs(num_seqs=3, return_computation_graph=True)
    
    captured_output = StringIO()
    sys.stdout = captured_output
    visualize_computation_graph(result["graph"], result["node_sequences"], show_first_only=True)
    sys.stdout = sys.__stdout__
    
    output = captured_output.getvalue()
    assert "node_id: 0" in output


# ============================================================================
# Tests for Pool with seqs parameter (list-based sequence selection)
# ============================================================================

def test_pool_creation_with_multiple_seqs():
    """Test Pool creation with multiple sequences."""
    seqs = ['AAAAAAAAAAAAAAAA', 'TTTTTTTTTTTTTTTT', 'GGGGGGGGGGGGGGGG', 'CCCCCCCCCCCCCCCC']
    pool = Pool(seqs=seqs)
    assert pool.num_states == 4
    assert pool.is_sequential_compatible()


def test_pool_sequential_iteration_through_seqs():
    """Test Pool iteration through all sequences in list."""
    seqs = ['AAAAAAAAAAAAAAAA', 'TTTTTTTTTTTTTTTT', 'GGGGGGGGGGGGGGGG']
    pool = Pool(seqs=seqs)
    
    sequences = []
    for state in range(pool.num_states):
        pool.set_sequential_op_states(state)
        sequences.append(pool.seq)
    
    assert sequences == seqs


def test_pool_state_setting_with_multiple_seqs():
    """Test Pool state setting with multiple sequences."""
    seqs = ['AAAAAAAAAAAAAAAA', 'TTTTTTTTTTTTTTTT', 'GGGGGGGGGGGGGGGG', 'CCCCCCCCCCCCCCCC']
    pool = Pool(seqs=seqs, mode='sequential')
    
    pool.set_sequential_op_states(0)
    assert pool.seq == 'AAAAAAAAAAAAAAAA'
    
    pool.set_sequential_op_states(1)
    assert pool.seq == 'TTTTTTTTTTTTTTTT'
    
    pool.set_sequential_op_states(2)
    assert pool.seq == 'GGGGGGGGGGGGGGGG'
    
    pool.set_sequential_op_states(3)
    assert pool.seq == 'CCCCCCCCCCCCCCCC'


def test_pool_wrapping_with_seqs():
    """Test Pool wraps around when state exceeds length."""
    seqs = ['AAAAAAAAAAAAAAAA', 'BBBBBBBBBBBBBBBB', 'CCCCCCCCCCCCCCCC']
    pool = Pool(seqs=seqs, mode='sequential')
    
    pool.set_sequential_op_states(3)  # Should wrap to 0
    assert pool.seq == 'AAAAAAAAAAAAAAAA'
    
    pool.set_sequential_op_states(4)  # Should wrap to 1
    assert pool.seq == 'BBBBBBBBBBBBBBBB'


def test_pool_with_pool_objects_as_seqs():
    """Test Pool with Pool objects as elements in seqs list."""
    pool1 = Pool(seqs=["AAAAAAAAAAAAAAAA"])
    pool2 = Pool(seqs=["TTTTTTTTTTTTTTTT"])
    seq_pool = Pool(seqs=[pool1, pool2, "GGGGGGGGGGGGGGGG"])
    
    seq_pool.set_sequential_op_states(0)
    assert seq_pool.seq == "AAAAAAAAAAAAAAAA"
    
    seq_pool.set_sequential_op_states(1)
    assert seq_pool.seq == "TTTTTTTTTTTTTTTT"
    
    seq_pool.set_sequential_op_states(2)
    assert seq_pool.seq == "GGGGGGGGGGGGGGGG"


def test_pool_repr_with_multiple_seqs():
    """Test Pool __repr__ with multiple sequences."""
    seqs = ['AAAAAAAAAAAAAAAA', 'BBBBBBBBBBBBBBBB', 'CCCCCCCCCCCCCCCC']
    pool = Pool(seqs=seqs)
    assert "Pool(3 seqs)" in repr(pool)


def test_pool_repr_long_list():
    """Test Pool __repr__ with many sequences."""
    seqs = [f"SEQ{i}" * 3 for i in range(10)]
    pool = Pool(seqs=seqs)
    assert "Pool(10 seqs)" in repr(pool)


def test_pool_selects_from_list():
    """Test that Pool can select all items from the provided list."""
    seqs = ['AAAAAAAAAAAAAAAA', 'TTTTTTTTTTTTTTTT', 'GGGGGGGGGGGGGGGG']
    pool = Pool(seqs=seqs)
    
    for i in range(20):
        pool.set_sequential_op_states(i)
        assert pool.seq in seqs


def test_pool_deterministic_with_seqs():
    """Test that Pool is deterministic with same state."""
    seqs = ['AAAAAAAAAAAAAAAA', 'TTTTTTTTTTTTTTTT', 'GGGGGGGGGGGGGGGG', 'CCCCCCCCCCCCCCCC']
    pool = Pool(seqs=seqs)
    
    pool.set_sequential_op_states(42)
    seq1 = pool.seq
    pool.set_sequential_op_states(42)
    seq2 = pool.seq
    assert seq1 == seq2


def test_pool_single_sequence_in_list():
    """Test Pool with single sequence in list always returns that sequence."""
    seqs = ['ONLYONESEQUENCES']
    pool = Pool(seqs=seqs)
    
    for i in range(10):
        pool.set_sequential_op_states(i)
        assert pool.seq == 'ONLYONESEQUENCES'


def test_pool_concatenation_finite_with_seqs():
    """Test Pool concatenation with seqs creates finite pool."""
    pool1 = Pool(seqs=['AAAAAAAAAAAAAAAA', 'BBBBBBBBBBBBBBBB'])
    pool2 = Pool(seqs=['XXXXXXXXXXXXXXXX', 'YYYYYYYYYYYYYYYY'])
    combined = pool1 + pool2
    
    assert combined.num_states == 4
    assert combined.is_sequential_compatible()
    
    sequences = []
    for state in range(combined.num_states):
        combined.set_state(state)
        sequences.append(combined.seq)
    
    assert len(sequences) == 4
    assert all(len(seq) == 32 for seq in sequences)


def test_pool_different_lengths_error():
    """Test that Pool raises error when sequences have different lengths."""
    seqs = ['AAAAAAAAAAAAAAAA', 'TTTTTTTTTTTTTTTTTT', 'GGGGGGGGGGGGGGGG']  # Different lengths
    with pytest.raises(ValueError, match="All sequences in seqs must have the same length"):
        Pool(seqs=seqs)


def test_pool_different_lengths_with_pools_error():
    """Test that Pool raises error when Pool objects have different lengths."""
    pool1 = Pool(seqs=["AAAAAAAAAAAAAAAA"])
    pool2 = Pool(seqs=["TTTTTTTTTTTTTTTTTT"])
    with pytest.raises(ValueError, match="All sequences in seqs must have the same length"):
        Pool(seqs=[pool1, pool2])


# ============================================================================
# COMPREHENSIVE COMPUTATION GRAPH TESTS
# ============================================================================
# These tests precisely delineate the behavior of the computation graph
# structure, including node properties, parent-child relationships,
# and backward compatibility for complex composite pools.
# ============================================================================

class TestComputationGraphNodeStructure:
    """Tests for the exact structure of computation graph nodes."""
    
    def test_base_pool_node_structure(self):
        """Test that a base Pool produces correct node structure."""
        pool = Pool(seqs=["ACGT"])
        result = pool.generate_library(num_seqs=1, return_computation_graph=True)
        
        nodes = result["graph"]["nodes"]
        assert len(nodes) == 1
        
        node = nodes[0]
        # Verify all expected fields exist
        assert "node_id" in node
        assert "type" in node
        assert "op" in node
        assert "num_states" in node
        assert "mode" in node
        assert "parent_ids" in node
        assert "name" in node
        
        # Verify exact values for base Pool
        assert node["node_id"] == 0
        assert node["type"] == "Pool"
        assert node["op"] is None  # Base pool has no operation
        assert node["num_states"] == 1
        assert node["mode"] == "random"  # Default mode
        assert node["parent_ids"] == []  # Leaf node
        assert node["name"] is None  # No name by default
    
    def test_named_pool_node_structure(self):
        """Test that named Pool includes name in node structure."""
        pool = Pool(seqs=["ACGT"], name="promoter")
        result = pool.generate_library(num_seqs=1, return_computation_graph=True)
        
        node = result["graph"]["nodes"][0]
        assert node["name"] == "promoter"
    
    def test_sequential_pool_node_structure(self):
        """Test that sequential mode Pool has correct mode field."""
        pool = Pool(seqs=["AAAA", "TTTT", "GGGG"], mode='sequential')
        result = pool.generate_library(num_seqs=3, return_computation_graph=True)
        
        node = result["graph"]["nodes"][0]
        assert node["mode"] == "sequential"
        assert node["num_states"] == 3
    
    def test_multi_seq_pool_num_states(self):
        """Test that num_states reflects number of sequences."""
        seqs = ["AAA", "TTT", "GGG", "CCC", "ATA"]
        pool = Pool(seqs=seqs)
        result = pool.generate_library(num_seqs=1, return_computation_graph=True)
        
        node = result["graph"]["nodes"][0]
        assert node["num_states"] == 5
    
    def test_literal_string_node_structure(self):
        """Test that string literals produce correct node structure."""
        pool = Pool(seqs=["AAAA"])
        combined = pool + "SUFFIX"
        result = combined.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        # Find the literal node
        literal_node = None
        for node in result["graph"]["nodes"]:
            if node["type"] == "literal":
                literal_node = node
                break
        
        assert literal_node is not None
        assert literal_node["value_type"] == "str"
        assert literal_node["value"] == "SUFFIX"
        assert literal_node["parent_ids"] == []
    
    def test_slice_literal_node_structure(self):
        """Test that slice operations produce correct node structure."""
        pool = Pool(seqs=["ACGTACGT"])
        sliced = pool[2:6]
        result = sliced.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        # Find the slice literal node
        slice_node = None
        for node in result["graph"]["nodes"]:
            if node.get("value_type") == "slice":
                slice_node = node
                break
        
        assert slice_node is not None
        assert slice_node["type"] == "literal"
        assert "slice(2, 6," in slice_node["value"]


class TestComputationGraphOperations:
    """Tests for computation graph operations (+, *, slice)."""
    
    def test_concatenation_op_is_plus(self):
        """Test that concatenation uses '+' operation."""
        pool1 = Pool(seqs=["AAA"])
        pool2 = Pool(seqs=["TTT"])
        combined = pool1 + pool2
        result = combined.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        root = result["graph"]["nodes"][0]
        assert root["op"] == "+"
    
    def test_repetition_op_is_star(self):
        """Test that repetition uses '*' operation."""
        pool = Pool(seqs=["AAA"])
        repeated = pool * 3
        result = repeated.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        root = result["graph"]["nodes"][0]
        assert root["op"] == "*"
    
    def test_slice_op_is_slice(self):
        """Test that slicing uses 'slice' operation."""
        pool = Pool(seqs=["ACGTACGT"])
        sliced = pool[1:5]
        result = sliced.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        root = result["graph"]["nodes"][0]
        assert root["op"] == "slice"
    
    def test_concatenation_has_two_parents(self):
        """Test that concatenation node has exactly two parent IDs."""
        pool1 = Pool(seqs=["AAA"])
        pool2 = Pool(seqs=["TTT"])
        combined = pool1 + pool2
        result = combined.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        root = result["graph"]["nodes"][0]
        assert len(root["parent_ids"]) == 2
    
    def test_repetition_has_two_parents(self):
        """Test that repetition node has pool and int as parents."""
        pool = Pool(seqs=["AAA"])
        repeated = pool * 4
        result = repeated.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        root = result["graph"]["nodes"][0]
        assert len(root["parent_ids"]) == 2
        
        # One parent should be Pool, one should be int literal
        pool_parents = [n for n in result["graph"]["nodes"] 
                       if n["type"] == "Pool" and n["op"] is None]
        int_parents = [n for n in result["graph"]["nodes"] 
                      if n.get("value_type") == "int"]
        
        assert len(pool_parents) == 1
        assert len(int_parents) == 1
        assert int_parents[0]["value"] == 4
    
    def test_slice_has_two_parents(self):
        """Test that slice node has pool and slice literal as parents."""
        pool = Pool(seqs=["ACGTACGT"])
        sliced = pool[2:7]
        result = sliced.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        root = result["graph"]["nodes"][0]
        assert len(root["parent_ids"]) == 2


class TestComputationGraphParentRelationships:
    """Tests for parent-child relationships in computation graph."""
    
    def test_parent_ids_reference_valid_nodes(self):
        """Test that parent_ids reference existing node_ids."""
        pool1 = Pool(seqs=["AAA"])
        pool2 = Pool(seqs=["TTT"])
        pool3 = Pool(seqs=["GGG"])
        combined = pool1 + pool2 + pool3
        result = combined.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        nodes = result["graph"]["nodes"]
        all_node_ids = {n["node_id"] for n in nodes}
        
        for node in nodes:
            for parent_id in node["parent_ids"]:
                assert parent_id in all_node_ids, \
                    f"parent_id {parent_id} not found in nodes"
    
    def test_leaf_nodes_have_no_parents(self):
        """Test that leaf Pool nodes and literals have empty parent_ids."""
        pool1 = Pool(seqs=["AAA"])
        pool2 = Pool(seqs=["TTT"])
        combined = pool1 + pool2 + "SUFFIX"
        result = combined.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        for node in result["graph"]["nodes"]:
            # Base Pools (op is None) should have no parents
            if node["type"] == "Pool" and node["op"] is None:
                assert node["parent_ids"] == [], \
                    f"Leaf pool should have no parents: {node}"
            # Literals should have no parents
            if node["type"] == "literal":
                assert node["parent_ids"] == [], \
                    f"Literal should have no parents: {node}"
    
    def test_chained_concatenation_structure(self):
        """Test that chained concatenation creates proper tree structure."""
        # a + b + c creates ((a + b) + c)
        a = Pool(seqs=["AAA"])
        b = Pool(seqs=["BBB"])
        c = Pool(seqs=["CCC"])
        result_pool = a + b + c
        
        result = result_pool.generate_seqs(num_seqs=1, return_computation_graph=True)
        nodes = result["graph"]["nodes"]
        
        # Should have 5 nodes: 2 concatenation ops + 3 base pools
        assert len(nodes) == 5
        
        # Root should be a concatenation
        root = nodes[0]
        assert root["op"] == "+"
        
        # Count concatenation operations
        concat_count = sum(1 for n in nodes if n["op"] == "+")
        assert concat_count == 2
    
    def test_node_id_assignment_is_bfs(self):
        """Test that node IDs are assigned in BFS order from root."""
        pool1 = Pool(seqs=["AAA"])
        pool2 = Pool(seqs=["TTT"])
        combined = pool1 + pool2
        result = combined.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        nodes = result["graph"]["nodes"]
        
        # Root should be node 0
        root = nodes[0]
        assert root["node_id"] == 0
        assert root["op"] == "+"  # The concatenation operation
        
        # Children should have higher IDs
        for parent_id in root["parent_ids"]:
            assert parent_id > 0


class TestComputationGraphSharedAncestors:
    """Tests for handling shared ancestors (diamond patterns) in computation graph."""
    
    def test_shared_pool_appears_once(self):
        """Test that a Pool used multiple times appears only once in graph."""
        shared = Pool(seqs=["SHARED"])
        combined = shared + shared
        result = combined.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        nodes = result["graph"]["nodes"]
        
        # Should only have 2 nodes: the concat and the shared pool
        assert len(nodes) == 2
        
        # The shared pool should be referenced twice in parent_ids
        root = nodes[0]
        # Both parents point to the same node
        assert len(root["parent_ids"]) == 2
        assert root["parent_ids"][0] == root["parent_ids"][1]
    
    def test_diamond_pattern_deduplication(self):
        """Test that diamond pattern in graph is properly deduplicated."""
        base = Pool(seqs=["BASE"])
        left = base + "LEFT"
        right = base + "RIGHT"
        diamond = left + right
        
        result = diamond.generate_seqs(num_seqs=1, return_computation_graph=True)
        nodes = result["graph"]["nodes"]
        
        # Count base Pools (op=None, type=Pool)
        base_pools = [n for n in nodes if n["type"] == "Pool" and n["op"] is None]
        
        # The "BASE" pool should appear only once despite being used twice
        assert len(base_pools) == 1
        assert base_pools[0]["num_states"] == 1
    
    def test_triple_shared_reference(self):
        """Test pool referenced three times appears once."""
        shared = Pool(seqs=["X"])
        combined = shared + shared + shared
        result = combined.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        nodes = result["graph"]["nodes"]
        base_pools = [n for n in nodes if n["type"] == "Pool" and n["op"] is None]
        
        # Only one base pool despite being used 3 times
        assert len(base_pools) == 1


class TestComputationGraphSpecializedPools:
    """Tests for computation graph with specialized pool types."""
    
    def test_kmer_pool_op(self):
        """Test that KmerPool has op='kmer'."""
        from poolparty import KmerPool
        
        kmer = KmerPool(length=4, alphabet='dna')
        result = kmer.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        root = result["graph"]["nodes"][0]
        assert root["op"] == "kmer"
        assert root["num_states"] == 256  # 4^4
    
    def test_random_mutation_pool_op(self):
        """Test that RandomMutationPool has op='mutate'."""
        from poolparty import RandomMutationPool
        
        base = Pool(seqs=["AAAA"])
        mutated = RandomMutationPool(base, mutation_rate=0.1)
        result = mutated.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        root = result["graph"]["nodes"][0]
        assert root["op"] == "mutate"
        assert root["num_states"] == "infinite"
    
    def test_k_mutation_pool_op(self):
        """Test that KMutationPool has op='k_mutate'."""
        from poolparty import KMutationPool
        
        mutated = KMutationPool("ACGTACGT", k=2)
        result = mutated.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        root = result["graph"]["nodes"][0]
        assert root["op"] == "k_mutate"
    
    def test_insertion_scan_pool_op(self):
        """Test that InsertionScanPool has op='insertion_scan'."""
        from poolparty import InsertionScanPool
        
        scanner = InsertionScanPool("AAAAAAAAAA", "TTT")
        result = scanner.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        root = result["graph"]["nodes"][0]
        assert root["op"] == "insertion_scan"
    
    def test_mutation_pool_has_parent(self):
        """Test that mutation pool correctly references its parent."""
        from poolparty import RandomMutationPool
        
        base = Pool(seqs=["GGGGGGGG"], name="base_seq")
        mutated = RandomMutationPool(base, mutation_rate=0.2)
        result = mutated.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        root = result["graph"]["nodes"][0]
        assert len(root["parent_ids"]) == 1
        
        # Find parent node and verify it's the base pool
        parent_id = root["parent_ids"][0]
        parent_node = None
        for n in result["graph"]["nodes"]:
            if n["node_id"] == parent_id:
                parent_node = n
                break
        
        assert parent_node is not None
        assert parent_node["name"] == "base_seq"
        assert parent_node["op"] is None


class TestComputationGraphNodeSequences:
    """Tests for node_sequences in computation graph output."""
    
    def test_node_sequences_keys_are_strings(self):
        """Test that node_sequences keys are string node_ids."""
        pool = Pool(seqs=["ACGT"])
        result = pool.generate_library(num_seqs=3, return_computation_graph=True)
        
        for key in result["node_sequences"].keys():
            assert isinstance(key, str)
    
    def test_node_sequences_match_num_seqs(self):
        """Test that each node has correct number of sequences."""
        pool1 = Pool(seqs=["AAA"])
        pool2 = Pool(seqs=["TTT"])
        combined = pool1 + pool2
        
        num_seqs = 5
        result = combined.generate_seqs(num_seqs=num_seqs, return_computation_graph=True)
        
        for node_id, seqs in result["node_sequences"].items():
            if isinstance(seqs, list):
                assert len(seqs) == num_seqs
    
    def test_node_sequences_for_literal(self):
        """Test that literal nodes have appropriate sequences."""
        pool = Pool(seqs=["AAA"])
        combined = pool + "SUFFIX"
        result = combined.generate_seqs(num_seqs=3, return_computation_graph=True)
        
        # Find the literal node
        literal_node_id = None
        for n in result["graph"]["nodes"]:
            if n["type"] == "literal":
                literal_node_id = str(n["node_id"])
                break
        
        assert literal_node_id in result["node_sequences"]
        # Literal should be the constant value repeated
        seqs = result["node_sequences"][literal_node_id]
        if isinstance(seqs, list):
            assert all(s == "SUFFIX" for s in seqs)
        else:
            assert seqs == "SUFFIX"
    
    def test_root_node_sequences_match_output(self):
        """Test that root node sequences match the final output."""
        pool1 = Pool(seqs=["AAA"])
        pool2 = Pool(seqs=["TTT"])
        combined = pool1 + pool2
        
        result = combined.generate_seqs(num_seqs=3, return_computation_graph=True)
        
        root_seqs = result["node_sequences"]["0"]
        assert root_seqs == result["sequences"]
    
    def test_node_sequences_reflect_state_changes(self):
        """Test that sequential node sequences change across iterations."""
        pool = Pool(seqs=["AAA", "TTT", "GGG"], mode='sequential')
        result = pool.generate_library(num_seqs=3, return_computation_graph=True)
        
        node_seqs = result["node_sequences"]["0"]
        assert node_seqs[0] == "AAA"
        assert node_seqs[1] == "TTT"
        assert node_seqs[2] == "GGG"


class TestComputationGraphBackwardCompatibility:
    """
    Backward compatibility tests capturing current computation graph structure.
    
    These tests ensure that changes to the computation graph implementation
    maintain backward compatibility with existing graph structures.
    """
    
    def test_simple_library_graph_structure(self):
        """Test graph structure for: promoter + spacer + barcode."""
        from poolparty import KmerPool
        
        promoter = Pool(seqs=["TTGACA", "TTGATA"], name="promoter")
        spacer = "NNN"
        barcode = KmerPool(length=4, name="barcode")
        library = promoter + spacer + barcode
        
        result = library.generate_seqs(num_seqs=5, return_computation_graph=True)
        graph = result["graph"]
        
        # Expected structure:
        # - Root: + (concat promoter+spacer with barcode)
        # - Child 1: + (concat promoter with spacer)
        #   - Grandchild 1: Pool (promoter)
        #   - Grandchild 2: literal "NNN"
        # - Child 2: KmerPool (barcode)
        
        nodes = graph["nodes"]
        assert len(nodes) == 5
        
        # Verify operation types (only Pool nodes have 'op')
        ops = {n["node_id"]: n.get("op") for n in nodes if n["type"] == "Pool"}
        root_op = ops[0]
        assert root_op == "+"
        
        # Find named pools
        named_pools = {n["name"]: n for n in nodes if n.get("name")}
        assert "promoter" in named_pools
        assert "barcode" in named_pools
        
        # Verify promoter structure
        assert named_pools["promoter"]["num_states"] == 2
        assert named_pools["promoter"]["op"] is None
        
        # Verify barcode structure
        assert named_pools["barcode"]["op"] == "kmer"
        assert named_pools["barcode"]["num_states"] == 256
    
    def test_mutated_pool_library_graph_structure(self):
        """Test graph structure for: mutated + spacer + barcode."""
        from poolparty import KmerPool, RandomMutationPool
        
        promoter = Pool(seqs=["TTGACA", "TTGATA"])
        mutated = RandomMutationPool(promoter, mutation_rate=0.1)
        spacer = "NNN"
        barcode = KmerPool(length=4)
        library = mutated + spacer + barcode
        
        result = library.generate_seqs(num_seqs=5, return_computation_graph=True)
        graph = result["graph"]
        nodes = graph["nodes"]
        
        # Expected: 6 nodes
        # Root (+), intermediate (+), mutate, promoter pool, literal NNN, kmer
        assert len(nodes) == 6
        
        # Verify mutation pool is present (only Pool nodes have 'op')
        mutation_nodes = [n for n in nodes if n["type"] == "Pool" and n.get("op") == "mutate"]
        assert len(mutation_nodes) == 1
        assert mutation_nodes[0]["num_states"] == "infinite"
        
        # Verify mutation pool's parent is the base promoter
        mut_node = mutation_nodes[0]
        assert len(mut_node["parent_ids"]) == 1
    
    def test_complex_composite_graph_structure(self):
        """
        Test graph structure matching notebook example:
        spacer_1 + promoter + spacer_2*2 + spacer_3 + barcode
        
        This captures the exact structure from the notebook for backward compatibility.
        """
        from poolparty import (
            Pool, KmerPool, RandomMutationPool, 
            InsertionScanPool, KMutationPool
        )
        
        promoter = Pool([
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
        ], name="promoter")
        
        spacer_1 = RandomMutationPool(
            "CCCCCCCCCCCCCCCCCCC", 
            mutation_rate=0.1,
            name="spacer_1"
        )
        spacer_2 = InsertionScanPool(
            "DDDDDDDDDDDDDDDDDDDD", 
            "eeee", 
            mode="random",
            name="spacer_2"
        )
        spacer_3 = KMutationPool(
            "FFFFFFFFFFFFF", 
            k=2, 
            adjacent=True,
            mode="sequential",
            name="spacer_3"
        )
        barcode = KmerPool(length=6, alphabet="dna", name="barcode")
        
        library = spacer_1 + promoter + spacer_2 * 2 + spacer_3 + barcode
        
        result = library.generate_seqs(num_seqs=10, return_computation_graph=True)
        graph = result["graph"]
        nodes = graph["nodes"]
        
        # Find all named pools
        named_pools = {}
        for n in nodes:
            if n.get("name"):
                named_pools[n["name"]] = n
        
        # Verify all named pools are present
        assert "promoter" in named_pools
        assert "spacer_1" in named_pools
        assert "spacer_2" in named_pools
        assert "spacer_3" in named_pools
        assert "barcode" in named_pools
        
        # Verify operation types for each specialized pool
        assert named_pools["spacer_1"]["op"] == "mutate"
        assert named_pools["spacer_1"]["num_states"] == "infinite"
        
        assert named_pools["spacer_2"]["op"] == "insertion_scan"
        
        assert named_pools["spacer_3"]["op"] == "k_mutate"
        assert named_pools["spacer_3"]["mode"] == "sequential"
        
        assert named_pools["barcode"]["op"] == "kmer"
        assert named_pools["barcode"]["num_states"] == 4096  # 4^6
        
        assert named_pools["promoter"]["op"] is None
        assert named_pools["promoter"]["num_states"] == 2
        
        # Verify repetition node exists (spacer_2 * 2)
        rep_nodes = [n for n in nodes if n["type"] == "Pool" and n.get("op") == "*"]
        assert len(rep_nodes) == 1
        
        # Verify the int literal for repetition
        int_literals = [n for n in nodes if n.get("value_type") == "int"]
        assert len(int_literals) == 1
        assert int_literals[0]["value"] == 2
    
    def test_tiling_library_graph_structure(self):
        """Test graph structure for tiling/scanning library design."""
        from poolparty import Pool, KmerPool, RandomMutationPool, InsertionScanPool
        
        promoter = Pool(["TATATAA"], name="Promoter")
        upstream_base = Pool(["GCGCGCGCGC"], name="Upstream_Base")
        mutated_rbs = RandomMutationPool(
            upstream_base, 
            mutation_rate=0.25, 
            mark_changes=True, 
            name="Mutated_RBS"
        )
        backbone = Pool(["GGGGGGGGGGGGGGGG" * 5], name="Backbone")
        linker = Pool(["AAA"], name="Linker")
        scan_pool = InsertionScanPool(
            backbone, 
            linker, 
            start=2,  
            mark_changes=True,
            mode="sequential", 
            name="Linker_Scan"
        )
        barcode = KmerPool(length=5, alphabet="dna", name="Barcode")
        terminator = Pool(["TTTTT"], name="Terminator")
        
        library = promoter + mutated_rbs + scan_pool + barcode + terminator
        
        result = library.generate_seqs(num_seqs=20, return_computation_graph=True)
        graph = result["graph"]
        nodes = graph["nodes"]
        
        # Collect named pools
        named_pools = {n["name"]: n for n in nodes if n.get("name")}
        
        # Verify all components are present
        expected_names = [
            "Promoter", "Upstream_Base", "Mutated_RBS",
            "Backbone", "Linker", "Linker_Scan", "Barcode", "Terminator"
        ]
        for name in expected_names:
            assert name in named_pools, f"Missing pool: {name}"
        
        # Verify hierarchical relationship: Mutated_RBS -> Upstream_Base
        mutated_node = named_pools["Mutated_RBS"]
        assert len(mutated_node["parent_ids"]) == 1
        parent_id = mutated_node["parent_ids"][0]
        parent_node = next(n for n in nodes if n["node_id"] == parent_id)
        assert parent_node["name"] == "Upstream_Base"
        
        # Verify InsertionScanPool has background and linker as parents
        scan_node = named_pools["Linker_Scan"]
        assert scan_node["op"] == "insertion_scan"
        assert scan_node["mode"] == "sequential"
        assert len(scan_node["parent_ids"]) == 2


class TestComputationGraphInfiniteStates:
    """Tests for handling infinite states in computation graph."""
    
    def test_infinite_states_serialization(self):
        """Test that infinite states are serialized as 'infinite' string."""
        from poolparty import RandomMutationPool
        
        base = Pool(seqs=["ACGT"])
        mutated = RandomMutationPool(base, mutation_rate=0.5)
        result = mutated.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        root = result["graph"]["nodes"][0]
        assert root["num_states"] == "infinite"
        assert isinstance(root["num_states"], str)
    
    def test_finite_states_serialization(self):
        """Test that finite states are serialized as integers."""
        from poolparty import KmerPool
        
        kmer = KmerPool(length=3)
        result = kmer.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        root = result["graph"]["nodes"][0]
        assert root["num_states"] == 64  # 4^3
        assert isinstance(root["num_states"], int)
    
    def test_composite_with_infinite_states(self):
        """Test composite pool with infinite state component."""
        from poolparty import RandomMutationPool
        
        base = Pool(seqs=["AAAA"])
        mutated = RandomMutationPool(base, mutation_rate=0.1)
        combined = "PREFIX" + mutated + "SUFFIX"
        
        result = combined.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        # Find the mutation node (only Pool nodes have 'op')
        mut_node = None
        for n in result["graph"]["nodes"]:
            if n["type"] == "Pool" and n.get("op") == "mutate":
                mut_node = n
                break
        
        assert mut_node is not None
        assert mut_node["num_states"] == "infinite"


class TestComputationGraphEdgeCases:
    """Tests for edge cases in computation graph."""
    
    def test_empty_graph_not_possible(self):
        """Test that there's always at least one node."""
        pool = Pool(seqs=["A"])
        result = pool.generate_library(num_seqs=1, return_computation_graph=True)
        
        assert len(result["graph"]["nodes"]) >= 1
    
    def test_deeply_nested_graph(self):
        """Test deeply nested computation graph."""
        pool = Pool(seqs=["A"])
        
        # Create deep nesting with the same literal "X"
        # The "X" literal gets deduplicated since it's the same object
        for _ in range(10):
            pool = pool + "X"
        
        result = pool.generate_library(num_seqs=1, return_computation_graph=True)
        
        # Count nodes: 10 concat ops + 1 base pool + 1 shared "X" literal = 12 nodes
        # (The string "X" is deduplicated because identical strings are interned in Python)
        assert len(result["graph"]["nodes"]) == 12
        
        # Verify there's exactly one literal "X" node (shared across all concats)
        literal_nodes = [n for n in result["graph"]["nodes"] if n["type"] == "literal"]
        assert len(literal_nodes) == 1
        assert literal_nodes[0]["value"] == "X"
        
        # All parent references should be valid
        all_ids = {n["node_id"] for n in result["graph"]["nodes"]}
        for n in result["graph"]["nodes"]:
            for pid in n["parent_ids"]:
                assert pid in all_ids
    
    def test_wide_graph(self):
        """Test wide computation graph with many siblings."""
        pools = [Pool(seqs=[f"SEQ{i}"]) for i in range(5)]
        combined = pools[0]
        for p in pools[1:]:
            combined = combined + p
        
        result = combined.generate_library(num_seqs=1, return_computation_graph=True)
        
        # Should have 4 concat ops + 5 base pools = 9 nodes
        assert len(result["graph"]["nodes"]) == 9
    
    def test_mixed_string_and_pool_parents(self):
        """Test graph with mix of string and Pool parents."""
        pool = Pool(seqs=["MIDDLE"])
        combined = "START" + pool + "END"
        
        result = combined.generate_seqs(num_seqs=1, return_computation_graph=True)
        nodes = result["graph"]["nodes"]
        
        # Count node types
        pool_nodes = [n for n in nodes if n["type"] == "Pool"]
        literal_nodes = [n for n in nodes if n["type"] == "literal"]
        
        assert len(pool_nodes) == 3  # 2 concat ops + 1 base pool
        assert len(literal_nodes) == 2  # START and END
    
    def test_radd_graph_structure(self):
        """Test that reverse addition creates correct graph."""
        pool = Pool(seqs=["POOL"])
        combined = "PREFIX" + pool  # Uses __radd__
        
        result = combined.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        root = result["graph"]["nodes"][0]
        assert root["op"] == "+"
        assert len(root["parent_ids"]) == 2
    
    def test_rmul_graph_structure(self):
        """Test that reverse multiplication creates correct graph."""
        pool = Pool(seqs=["ABC"])
        repeated = 3 * pool  # Uses __rmul__
        
        result = repeated.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        root = result["graph"]["nodes"][0]
        assert root["op"] == "*"
        assert len(root["parent_ids"]) == 2


class TestComputationGraphDeterminism:
    """Tests for deterministic behavior of computation graph."""
    
    def test_graph_structure_is_deterministic(self):
        """Test that same pool always produces same graph structure."""
        def create_library():
            from poolparty import KmerPool
            p1 = Pool(seqs=["AAA"])
            p2 = Pool(seqs=["TTT"])
            k = KmerPool(length=2)
            return p1 + p2 + k
        
        lib1 = create_library()
        lib2 = create_library()
        
        result1 = lib1.generate_seqs(num_seqs=1, return_computation_graph=True)
        result2 = lib2.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        # Graph structures should be identical
        nodes1 = result1["graph"]["nodes"]
        nodes2 = result2["graph"]["nodes"]
        
        assert len(nodes1) == len(nodes2)
        
        for n1, n2 in zip(nodes1, nodes2):
            assert n1["type"] == n2["type"]
            assert n1["op"] == n2["op"]
            assert len(n1["parent_ids"]) == len(n2["parent_ids"])
    
    def test_node_id_assignment_is_consistent(self):
        """Test that node_id assignment is consistent across calls."""
        pool1 = Pool(seqs=["AAA"])
        pool2 = Pool(seqs=["TTT"])
        combined = pool1 + pool2
        
        result1 = combined.generate_seqs(num_seqs=1, return_computation_graph=True)
        result2 = combined.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        # Node IDs should be assigned the same way
        for n1, n2 in zip(result1["graph"]["nodes"], result2["graph"]["nodes"]):
            assert n1["node_id"] == n2["node_id"]


class TestComputationGraphWithModes:
    """Tests for computation graph with different modes."""
    
    def test_random_mode_in_graph(self):
        """Test that random mode is captured in graph."""
        pool = Pool(seqs=["AAA", "TTT"], mode='random')
        result = pool.generate_library(num_seqs=1, return_computation_graph=True)
        
        assert result["graph"]["nodes"][0]["mode"] == "random"
    
    def test_sequential_mode_in_graph(self):
        """Test that sequential mode is captured in graph."""
        pool = Pool(seqs=["AAA", "TTT"], mode='sequential')
        result = pool.generate_library(num_seqs=2, return_computation_graph=True)
        
        assert result["graph"]["nodes"][0]["mode"] == "sequential"
    
    def test_mixed_modes_in_graph(self):
        """Test graph with both random and sequential mode pools."""
        seq_pool = Pool(seqs=["AAA", "TTT"], mode='sequential', name="seq")
        rand_pool = Pool(seqs=["GGG", "CCC"], mode='random', name="rand")
        combined = seq_pool + rand_pool
        
        result = combined.generate_seqs(num_seqs=4, return_computation_graph=True)
        nodes = result["graph"]["nodes"]
        
        named = {n["name"]: n for n in nodes if n.get("name")}
        assert named["seq"]["mode"] == "sequential"
        assert named["rand"]["mode"] == "random"


class TestComputationGraphLiteralDeduplication:
    """Tests for literal deduplication behavior in computation graph."""
    
    def test_identical_string_literals_deduplicated(self):
        """Test that identical string literals are deduplicated."""
        pool1 = Pool(seqs=["AAA"])
        pool2 = Pool(seqs=["TTT"])
        # Same literal "X" used twice
        combined = pool1 + "X" + pool2 + "X"
        
        result = combined.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        # Count string literal nodes
        literal_nodes = [n for n in result["graph"]["nodes"] 
                        if n["type"] == "literal" and n.get("value") == "X"]
        
        # Only one "X" literal despite being used twice
        assert len(literal_nodes) == 1
    
    def test_different_string_literals_not_deduplicated(self):
        """Test that different string literals are not deduplicated."""
        pool = Pool(seqs=["AAA"])
        combined = "START" + pool + "MIDDLE" + pool + "END"
        
        result = combined.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        # Count unique literal values
        literal_values = {n["value"] for n in result["graph"]["nodes"] 
                         if n["type"] == "literal"}
        
        # Should have START, MIDDLE, END
        assert "START" in literal_values
        assert "MIDDLE" in literal_values
        assert "END" in literal_values
    
    def test_int_literal_for_repetition(self):
        """Test that int literals for repetition are captured correctly."""
        pool = Pool(seqs=["AAA"])
        repeated = pool * 5
        
        result = repeated.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        int_nodes = [n for n in result["graph"]["nodes"] 
                    if n["type"] == "literal" and n.get("value_type") == "int"]
        
        assert len(int_nodes) == 1
        assert int_nodes[0]["value"] == 5


class TestComputationGraphComplexPatterns:
    """Tests for complex computation graph patterns."""
    
    def test_nested_repetition(self):
        """Test nested repetition operations."""
        pool = Pool(seqs=["A"])
        # (pool * 2) * 3
        repeated = (pool * 2) * 3
        
        result = repeated.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        # Count repetition operations
        rep_nodes = [n for n in result["graph"]["nodes"] 
                    if n["type"] == "Pool" and n.get("op") == "*"]
        
        assert len(rep_nodes) == 2
        
        # Verify output sequence
        assert result["sequences"][0] == "AAAAAA"  # 2*3 = 6 A's
    
    def test_mixed_operations(self):
        """Test graph with concatenation, repetition, and slice."""
        base = Pool(seqs=["ACGTACGT"])
        sliced = base[2:6]
        repeated = sliced * 2
        combined = "START" + repeated + "END"
        
        result = combined.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        # Verify all operations present
        ops = {n.get("op") for n in result["graph"]["nodes"] if n["type"] == "Pool"}
        
        assert "+" in ops
        assert "*" in ops
        assert "slice" in ops
        assert None in ops  # Base pool
    
    def test_insertion_scan_with_pool_parents(self):
        """Test InsertionScanPool with Pool objects as parents."""
        from poolparty import InsertionScanPool
        
        background = Pool(seqs=["GGGGGGGGGG"], name="background")
        insert = Pool(seqs=["AAA", "TTT"], name="insert")
        scanner = InsertionScanPool(background, insert, name="scanner")
        
        result = scanner.generate_seqs(num_seqs=1, return_computation_graph=True)
        nodes = result["graph"]["nodes"]
        
        # Find the scanner node
        scanner_node = next(n for n in nodes if n.get("name") == "scanner")
        
        # Should have 2 parents: background and insert
        assert len(scanner_node["parent_ids"]) == 2
        
        # Verify parents exist
        parent_names = set()
        for pid in scanner_node["parent_ids"]:
            parent = next(n for n in nodes if n["node_id"] == pid)
            if parent.get("name"):
                parent_names.add(parent["name"])
        
        assert "background" in parent_names
        assert "insert" in parent_names
    
    def test_k_mutation_with_string_parent(self):
        """Test KMutationPool with string as parent."""
        from poolparty import KMutationPool
        
        # String parent (not Pool)
        mutated = KMutationPool("ACGTACGT", k=1, name="k_mut")
        
        result = mutated.generate_seqs(num_seqs=1, return_computation_graph=True)
        nodes = result["graph"]["nodes"]
        
        mut_node = next(n for n in nodes if n.get("name") == "k_mut")
        
        # Should have 1 parent (the string)
        assert len(mut_node["parent_ids"]) == 1
        
        # Parent should be a literal string
        parent_id = mut_node["parent_ids"][0]
        parent = next(n for n in nodes if n["node_id"] == parent_id)
        assert parent["type"] == "literal"
        assert parent["value_type"] == "str"
        assert parent["value"] == "ACGTACGT"


class TestComputationGraphOutputFormat:
    """Tests for the exact output format of computation graphs."""
    
    def test_graph_dict_has_nodes_key(self):
        """Test that graph dict always has 'nodes' key."""
        pool = Pool(seqs=["ACGT"])
        result = pool.generate_library(num_seqs=1, return_computation_graph=True)
        
        assert "nodes" in result["graph"]
        assert isinstance(result["graph"]["nodes"], list)
    
    def test_pool_node_has_all_required_fields(self):
        """Test that Pool nodes have all required fields."""
        pool = Pool(seqs=["ACGT"], name="test", mode='sequential')
        result = pool.generate_library(num_seqs=1, return_computation_graph=True)
        
        node = result["graph"]["nodes"][0]
        
        required_fields = ["node_id", "type", "op", "num_states", "mode", "name", "parent_ids"]
        for field in required_fields:
            assert field in node, f"Missing field: {field}"
    
    def test_literal_node_has_all_required_fields(self):
        """Test that literal nodes have all required fields."""
        pool = Pool(seqs=["ACGT"])
        combined = pool + "SUFFIX"
        result = combined.generate_seqs(num_seqs=1, return_computation_graph=True)
        
        literal_node = next(n for n in result["graph"]["nodes"] if n["type"] == "literal")
        
        required_fields = ["node_id", "type", "value_type", "value", "parent_ids"]
        for field in required_fields:
            assert field in literal_node, f"Missing field: {field}"
    
    def test_node_sequences_format(self):
        """Test the format of node_sequences in output."""
        pool1 = Pool(seqs=["AAA"])
        pool2 = Pool(seqs=["TTT"])
        combined = pool1 + pool2
        
        result = combined.generate_seqs(num_seqs=3, return_computation_graph=True)
        
        # All node_ids should be present as string keys
        for node in result["graph"]["nodes"]:
            node_id_str = str(node["node_id"])
            assert node_id_str in result["node_sequences"]
    
    def test_return_dict_structure(self):
        """Test the complete structure of the return dict."""
        pool = Pool(seqs=["ACGT"])
        result = pool.generate_library(num_seqs=2, return_computation_graph=True)
        
        # Must have exactly these three keys
        assert set(result.keys()) == {"sequences", "graph", "node_sequences"}
        
        # sequences should be a list
        assert isinstance(result["sequences"], list)
        assert len(result["sequences"]) == 2
        
        # graph should have nodes
        assert "nodes" in result["graph"]
        
        # node_sequences should be a dict
        assert isinstance(result["node_sequences"], dict)
