"""Tests for the Party context manager and core architecture.

Pool operators now work on Counters:
- pool1 + pool2: Stack (union of states)
- pool * n: Repeat (repeat states n times)
- pool[start:stop]: State slice (select subset of states)

For sequence operations, use join(), seq_slice(), etc.
"""

import pytest
import poolparty as pp
from poolparty import join
from poolparty.operations.seq_slice import seq_slice
from poolparty.operations.stack import stack, StackOp
from poolparty.counter import ConflictingStateAssignmentError


class TestBasicUsage:
    """Test basic Party usage patterns."""
    
    def test_simple_from_seqs(self):
        """Test creating a simple pool from sequences."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA', 'TTT', 'GGG']).named('seq')
        
        df = pool.generate_seqs(num_seqs=3)
        assert len(df) == 3
        assert 'seq' in df.columns
        assert list(df['seq']) == ['AAA', 'TTT', 'GGG']
    
    def test_get_kmers_sequential(self):
        """Test generating k-mers in sequential mode."""
        with pp.Party() as party:
            pool = pp.get_kmers(length=2, alphabet='AB', mode='sequential').named('kmer')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        assert len(df) == 4  # 2^2 = 4 k-mers
        assert list(df['seq']) == ['AA', 'AB', 'BA', 'BB']
    
    def test_get_kmers_random(self):
        """Test generating k-mers in random mode."""
        with pp.Party() as party:
            pool = pp.get_kmers(length=5, alphabet='dna', mode='random').named('kmer')
        
        df = pool.generate_seqs(num_seqs=10, seed=42)
        assert len(df) == 10
        # All should be valid 5-mers of DNA
        for kmer in df['seq']:
            assert len(kmer) == 5
            assert all(c in 'ACGT' for c in kmer)
    
    def test_join_pools(self):
        """Test joining pools using join()."""
        with pp.Party() as party:
            left = pp.from_seqs(['AAA'])
            right = pp.from_seqs(['TTT'])
            oligo = join([left, '...', right]).named('oligo')
        
        df = oligo.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'AAA...TTT'
    
    def test_join_with_kmer(self):
        """Test joining a sequence with a random barcode."""
        with pp.Party() as party:
            seq_pool = pp.from_seqs(['ACGT'])
            barcode = pp.get_kmers(length=4, alphabet='dna', mode='random')
            oligo = join([seq_pool, '...', barcode]).named('oligo')
        
        df = oligo.generate_seqs(num_seqs=5, seed=42)
        assert len(df) == 5
        for s in df['seq']:
            assert s.startswith('ACGT...')
            assert len(s) == 4 + 3 + 4  # seq + ... + barcode


class TestMutationScan:
    """Test mutation scanning operations."""
    
    def test_single_mutation_sequential(self):
        """Test single mutation in sequential mode."""
        with pp.Party() as party:
            mutants = pp.mutation_scan('ACGT', k=1, mode='sequential').named('mutant')
        
        df = mutants.generate_seqs(num_complete_iterations=1)
        # 4 positions * 3 mutations each = 12 mutants
        assert len(df) == 12
        
        # Check all are single mutations
        for mutant in df['seq']:
            assert len(mutant) == 4
            diffs = sum(1 for a, b in zip('ACGT', mutant) if a != b)
            assert diffs == 1
    
    def test_double_mutation_sequential(self):
        """Test double mutation in sequential mode."""
        with pp.Party() as party:
            mutants = pp.mutation_scan('ACGT', k=2, mode='sequential').named('mutant')
        
        df = mutants.generate_seqs(num_complete_iterations=1)
        # C(4,2) * 3^2 = 6 * 9 = 54 mutants
        assert len(df) == 54
    
    def test_mutation_scan_random(self):
        """Test mutation scan in random mode."""
        with pp.Party() as party:
            mutants = pp.mutation_scan('ACGTACGT', k=1, mode='random').named('mutant')
        
        df = mutants.generate_seqs(num_seqs=10, seed=42)
        assert len(df) == 10


class TestBreakpointScan:
    """Test breakpoint scanning operations."""
    
    def test_single_breakpoint(self):
        """Test splitting with one breakpoint."""
        with pp.Party() as party:
            left, right = pp.breakpoint_scan('ACGT', num_breakpoints=1, mode='sequential')
            left = left.named('left')
            right = right.named('right')
        
        df = left.generate_seqs(num_complete_iterations=1, aux_pools=[right])
        # 3 possible breakpoint positions
        assert len(df) == 3
        
        # Check all splits are valid
        for _, row in df.iterrows():
            assert row['left.seq'] + row['right.seq'] == 'ACGT'
    
    def test_double_breakpoint(self):
        """Test splitting with two breakpoints."""
        with pp.Party() as party:
            left, mid, right = pp.breakpoint_scan('ACGTACGT', num_breakpoints=2, 
                                                   mode='sequential')
            left = left.named('left')
            mid = mid.named('mid')
            right = right.named('right')
        
        df = left.generate_seqs(num_complete_iterations=1, aux_pools=[mid, right])
        
        # Check all splits are valid
        for _, row in df.iterrows():
            assert row['left.seq'] + row['mid.seq'] + row['right.seq'] == 'ACGTACGT'
    
    def test_breakpoint_with_mutation_join_raises_conflict(self):
        """Test combining breakpoint scan with mutation raises conflict.
        
        Breakpoint scan creates synchronized pools that share a counter.
        Concatenating one with a mutation of the other creates a counter
        graph with conflicting state assignments.
        """
        with pp.Party() as party:
            left, right = pp.breakpoint_scan('ACGTACGT', num_breakpoints=1, 
                                              mode='sequential')
            mutated_right = pp.mutation_scan(right, k=1, mode='sequential')
            oligo = join([left, mutated_right]).named('oligo')
        
        with pytest.raises(ConflictingStateAssignmentError):
            oligo.generate_seqs(num_seqs=10)


class TestStatePersistence:
    """Test that Pool maintains state between generate_seqs() calls."""
    
    def test_state_continues(self):
        """Test that sequential iteration continues across generate_seqs() calls."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C', 'D'], mode='sequential').named('seq')
        
        df1 = pool.generate_seqs(num_seqs=2)
        df2 = pool.generate_seqs(num_seqs=2)
        
        assert list(df1['seq']) == ['A', 'B']
        assert list(df2['seq']) == ['C', 'D']
    
    def test_reset_via_init_state(self):
        """Test that init_state=0 resets the state."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], mode='sequential').named('seq')
        
        df1 = pool.generate_seqs(num_seqs=2)
        df2 = pool.generate_seqs(num_seqs=2, init_state=0)
        
        assert list(df1['seq']) == ['A', 'B']
        assert list(df2['seq']) == ['A', 'B']  # Reset to start
    
    def test_init_state(self):
        """Test starting from a specific state."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C', 'D'], mode='sequential').named('seq')
        
        df = pool.generate_seqs(num_seqs=2, init_state=2)
        assert list(df['seq']) == ['C', 'D']


class TestMixedModes:
    """Test combining sequential and random operations."""
    
    def test_sequential_with_random_barcode(self):
        """Test sequential mutations with random barcodes."""
        with pp.Party() as party:
            mutants = pp.mutation_scan('ACGT', k=1, mode='sequential')
            barcode = pp.get_kmers(length=4, alphabet='dna', mode='random')
            oligo = join([mutants, '.', barcode]).named('oligo')
        
        df = oligo.generate_seqs(num_complete_iterations=1, seed=42)
        assert len(df) == 12  # 12 single mutants
        
        # Check that barcodes are varied (random)
        barcodes = [s.split('.')[-1] for s in df['seq']]
        assert len(set(barcodes)) > 1  # Not all the same


class TestDesignCards:
    """Test design card metadata."""
    
    def test_mutation_scan_metadata(self):
        """Test that mutation scan includes position and char metadata."""
        with pp.Party() as party:
            mutants = pp.mutation_scan('ACGT', k=1, mode='sequential', op_name='mutate').named('mutant')
        
        df = mutants.generate_seqs(num_seqs=3)
        
        assert 'mutant.op.key.positions' in df.columns
        assert 'mutant.op.key.wt_chars' in df.columns
        assert 'mutant.op.key.mut_chars' in df.columns
    
    def test_from_seqs_metadata(self):
        """Test that from_seqs includes name and index metadata."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA', 'TTT'], names=['seq_a', 'seq_b'], op_name='seqs').named('myseq')
        
        df = pool.generate_seqs(num_seqs=2)
        
        assert 'myseq.op.key.seq_name' in df.columns
        assert 'myseq.op.key.seq_index' in df.columns
        assert list(df['myseq.op.key.seq_name']) == ['seq_a', 'seq_b']


class TestSeqSliceOp:
    """Test sequence slicing operations using seq_slice()."""
    
    def test_seq_slice_with_range(self):
        """Test sequence slicing with a range."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGTACGT'])
            sliced = seq_slice(pool, slice(0, 4)).named('sliced')
        
        df = sliced.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'ACGT'
    
    def test_seq_slice_negative_index(self):
        """Test sequence slicing with negative index."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGTACGT'])
            last = seq_slice(pool, -1).named('last')
        
        df = last.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'T'
    
    def test_seq_slice_with_step(self):
        """Test sequence slicing with step."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGH'])
            every_other = seq_slice(pool, slice(None, None, 2)).named('every_other')
        
        df = every_other.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'ACEG'
    
    def test_seq_slice_reverse(self):
        """Test reversing a sequence with slice."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'])
            reversed_seq = seq_slice(pool, slice(None, None, -1)).named('reversed')
        
        df = reversed_seq.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'TGCA'
    
    def test_seq_slice_with_mutation(self):
        """Test combining seq_slice with mutation."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGTACGT'])
            first_half = seq_slice(pool, slice(0, 4))
            mutated = pp.mutation_scan(first_half, k=1, mode='sequential').named('mutated')
        
        df = mutated.generate_seqs(num_seqs=3)
        # All should be 4 characters
        for s in df['seq']:
            assert len(s) == 4


class TestStateSliceOp:
    """Test state slicing operations using pool[start:stop]."""
    
    def test_state_slice_with_range(self):
        """Test state slicing with a range."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C', 'D', 'E'])
            sliced = pool[1:4].named('seq')  # States 1, 2, 3 -> B, C, D
        
        df = sliced.generate_seqs(num_complete_iterations=1)
        assert list(df['seq']) == ['B', 'C', 'D']
    
    def test_state_slice_single(self):
        """Test state slicing with single index."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'])
            single = pool[1].named('seq')  # State 1 -> B
        
        df = single.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'B'
    
    def test_state_slice_negative(self):
        """Test state slicing with negative index."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'])
            last = pool[-1].named('seq')  # Last state -> C
        
        df = last.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'C'


class TestStackOp:
    """Test stacking operations using pool1 + pool2."""
    
    def test_stack_two_pools(self):
        """Test stacking two pools."""
        with pp.Party() as party:
            a = pp.from_seqs(['A', 'B'])
            b = pp.from_seqs(['X', 'Y'])
            stacked = (a + b).named('seq')
        
        df = stacked.generate_seqs(num_complete_iterations=1)
        assert list(df['seq']) == ['A', 'B', 'X', 'Y']
    
    def test_stack_num_states(self):
        """Test that stack num_states is sum of parent states."""
        with pp.Party() as party:
            a = pp.from_seqs(['A', 'B'])  # 2 states
            b = pp.from_seqs(['X', 'Y', 'Z'])  # 3 states
            stacked = a + b
            assert stacked.num_states == 5


class TestRepeatOp:
    """Test repeat operations using pool * n."""
    
    def test_repeat_pool(self):
        """Test repeating a pool."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B'])
            repeated = (pool * 2).named('seq')
        
        df = repeated.generate_seqs(num_complete_iterations=1)
        assert list(df['seq']) == ['A', 'B', 'A', 'B']
    
    def test_repeat_num_states(self):
        """Test that repeat num_states is original * n."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'])  # 3 states
            repeated = pool * 4
            assert repeated.num_states == 12


class TestErrors:
    """Test error handling."""
    
    def test_nested_party_error(self):
        """Test error for nested Party contexts."""
        with pp.Party():
            with pytest.raises(RuntimeError, match="Nested Party contexts"):
                with pp.Party():
                    pass
    
    def test_num_seqs_required(self):
        """Test error when neither num_seqs nor num_complete_iterations given."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA']).named('seq')
        
        with pytest.raises(ValueError, match="Must specify"):
            pool.generate_seqs()
    
    def test_pool_plus_string_raises(self):
        """Test that pool + string raises error (use join instead)."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'])
            with pytest.raises(Exception):  # beartype raises roar.BeartypeCallHintParamViolation
                _ = pool + 'TTT'


class TestCounterManagerIntegration:
    """Test CounterManager integration with Party."""
    
    def test_counter_manager_accessible(self):
        """Test that party.counter_manager is accessible."""
        with pp.Party() as party:
            assert party.counter_manager is not None
            assert isinstance(party.counter_manager, pp.CounterManager)
    
    def test_counters_registered_from_seqs(self):
        """Test that counters are registered when operations are created."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C']).named('seq')
            
            # Should have registered counters
            names = party.counter_manager.get_all_names()
            assert len(names) > 0
    
    def test_counters_registered_mutation_scan(self):
        """Test that mutation_scan counters are registered."""
        with pp.Party() as party:
            mutants = pp.mutation_scan('ACGT', k=1, mode='sequential').named('mutant')
            
            # Should have multiple counters registered
            names = party.counter_manager.get_all_names()
            assert len(names) > 0
    
    def test_test_iteration_works(self, capsys):
        """Test that test_iteration() works on registered counters."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C']).named('seq')
            
            # test_iteration should work on the pool's counter
            df = party.counter_manager.test_iteration(
                pool.counter, 
                display_df=False,
                return_df=True
            )
            
            # Should have iterated through all states
            assert len(df) == pool.counter.num_states
    
    def test_print_graph_works(self, capsys):
        """Test that print_graph() shows the counter DAG."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C']).named('seq')
            
            # Should not raise an error
            party.counter_manager.print_graph()
            
            # Check that something was printed
            captured = capsys.readouterr()
            assert len(captured.out) > 0
    
    def test_counter_manager_deactivated_on_exit(self):
        """Test that CounterManager is deactivated when exiting Party context."""
        from poolparty.counter import CounterManager
        
        with pp.Party() as party:
            # Inside context, manager should be active
            assert CounterManager._active_manager is party.counter_manager
        
        # Outside context, manager should be deactivated
        assert CounterManager._active_manager is None
    
    def test_complex_dag_counters_registered(self):
        """Test that counters from a complex DAG are all registered."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])
            mutants = pp.mutation_scan(seq, k=1, mode='sequential')
            barcode = pp.get_kmers(length=4, alphabet='dna', mode='random')
            oligo = pp.join([mutants, '---', barcode]).named('oligo')
            
            # Should have multiple counters registered
            names = party.counter_manager.get_all_names()
            # At minimum: from_seqs counter, mutation_scan counter, get_kmers counter,
            # join counter, and their pool counters
            assert len(names) >= 4


class TestPrintGraph:
    """Test Party.print_graph() and Pool.print_tree() visualization."""
    
    def test_print_graph_simple_clean(self, capsys):
        """Test print_graph() with clean style (default)."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], pool_name='mypool')
        
        party.print_graph()  # clean is default
        captured = capsys.readouterr()
        
        # Should contain pool name followed by (pool, ...)
        assert 'mypool (pool,' in captured.out
        # Should contain operation with brackets
        assert '[op=' in captured.out
        # Should show n= for num_states in clean mode
        assert 'n=' in captured.out
    
    def test_print_graph_minimal(self, capsys):
        """Test print_graph() with minimal style."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], pool_name='mypool')
        
        party.print_graph(style='minimal')
        captured = capsys.readouterr()
        
        # Should contain pool name followed by (pool)
        assert 'mypool (pool)' in captured.out
        # Should contain op name followed by [op]
        assert '[op]' in captured.out
        # Should not contain n=
        assert 'n=' not in captured.out
    
    def test_print_graph_repr(self, capsys):
        """Test print_graph() with repr style."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], pool_name='mypool')
        
        party.print_graph(style='repr')
        captured = capsys.readouterr()
        
        # Should contain full repr format
        assert 'Pool(' in captured.out
        assert "name='mypool'" in captured.out
        assert 'num_states=' in captured.out
    
    def test_print_graph_chain(self, capsys):
        """Test print_graph() with a chain of pools."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'], pool_name='seq')
            mutants = pp.mutation_scan(seq, k=1, mode='sequential', pool_name='mutants')
        
        party.print_graph()
        captured = capsys.readouterr()
        
        # Should contain both pool names
        assert 'mutants (pool,' in captured.out
        assert 'seq (pool,' in captured.out
        # Should show tree structure with connectors
        assert '└──' in captured.out
    
    def test_print_graph_multi_parent(self, capsys):
        """Test print_graph() with multiple parent pools (join)."""
        with pp.Party() as party:
            left = pp.from_seqs(['AAA'], pool_name='left')
            right = pp.from_seqs(['TTT'], pool_name='right')
            oligo = pp.join([left, right], pool_name='oligo')
        
        party.print_graph()
        captured = capsys.readouterr()
        
        # Should contain all pool names
        assert 'oligo (pool,' in captured.out
        assert 'left (pool,' in captured.out
        assert 'right (pool,' in captured.out
        # Should show branching structure
        assert '├──' in captured.out or '└──' in captured.out
    
    def test_print_graph_multiple_roots(self, capsys):
        """Test print_graph() with multiple root pools."""
        with pp.Party() as party:
            pool1 = pp.from_seqs(['A', 'B'], pool_name='pool1')
            pool2 = pp.from_seqs(['X', 'Y'], pool_name='pool2')
        
        party.print_graph()
        captured = capsys.readouterr()
        
        # Should contain both pool names
        assert 'pool1 (pool,' in captured.out
        assert 'pool2 (pool,' in captured.out
    
    def test_print_graph_shows_mode(self, capsys):
        """Test print_graph() shows operation mode."""
        with pp.Party() as party:
            seq_pool = pp.from_seqs(['ACGT'], pool_name='seq')
            mutants = pp.mutation_scan(seq_pool, k=1, mode='sequential', pool_name='mutants')
        
        party.print_graph()
        captured = capsys.readouterr()
        
        # Should show sequential mode
        assert 'mode=sequential' in captured.out
    
    def test_print_graph_shows_factory_name(self, capsys):
        """Test print_graph() shows operation factory_name."""
        with pp.Party() as party:
            seq_pool = pp.from_seqs(['ACGT'], pool_name='seq')
            mutants = pp.mutation_scan(seq_pool, k=1, mode='sequential', pool_name='mutants')
        
        party.print_graph()
        captured = capsys.readouterr()
        
        # Should show factory names in clean mode
        assert 'op=from_seqs' in captured.out
        assert 'op=mutation_scan' in captured.out
    
    def test_print_graph_no_pools(self, capsys):
        """Test print_graph() with no pools registered."""
        with pp.Party() as party:
            pass
        
        party.print_graph()
        captured = capsys.readouterr()
        
        assert '(no pools registered)' in captured.out
    
    def test_pool_print_tree(self, capsys):
        """Test Pool.print_tree() method directly."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], pool_name='mypool')
            pool.print_tree()  # clean is default
        
        captured = capsys.readouterr()
        
        # Should start with pool name
        assert captured.out.startswith('mypool (pool,')
        # Should show n=3
        assert 'n=3' in captured.out
    
    def test_pool_print_tree_repr(self, capsys):
        """Test Pool.print_tree() with repr style."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], pool_name='mypool')
            pool.print_tree(style='repr')
        
        captured = capsys.readouterr()
        
        # Should start with Pool repr
        assert captured.out.startswith('Pool(')
        assert 'num_states=3' in captured.out
    
    def test_operation_print_tree(self, capsys):
        """Test Operation.print_tree() method directly."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], pool_name='mypool')
            pool.operation.print_tree()  # clean is default
        
        captured = capsys.readouterr()
        
        # Should contain [op= for operation info
        assert '[op=' in captured.out
    
    def test_operation_print_tree_repr(self, capsys):
        """Test Operation.print_tree() with repr style."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], pool_name='mypool')
            pool.operation.print_tree(style='repr')
        
        captured = capsys.readouterr()
        
        # Should start with operation class name
        assert captured.out.startswith('FromSeqsOp(')
    
    def test_print_graph_with_repeat(self, capsys):
        """Test print_graph() with repeat operation."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B'], pool_name='base')
            repeated = (pool * 2).named('repeated')
        
        party.print_graph()
        captured = capsys.readouterr()
        
        # Should contain both pool names
        assert 'repeated (pool,' in captured.out
        assert 'base (pool,' in captured.out
        # Repeated should have n=4
        assert 'n=4' in captured.out
    
    def test_print_graph_with_stack(self, capsys):
        """Test print_graph() with stack operation."""
        with pp.Party() as party:
            a = pp.from_seqs(['A'], pool_name='a')
            b = pp.from_seqs(['B'], pool_name='b')
            stacked = (a + b).named('stacked')
        
        party.print_graph()
        captured = capsys.readouterr()
        
        # Should contain all pool names
        assert 'stacked (pool,' in captured.out
        assert 'a (pool,' in captured.out
        assert 'b (pool,' in captured.out
