"""Tests for the Join operation.

Note: The + operator on Pools now does stacking (union of states), not joining.
Use join() for sequence joining.
"""

import pytest
import poolparty as pp
from poolparty.fixed_ops.join import join
from poolparty.fixed_ops.fixed import FixedOp


class TestJoinFactory:
    """Test join factory function."""
    
    def test_returns_pool(self):
        """Test that join returns a Pool."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            combined = join([a, b])
            assert combined is not None
            assert hasattr(combined, 'operation')
    
    def test_creates_fixed_op(self):
        """Test that join creates a FixedOp."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            combined = join([a, b])
            assert isinstance(combined.operation, FixedOp)


class TestJoinPools:
    """Test joining multiple pools."""
    
    def test_two_pools(self):
        """Test joining two pools."""
        with pp.Party() as party:
            left = pp.from_seqs(['AAA'])
            right = pp.from_seqs(['TTT'])
            combined = join([left, right]).named('seq')
        
        df = combined.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'AAATTT'
    
    def test_three_pools(self):
        """Test joining three pools."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            c = pp.from_seqs(['GGG'])
            combined = join([a, b, c]).named('seq')
        
        df = combined.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'AAATTTGGG'
    
    def test_many_pools(self):
        """Test joining many pools."""
        with pp.Party() as party:
            pools = [pp.from_seqs([c]) for c in 'ABCDE']
            combined = join(pools).named('seq')
        
        df = combined.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'ABCDE'


class TestJoinWithStrings:
    """Test joining pools with string literals."""
    
    def test_pool_and_string(self):
        """Test joining pool with string."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'])
            combined = join([pool, '...']).named('seq')
        
        df = combined.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'AAA...'
    
    def test_string_and_pool(self):
        """Test joining string with pool."""
        with pp.Party() as party:
            pool = pp.from_seqs(['TTT'])
            combined = join(['...', pool]).named('seq')
        
        df = combined.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == '...TTT'
    
    def test_pool_string_pool(self):
        """Test pool-string-pool pattern."""
        with pp.Party() as party:
            left = pp.from_seqs(['AAA'])
            right = pp.from_seqs(['TTT'])
            combined = join([left, '...', right]).named('seq')
        
        df = combined.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'AAA...TTT'
    
    def test_multiple_strings(self):
        """Test multiple string literals."""
        with pp.Party() as party:
            pool = pp.from_seqs(['X'])
            combined = join(['[', pool, ']']).named('seq')
        
        df = combined.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == '[X]'
    
    def test_only_strings(self):
        """Test joining only strings (converts to pools)."""
        with pp.Party() as party:
            combined = join(['A', 'B', 'C']).named('seq')
        
        df = combined.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'ABC'


class TestJoinVsStack:
    """Test that join and stack are different."""
    
    def test_join_joins_sequences(self):
        """Test that join joins sequences."""
        with pp.Party() as party:
            left = pp.from_seqs(['AAA'])
            right = pp.from_seqs(['TTT'])
            
            combined = join([left, right]).named('seq')
        
        df = combined.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'AAATTT'  # Joined
    
    def test_stack_unions_states(self):
        """Test that + (stack) unions states."""
        with pp.Party() as party:
            left = pp.from_seqs(['AAA'], mode='sequential')
            right = pp.from_seqs(['TTT'], mode='sequential')
            
            stacked = (left + right).named('seq')
        
        df = stacked.generate_library(num_cycles=1)
        assert list(df['seq']) == ['AAA', 'TTT']  # Union, not joined


class TestJoinFixedMode:
    """Test that join is always fixed mode."""
    
    def test_mode_is_fixed(self):
        """Test that join creates fixed mode operation."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            combined = join([a, b])
            assert combined.operation.mode == 'fixed'
    
    def test_num_states_is_one(self):
        """Test that join has num_states=1."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            combined = join([a, b])
            assert combined.operation.num_states == 1
    
    def test_variability_from_parents(self):
        """Test that variability comes from parent pools."""
        with pp.Party() as party:
            # Each parent has 2 states = 4 combinations
            left = pp.from_seqs(['A', 'B'], mode='sequential')
            right = pp.from_seqs(['X', 'Y'], mode='sequential')
            combined = join([left, right]).named('seq')
        
        df = combined.generate_library(num_seqs=4)
        # Should get all 4 combinations
        expected = {'AX', 'AY', 'BX', 'BY'}
        assert set(df['seq']) == expected or len(set(df['seq'])) == 4


class TestJoinDesignCards:
    """Test join design cards."""
    
    def test_no_design_card_keys(self):
        """Test that join has no design card keys."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            combined = join([a, b])
            assert len(combined.operation.design_card_keys) == 0
    
    def test_parent_design_cards_preserved(self):
        """Test that parent design cards are still included."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'], seq_names=['seq_a'])
            b = pp.from_seqs(['TTT'], seq_names=['seq_b'])
            combined = join([a, b]).named('seq')
        
        df = combined.generate_library(num_seqs=1, report_design_cards=True)
        # Parent design cards should be present
        assert 'from_seqs.seq_name' in df.columns or len([c for c in df.columns if 'seq_name' in c]) > 0


class TestJoinCompute:
    """Test join compute methods directly."""
    
    def test_compute_joins_sequences(self):
        """Test compute method joins parent sequences."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            combined = join([a, b])
        
        card = combined.operation.compute_design_card(['AAA', 'TTT'])
        result = combined.operation.compute_seq_from_card(['AAA', 'TTT'], card)
        assert result['seq_0'] == 'AAATTT'
    
    def test_compute_empty_string(self):
        """Test compute with empty string parent."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs([''])
            combined = join([a, b])
        
        card = combined.operation.compute_design_card(['AAA', ''])
        result = combined.operation.compute_seq_from_card(['AAA', ''], card)
        assert result['seq_0'] == 'AAA'
    
    def test_compute_many_parents(self):
        """Test compute with many parent sequences."""
        with pp.Party() as party:
            pools = [pp.from_seqs([c]) for c in 'ABCDE']
            combined = join(pools)
        
        card = combined.operation.compute_design_card(['A', 'B', 'C', 'D', 'E'])
        result = combined.operation.compute_seq_from_card(['A', 'B', 'C', 'D', 'E'], card)
        assert result['seq_0'] == 'ABCDE'


class TestJoinChaining:
    """Test chaining join operations."""
    
    def test_nested_join(self):
        """Test nested join operations."""
        with pp.Party() as party:
            a = pp.from_seqs(['A'])
            b = pp.from_seqs(['B'])
            c = pp.from_seqs(['C'])
            
            ab = join([a, b])
            abc = join([ab, c]).named('seq')
        
        df = abc.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'ABC'


class TestJoinWithOtherOperations:
    """Test join with other operation types."""
    
    def test_with_mutagenize(self):
        """Test joining with mutation scan output."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'], mode='sequential')
            mutants = pp.mutagenize(seq, num_mutations=1, mode='sequential')
            barcode = pp.from_seqs(['NNNN'], mode='sequential')
            combined = join([mutants, '.', barcode]).named('seq')
        
        df = combined.generate_library(num_seqs=3)
        for s in df['seq']:
            assert s.endswith('.NNNN')
            assert len(s) == 9  # 4 + 1 + 4
    
    def test_with_get_kmers(self):
        """Test joining with k-mers."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])
            barcode = pp.get_kmers(length=4, mode='random')
            combined = join([seq, '...', barcode]).named('seq')
        
        df = combined.generate_library(num_seqs=10, seed=42)
        for s in df['seq']:
            assert s.startswith('ACGT...')
            assert len(s) == 11  # 4 + 3 + 4
    
    def test_with_breakpoint_scan(self):
        """Test joining synchronized breakpoint scan outputs.
        
        Breakpoint scan creates synchronized pools that share the same counter.
        When joined, the shared counter is only included once in the product, so they
        iterate together in lockstep.
        """
        with pp.Party() as party:
            left, right = pp.breakpoint_scan('ACGT', num_breakpoints=1)
            combined = join([left, '---', right]).named('seq')
        
        df = combined.generate_library(num_seqs=3)
        # Verify segments are joined with separator
        for s in df['seq']:
            assert '---' in s


class TestJoinCustomName:
    """Test join name parameter."""
    
    def test_default_name(self):
        """Test default operation name is 'join'."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            combined = join([a, b])
            assert combined.operation.name.endswith(':join')


class TestJoinSpacerStr:
    """Test spacer_str parameter for join."""
    
    def test_spacer_str_basic(self):
        """Test basic spacer_str usage."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            combined = join([a, b], spacer_str='-').named('seq')
        
        df = combined.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'AAA-TTT'
    
    def test_spacer_str_multiple_chars(self):
        """Test spacer_str with multiple characters."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            combined = join([a, b], spacer_str='---').named('seq')
        
        df = combined.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'AAA---TTT'
    
    def test_spacer_str_with_three_pools(self):
        """Test spacer_str with three pools."""
        with pp.Party() as party:
            a = pp.from_seqs(['A'])
            b = pp.from_seqs(['B'])
            c = pp.from_seqs(['C'])
            combined = join([a, b, c], spacer_str='.').named('seq')
        
        df = combined.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'A.B.C'
    
    def test_spacer_str_seq_length_calculation(self):
        """Test that seq_length includes spacer characters."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])  # length 3
            b = pp.from_seqs(['TTT'])  # length 3
            combined = join([a, b], spacer_str='--')  # spacer length 2
            # Total should be 3 + 3 + 2 = 8
            assert combined.seq_length == 8
    
    def test_spacer_str_seq_length_three_pools(self):
        """Test seq_length with three pools and spacer."""
        with pp.Party() as party:
            a = pp.from_seqs(['AA'])  # length 2
            b = pp.from_seqs(['BB'])  # length 2
            c = pp.from_seqs(['CC'])  # length 2
            combined = join([a, b, c], spacer_str='.')  # spacer length 1, 2 spacers
            # Total should be 2 + 2 + 2 + 1 + 1 = 8
            assert combined.seq_length == 8
    
    def test_spacer_str_with_strings(self):
        """Test spacer_str when joining strings."""
        with pp.Party() as party:
            combined = join(['A', 'B', 'C'], spacer_str='-').named('seq')
        
        df = combined.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'A-B-C'
    
    def test_spacer_str_compute_method(self):
        """Test that compute_seq_from_card uses spacer_str."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            combined = join([a, b], spacer_str='.')
        
        card = combined.operation.compute_design_card(['AAA', 'TTT'])
        result = combined.operation.compute_seq_from_card(['AAA', 'TTT'], card)
        assert result['seq_0'] == 'AAA.TTT'
    
    def test_spacer_str_with_single_pool(self):
        """Test spacer_str with single item (no spacer needed)."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            combined = join([a], spacer_str='-').named('seq')
        
        df = combined.generate_library(num_seqs=1)
        assert df['seq'].iloc[0] == 'AAA'  # No spacer for single item


class TestStackBranchIndex:
    """Test that stack operation counter tracks the active branch index."""
    
    def test_stack_state_matches_branch_index(self):
        """Test that op[X]:stack.state matches the active branch (0, 1, 2...)."""
        with pp.Party() as party:
            a = pp.from_seqs(['A1', 'A2', 'A3'], mode='sequential').named('A')
            b = pp.from_seqs(['B1', 'B2'], mode='sequential').named('B')
            c = pp.from_seqs(['C1', 'C2', 'C3', 'C4'], mode='sequential').named('C')
            stacked = pp.stack([a, b, c]).named('stacked')
        
        df = stacked.generate_library(num_cycles=1, report_design_cards=True)
        
        # Find the stack state column
        stack_state_col = [c for c in df.columns if 'stack.state' in c][0]
        
        # A has 3 states (indices 0-2), B has 2 states (indices 3-4), C has 4 states (indices 5-8)
        # Stack state should be: 0,0,0, 1,1, 2,2,2,2
        expected_branch_indices = [0, 0, 0, 1, 1, 2, 2, 2, 2]
        actual_branch_indices = list(df[stack_state_col])
        
        assert actual_branch_indices == expected_branch_indices
    
    def test_stack_state_two_branches(self):
        """Test stack state with two branches."""
        with pp.Party() as party:
            a = pp.from_seqs(['A1', 'A2'], mode='sequential').named('A')
            b = pp.from_seqs(['B1', 'B2', 'B3'], mode='sequential').named('B')
            stacked = pp.stack([a, b]).named('stacked')
        
        df = stacked.generate_library(num_cycles=1, report_design_cards=True)
        
        stack_state_col = [c for c in df.columns if 'stack.state' in c][0]
        
        # A has 2 states (branch 0), B has 3 states (branch 1)
        expected = [0, 0, 1, 1, 1]
        actual = list(df[stack_state_col])
        
        assert actual == expected
    
    def test_stack_state_matches_active_parent(self):
        """Test that stack state equals active_parent design card key."""
        with pp.Party() as party:
            a = pp.from_seqs(['A1', 'A2'], mode='sequential').named('A')
            b = pp.from_seqs(['B1'], mode='sequential').named('B')
            c = pp.from_seqs(['C1', 'C2', 'C3'], mode='sequential').named('C')
            stacked = pp.stack([a, b, c]).named('stacked')
        
        df = stacked.generate_library(num_cycles=1, report_design_cards=True)
        
        stack_state_col = [c for c in df.columns if 'stack.state' in c][0]
        active_parent_col = [c for c in df.columns if 'stack.key.active_parent' in c][0]
        
        # Both should have the same values
        assert list(df[stack_state_col]) == list(df[active_parent_col])
    
    def test_stack_num_states_equals_num_branches(self):
        """Test that StackOp.num_states equals number of parent pools."""
        with pp.Party() as party:
            a = pp.from_seqs(['A1', 'A2'])
            b = pp.from_seqs(['B1'])
            c = pp.from_seqs(['C1', 'C2', 'C3'])
            stacked = pp.stack([a, b, c])
            
            # Operation counter should have num_states = 3 (number of branches)
            assert stacked.operation.num_states == 3
            assert stacked.operation.counter.num_states == 3
