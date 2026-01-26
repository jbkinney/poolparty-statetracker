"""RegionScan operation - insert XML region tags at scanning positions."""
from poolparty.types import Union, Optional, Literal, RegionType
from numbers import Integral, Real
import numpy as np

from .parsing import build_region_tags, TAG_PATTERN, get_nontag_positions
from ..operation import Operation

# Type aliases
PositionsType = Union[list[int], tuple[int, ...], slice, None]
ModeType = Literal['random', 'sequential']
StrandType = Literal['+', '-', 'both']


def region_scan(
    pool,
    region: str = 'region',
    positions: PositionsType = None,
    region_constraint: RegionType = None,
    remove_tags: Optional[bool] = None,
    strand: str = '+',
    region_length: int = 0,
    prefix: Optional[str] = None,
    mode: str = 'random',
    num_states: Optional[int] = None,
    iter_order: Optional[Real] = None,
    _factory_name: Optional[str] = None,
):
    """
    Insert XML-style region tags at scanning positions in a sequence.

    Parameters
    ----------
    pool : Pool or str
        Input Pool or sequence string to insert tags into.
    region : str, default='region'
        Name for the region to insert.
    positions : PositionsType, default=None
        Valid insertion positions (0-based). If None, all positions are valid.
    region_constraint : RegionType, default=None
        Region to constrain the scan to. Can be region name (str) or [start, stop].
    remove_tags : Optional[bool], default=None
        If True and region_constraint is a region name, remove tags from output.
    strand : StrandType, default='+'
        Strand for the region: '+', '-', or 'both'.
    region_length : Integral, default=0
        Length of sequence to encompass. 0 creates zero-length regions (<name/>),
        >0 creates region tags (<name>BASES</name>).
    mode : ModeType, default='random'
        Position selection mode: 'random' or 'sequential'.
    _factory_name : Optional[str], default=None
        Sets default name of the resulting operation

    Returns
    -------
    Pool
        A Pool yielding sequences with the region tags inserted at selected positions.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..pool import Pool
    from ..party import get_active_party
    
    # Convert string input to pool if needed
    pool = from_seq(pool) if isinstance(pool, str) else pool
    
    # Validate region_length
    if region_length < 0:
        raise ValueError(f"region_length must be >= 0, got {region_length}")
    
    # Register the region with the Party
    party = get_active_party()
    registered_region = party.register_region(region, region_length)
    
    op = RegionScanOp(
        parent_pool=pool,
        region_name=region,
        positions=positions,
        region=region_constraint,
        remove_tags=remove_tags,
        strand=strand,
        region_length=int(region_length),
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        name=None,
        iter_order=iter_order,
        _factory_name=_factory_name,
    )
    result_pool = Pool(operation=op)
    
    # Add the region to the pool's region set
    result_pool.add_region(registered_region)
    
    return result_pool


def _validate_positions(positions: PositionsType, max_position: int, min_position: int = 0) -> list[int]:
    """Validate and normalize position specification."""
    if positions is None:
        return list(range(min_position, max_position + 1))
    
    if isinstance(positions, slice):
        start = positions.start if positions.start is not None else min_position
        stop = positions.stop if positions.stop is not None else max_position + 1
        step = positions.step if positions.step is not None else 1
        return list(range(start, stop, step))
    
    positions_list = list(positions)
    for p in positions_list:
        if p < min_position or p > max_position:
            raise ValueError(
                f"Position {p} out of range [{min_position}, {max_position}]"
            )
    return positions_list


class RegionScanOp(Operation):
    """Insert XML region tags at scanning positions."""
    factory_name = "region_scan"
    design_card_keys = ['position_index', 'start', 'stop', 'length', 'region_name', 'region_content', 'strand', 'region_seq']
    
    def __init__(
        self,
        parent_pool,
        region_name: str,
        positions: PositionsType = None,
        region: RegionType = None,
        remove_tags: Optional[bool] = None,
        spacer_str: str = '',
        strand: str = '+',
        region_length: int = 0,
        prefix: Optional[str] = None,
        mode: str = 'random',
        num_states: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        _factory_name: Optional[str] = None,
    ) -> None:
        """Initialize RegionScanOp."""
        from ..party import get_active_party
        
        
        self.region_name = region_name
        self._positions = positions
        self._mode = mode
        self._strand = strand
        self._region_length = region_length
        self._region = region  # Store early for logging
        self._valid_positions = None
        self._sequential_cache = None
        
        # Determine effective seq_length for cache building:
        # If region is a region name, use the region's registered length
        # Otherwise, use the parent pool's seq_length
        if isinstance(region, str):
            party = get_active_party()
            try:
                constraint_region = party.get_region_by_name(region)
                self._seq_length = constraint_region.seq_length
            except (ValueError, KeyError):
                # Region not yet registered, fall back to parent seq_length
                self._seq_length = parent_pool.seq_length
        else:
            self._seq_length = parent_pool.seq_length
        
        # Set factory name if provided
        if _factory_name is not None:
            self.factory_name = _factory_name
 
        # Calculate number of states
        if mode == 'sequential':
            if self._seq_length is not None:
                num_states = self._build_caches()
            else:
                num_states = 1
        elif mode == 'random':
            # num_states stays None for pure random mode
            pass
        else:
            num_states = 1
        
        # If strand='both', double the number of states
        if strand == 'both' and mode == 'sequential':
            num_states *= 2
        
        # Initialize as Operation
        super().__init__(
            parent_pools=[parent_pool],
            num_values=num_states,
            mode=mode,
            seq_length=None,  # Variable due to region tags
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            region=region,
            remove_tags=remove_tags,
        )
    
    def _build_caches(self) -> int:
        """Build caches for sequential enumeration based on seq_length."""
        if self._seq_length is None:
            if self._positions is not None:
                positions_list = _validate_positions(
                    self._positions,
                    max_position=1000000,
                    min_position=0,
                )
                return max(1, len(positions_list))
            return 1
        
        # For region tags, we need room for region_length bases
        max_start = self._seq_length - self._region_length
        if max_start < 0:
            max_start = 0
        
        num_all_positions = max_start + 1
        if self._positions is not None:
            indices = _validate_positions(
                self._positions,
                max_position=num_all_positions - 1,
                min_position=0,
            )
            num_states = len(indices)
        else:
            num_states = num_all_positions
        
        if num_states == 0:
            raise ValueError("No valid positions for region tag insertion")
        return num_states
    
    def _get_valid_region_positions(self, seq: str) -> tuple[list[int], list[int]]:
        """Get valid region tag insertion positions, excluding tag interiors.
        
        Returns tuple of (valid_nontag_indices, nontag_positions) where:
        - valid_nontag_indices: indices into nontag_positions that are valid start positions
        - nontag_positions: literal positions of all non-tag characters
        """
        # Get positions not inside existing tags
        nontag_positions = get_nontag_positions(seq)
        
        # For region tags, ensure there's room for region_length bases
        if self._region_length > 0:
            # Valid indices are those where we have room for region_length consecutive non-tag chars
            max_valid_idx = len(nontag_positions) - self._region_length
            if max_valid_idx < 0:
                all_valid_indices = []
            else:
                all_valid_indices = list(range(max_valid_idx + 1))
        else:
            # For zero-length regions, all positions are valid plus end (len of seq)
            all_valid_indices = list(range(len(nontag_positions) + 1))
        
        # Apply user position filter
        if self._positions is not None:
            indices = _validate_positions(
                self._positions,
                max_position=len(all_valid_indices) - 1,
                min_position=0,
            )
            filtered_indices = [all_valid_indices[i] for i in indices]
            return filtered_indices, nontag_positions
        
        return all_valid_indices, nontag_positions
    
    def compute(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
        parent_styles: list | None = None,
    ) -> dict:
        """Return design card and sequence with region tags inserted together."""
        from ..types import StyleList
        
        seq = parent_seqs[0]
        
        valid_indices, nontag_positions = self._get_valid_region_positions(seq)
        if len(valid_indices) == 0:
            raise ValueError("No valid positions for region tag insertion")
        
        # Determine strand for this state
        if self._strand == 'both':
            # For sequential mode, alternate strands
            if self.mode == 'sequential':
                state = self.state.value
                state = 0 if state is None else state
                num_pos = len(valid_indices)
                position_index = (state // 2) % num_pos
                strand = '+' if (state % 2) == 0 else '-'
            else:
                # Random mode: randomly choose strand
                if rng is None:
                    raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG")
                position_index = int(rng.integers(0, len(valid_indices)))
                strand = '+' if rng.random() < 0.5 else '-'
        else:
            strand = self._strand
            if self.mode in ('random', 'hybrid'):
                if rng is None:
                    raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG")
                position_index = int(rng.integers(0, len(valid_indices)))
            else:
                state = self.state.value
                state = 0 if state is None else state
                position_index = state % len(valid_indices)
        
        # Build region tags - extract content using non-tag indices
        nontag_idx = valid_indices[position_index]
        explicit_strand = (self._strand == 'both')
        if self._region_length > 0:
            # Extract content from non-tag characters only
            content = ''.join(
                seq[nontag_positions[i]]
                for i in range(nontag_idx, nontag_idx + self._region_length)
            )
            region_tag = build_region_tags(self.region_name, content, strand, explicit_strand=explicit_strand)
            start = nontag_idx
            stop = nontag_idx + self._region_length
            # Get raw sequence from literal start to end (including tags/gaps, excluding new region_tag)
            start_literal = nontag_positions[nontag_idx]
            end_nontag_idx = nontag_idx + self._region_length
            if end_nontag_idx < len(nontag_positions):
                end_literal = nontag_positions[end_nontag_idx]
            else:
                end_literal = nontag_positions[-1] + 1 if nontag_positions else len(seq)
            marked_seq = seq[start_literal:end_literal]
        else:
            region_tag = build_region_tags(self.region_name, '', strand, explicit_strand=explicit_strand)
            start = nontag_idx
            stop = nontag_idx
            marked_seq = ''
        
        # Insert tags at position
        if self._region_length > 0:
            # Region tags: replace content with tags
            # Get literal start and end positions from non-tag indices
            start_literal = nontag_positions[nontag_idx]
            end_nontag_idx = nontag_idx + self._region_length
            # End position is the literal position of the first char AFTER the region
            if end_nontag_idx < len(nontag_positions):
                end_literal = nontag_positions[end_nontag_idx]
            else:
                # One past the last non-tag character (preserves trailing tags)
                end_literal = nontag_positions[-1] + 1 if nontag_positions else len(seq)
            result_seq = seq[:start_literal] + region_tag + seq[end_literal:]
            
            # Adjust parent styles for region tag insertion
            # Opening tag length is from start of region_tag to first '>' + 1
            opening_tag_end = region_tag.index('>') + 1
            opening_tag_len = opening_tag_end
            # Closing tag length is the rest
            closing_tag_len = len(region_tag) - opening_tag_len - len(content)
            total_tag_len = opening_tag_len + closing_tag_len
        else:
            # Zero-length region: insert at position
            if nontag_idx < len(nontag_positions):
                raw_position = nontag_positions[nontag_idx]
            else:
                raw_position = len(seq)  # Insert at end
            result_seq = seq[:raw_position] + region_tag + seq[raw_position:]
        
        # Adjust parent styles to account for tag insertion
        output_styles: StyleList = []
        if parent_styles and len(parent_styles) > 0:
            input_styles = parent_styles[0]
            
            if self._region_length > 0:
                # Region tags: positions shift based on where they fall
                for spec, positions in input_styles:
                    adjusted_positions = []
                    for pos in positions:
                        if pos < start_literal:
                            # Before tags: unchanged
                            adjusted_positions.append(pos)
                        elif pos < end_literal:
                            # Inside region: shift by opening tag length
                            adjusted_positions.append(pos + opening_tag_len)
                        else:
                            # After region: shift by total tag length minus replaced content
                            shift = total_tag_len - (end_literal - start_literal) + len(content)
                            adjusted_positions.append(pos + shift)
                    if adjusted_positions:
                        output_styles.append((spec, np.array(adjusted_positions, dtype=np.int64)))
            else:
                # Zero-length region: positions >= insertion point shift by tag length
                region_tag_len = len(region_tag)
                for spec, positions in input_styles:
                    adjusted_positions = []
                    for pos in positions:
                        if pos < raw_position:
                            adjusted_positions.append(pos)
                        else:
                            adjusted_positions.append(pos + region_tag_len)
                    if adjusted_positions:
                        output_styles.append((spec, np.array(adjusted_positions, dtype=np.int64)))
        
        return {
            'position_index': position_index,
            'start': start,
            'stop': stop,
            'length': self._region_length,
            'region_name': self.region_name,
            'region_content': marked_seq,
            'strand': strand,
            'region_seq': region_tag,
            'seq': result_seq,
            'style': output_styles,
        }
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pool': self.parent_pools[0],
            'region_name': self.region_name,
            'positions': self._positions,
            'region': self._region,
            'remove_tags': self._remove_tags,
            'strand': self._strand,
            'region_length': self._region_length,
            'prefix': self.name_prefix,
            'mode': self.mode,
            'num_states': self.num_values if self.mode == 'random' and self.num_values is not None and self.num_values > 1 else None,
            'name': None,
            'iter_order': self.iter_order,
        }
