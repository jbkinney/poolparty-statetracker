"""Insert multiple XML region tags into a sequence."""
from poolparty.types import Union, Optional, Sequence, Literal
from numbers import Integral, Real
import numpy as np

from .parsing import build_region_tags, get_nontag_positions, nontag_pos_to_literal_pos
from ..operation import Operation

# Type aliases
PositionsType = Union[list[int], tuple[int, ...], slice, None]


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


def region_multiscan(
    pool,
    regions,
    num_insertions: int,
    positions: PositionsType = None,
    strand: str = '+',
    region_length: int = 0,
    insertion_mode: Literal['ordered', 'unordered'] = 'ordered',
    prefix: Optional[str] = None,
    mode: str = 'random',
    num_states: Optional[int] = None,
    iter_order: Optional[Real] = None,
):
    """
    Insert multiple XML-style region tags into a sequence.

    Parameters
    ----------
    pool : Pool or str
        Input Pool or sequence string to insert tags into.
    regions : Sequence[str] or str
        Region name(s) to insert. If a single string, used for all insertions.
    num_insertions : Integral
        Number of region tags to insert.
    positions : PositionsType, default=None
        Valid insertion positions (0-based). If None, all positions are valid.
    strand : StrandType, default='+'
        Strand for regions: '+', '-', or 'both'.
    region_length : Integral, default=0
        Length of sequence to encompass per region. 0 for zero-length regions.
    insertion_mode : str, default='ordered'
        How to assign regions to positions:
        - 'ordered': regions[i] goes to the i-th selected position
        - 'unordered': randomly assign regions to positions
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    mode : ModeType, default='random'
        Position selection mode: 'random'.
    num_states : Optional[Integral], default=None
        Number of states for random mode. If None, defaults to 1 (pure random sampling).
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool
        A Pool yielding sequences with multiple region tags inserted.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..pool import Pool
    from ..party import get_active_party

    pool = from_seq(pool) if isinstance(pool, str) else pool
    
    # Register all regions with the Party
    party = get_active_party()
    region_names = [regions] if isinstance(regions, str) else list(regions)
    registered_regions = []
    for region_name in region_names:
        registered_region = party.register_region(region_name, region_length)
        registered_regions.append(registered_region)
    
    op = RegionMultiScanOp(
        parent_pool=pool,
        regions=regions,
        num_insertions=int(num_insertions),
        positions=positions,
        strand=strand,
        region_length=int(region_length),
        insertion_mode=insertion_mode,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        name=None,
        iter_order=iter_order,
    )
    result_pool = Pool(operation=op)
    
    # Add all registered regions to the pool
    for registered_region in registered_regions:
        result_pool.add_region(registered_region)
    
    return result_pool


class RegionMultiScanOp(Operation):
    """Insert multiple XML region tags at selected positions."""

    factory_name = "region_multiscan"
    design_card_keys = ['indices', 'strands', 'region_tags']

    def __init__(
        self,
        parent_pool,
        regions,
        num_insertions: int,
        positions: PositionsType = None,
        strand: str = '+',
        region_length: int = 0,
        insertion_mode: str = 'ordered',
        prefix: Optional[str] = None,
        mode: str = 'random',
        num_states: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        if num_insertions < 1:
            raise ValueError(f"num_insertions must be >= 1, got {num_insertions}")
        if mode != 'random':
            raise ValueError("region_multiscan supports only mode='random'")
        if region_length < 0:
            raise ValueError(f"region_length must be >= 0, got {region_length}")

        self._positions = positions
        self._mode = mode
        self._strand = strand
        self._region_length = region_length
        self._seq_length = parent_pool.seq_length
        self.num_insertions = num_insertions
        self.insertion_mode = insertion_mode
        self._region_names = self._coerce_regions(regions)
        self._validate_region_counts()

        # num_states stays None for pure random mode
        super().__init__(
            parent_pools=[parent_pool],
            num_values=num_states,
            mode=mode,
            seq_length=None,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
        )

    def _coerce_regions(
        self, regions: Union[Sequence[str], str]
    ) -> list[str]:
        """Normalize regions input to a list of region names."""
        if isinstance(regions, str):
            regions = [regions]
        if not regions:
            raise ValueError("regions must not be empty")
        return list(regions)

    def _validate_region_counts(self) -> None:
        """Validate region counts against insertion_mode."""
        if self.insertion_mode not in ('ordered', 'unordered'):
            raise ValueError(
                "insertion_mode must be one of 'ordered', 'unordered'"
            )
        if self.insertion_mode == 'ordered' and len(self._region_names) != self.num_insertions:
            raise ValueError(
                "insertion_mode='ordered' requires len(regions) == num_insertions"
            )
        if self.insertion_mode == 'unordered' and len(self._region_names) < self.num_insertions:
            raise ValueError(
                "insertion_mode='unordered' requires len(regions) >= num_insertions"
            )

    def _get_valid_region_indices(self, seq: str) -> list[int]:
        """Return valid nontag indices (0 to n-1) for region tag insertion.
        
        Returns logical indices into the non-tag character positions,
        not literal string positions. This allows proper handling of
        multiple tag insertions without position corruption.
        """
        nontag_positions = get_nontag_positions(seq)
        num_nontag = len(nontag_positions)
        
        if self._region_length > 0:
            # For region tags, ensure room for content
            # Valid indices are 0 to (num_nontag - region_length)
            max_valid_idx = num_nontag - self._region_length
            if max_valid_idx < 0:
                return []
            all_valid = list(range(max_valid_idx + 1))
        else:
            # For zero-length regions, can insert at any position including end
            all_valid = list(range(num_nontag + 1))

        if self._positions is not None:
            indices = _validate_positions(
                self._positions,
                max_position=len(all_valid) - 1,
                min_position=0,
            )
            return [all_valid[i] for i in indices]

        return all_valid

    def _select_indices(
        self, valid_indices: list[int], rng: np.random.Generator
    ) -> list[int]:
        """Select nontag indices for region tag insertion.
        
        For region tags (region_length > 0), ensures selected indices
        are at least region_length apart to prevent overlapping regions.
        """
        if len(valid_indices) < self.num_insertions:
            raise ValueError(
                f"Not enough valid positions ({len(valid_indices)}) "
                f"for {self.num_insertions} insertions"
            )
        
        if self._region_length > 0:
            # For region tags, need non-overlapping selection
            # Use greedy algorithm with random shuffling
            shuffled = list(valid_indices)
            rng.shuffle(shuffled)
            
            chosen = []
            for idx in shuffled:
                # Check if this idx overlaps with any already chosen
                overlaps = False
                for c in chosen:
                    if abs(idx - c) < self._region_length:
                        overlaps = True
                        break
                if not overlaps:
                    chosen.append(idx)
                    if len(chosen) == self.num_insertions:
                        break
            
            if len(chosen) < self.num_insertions:
                raise ValueError(
                    f"Cannot select {self.num_insertions} non-overlapping positions "
                    f"with region_length={self._region_length} from "
                    f"{len(valid_indices)} valid positions"
                )
            return sorted(chosen)
        else:
            # Zero-length regions don't overlap
            chosen = rng.choice(
                valid_indices,
                size=self.num_insertions,
                replace=False,
            )
            return sorted(int(x) for x in chosen)

    def _select_strands(self, rng: np.random.Generator) -> list[str]:
        """Select strands for each insertion."""
        if self._strand == 'both':
            return ['+' if rng.random() < 0.5 else '-' for _ in range(self.num_insertions)]
        else:
            return [self._strand] * self.num_insertions

    def _select_region_tags(
        self, seq: str, indices: list[int], strands: list[str], rng: np.random.Generator
    ) -> list[str]:
        """Build region tags for each nontag index.
        
        Args:
            seq: The original sequence string.
            indices: Nontag indices (logical positions, not literal).
            strands: Strand for each region.
            rng: Random number generator.
        
        Returns:
            List of region tag strings.
        """
        if self.insertion_mode == 'ordered':
            names = self._region_names
        else:
            # unordered: randomly select from available regions
            idxs = rng.choice(len(self._region_names), size=self.num_insertions, replace=False)
            names = [self._region_names[int(i)] for i in idxs]
        
        tags = []
        for idx, strand, name in zip(indices, strands, names):
            if self._region_length > 0:
                # Convert nontag index to literal positions for content extraction
                literal_start = nontag_pos_to_literal_pos(seq, idx)
                literal_end = nontag_pos_to_literal_pos(seq, idx + self._region_length)
                content = seq[literal_start:literal_end]
                # Strip any existing tags from content (keep only actual characters)
                from .parsing import strip_all_tags
                content = strip_all_tags(content)
            else:
                content = ''
            tags.append(build_region_tags(name, content, strand))
        return tags

    def compute(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
        parent_styles: list | None = None,
    ) -> dict:
        """Return design card and sequence with region tags inserted together."""
        seq = parent_seqs[0]
        if rng is None:
            raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG")

        valid_indices = self._get_valid_region_indices(seq)
        indices = self._select_indices(valid_indices, rng)
        strands = self._select_strands(rng)
        region_tags = self._select_region_tags(seq, indices, strands, rng)
        
        # Build result sequence with tags inserted at nontag indices
        # Uses single-pass construction to avoid position corruption issues
        # when inserting multiple overlapping tags
        indices_list = list(indices)
        region_tags_list = list(region_tags)
        
        # Sort by index ascending for left-to-right processing
        inserts = sorted(zip(indices_list, region_tags_list), key=lambda x: x[0])
        
        # Build result string from left to right
        result_parts = []
        prev_end_idx = 0  # Next nontag index to copy from
        
        for nt_idx, tag in inserts:
            # Copy characters from prev_end_idx to nt_idx (exclusive)
            if prev_end_idx < nt_idx:
                start_literal = nontag_pos_to_literal_pos(seq, prev_end_idx)
                end_literal = nontag_pos_to_literal_pos(seq, nt_idx)
                result_parts.append(seq[start_literal:end_literal])
            
            # Add the region tag (which contains content for region tags)
            result_parts.append(tag)
            
            # Update prev_end_idx based on region type
            if self._region_length > 0:
                # Region tags: skip the characters that are now inside the tag
                prev_end_idx = nt_idx + self._region_length
            else:
                # Zero-length region: don't skip any characters
                prev_end_idx = nt_idx
        
        # Add remaining characters after the last tag
        nontag_positions = get_nontag_positions(seq)
        if prev_end_idx < len(nontag_positions):
            start_literal = nontag_pos_to_literal_pos(seq, prev_end_idx)
            result_parts.append(seq[start_literal:])
        elif prev_end_idx == len(nontag_positions):
            # Edge case: last tag ends exactly at end of sequence
            # There might be trailing tags to preserve
            last_nontag_literal = nontag_positions[-1] if nontag_positions else 0
            if last_nontag_literal + 1 < len(seq):
                result_parts.append(seq[last_nontag_literal + 1:])
        
        result_seq = ''.join(result_parts)
        
        # Region multiscan modifies sequence structure, so styles not meaningful
        return {
            'indices': indices_list,  # nontag indices, not literal positions
            'strands': strands,
            'region_tags': region_tags_list,
            'seq': result_seq,
            'style': [],
        }

    def _get_copy_params(self) -> dict:
        return {
            'parent_pool': self.parent_pools[0],
            'regions': self._region_names,
            'num_insertions': self.num_insertions,
            'positions': self._positions,
            'strand': self._strand,
            'region_length': self._region_length,
            'insertion_mode': self.insertion_mode,
            'prefix': self.name_prefix,
            'mode': self.mode,
            'num_states': self.num_values if self.mode == 'random' and self.num_values is not None and self.num_values > 1 else None,
            'name': None,
            'iter_order': self.iter_order,
        }
