"""Insert multiple XML markers into a sequence."""
from typing import Union, Optional, Sequence as TypingSequence
from numbers import Integral, Real
import numpy as np

from .parsing import build_marker_tag, get_positions_without_markers
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


def marker_multiscan(
    pool,
    markers,
    num_insertions: int,
    positions: PositionsType = None,
    strand: str = '+',
    marker_length: int = 0,
    insertion_mode: str = 'ordered',
    mode: str = 'random',
    num_hybrid_states: Optional[int] = None,
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
):
    """
    Insert multiple XML-style markers into a sequence.

    Parameters
    ----------
    pool : Pool or str
        Input Pool or sequence string to insert markers into.
    markers : Sequence[str] or str
        Marker name(s) to insert. If a single string, used for all insertions.
    num_insertions : Integral
        Number of markers to insert.
    positions : PositionsType, default=None
        Valid insertion positions (0-based). If None, all positions are valid.
    strand : StrandType, default='+'
        Strand for markers: '+', '-', or 'both'.
    marker_length : Integral, default=0
        Length of sequence to encompass per marker. 0 for zero-length markers.
    insertion_mode : str, default='ordered'
        How to assign markers to positions:
        - 'ordered': markers[i] goes to the i-th selected position
        - 'unordered': randomly assign markers to positions
    mode : ModeType, default='random'
        Position selection mode: 'random' or 'hybrid'.
    num_hybrid_states : Optional[Integral], default=None
        Number of pool states when using 'hybrid' mode.
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying Operation.
    iter_order : Optional[Real], default=None
        Iteration order priority for the resulting Pool.
    op_iter_order : Optional[Real], default=None
        Iteration order priority for the underlying Operation.

    Returns
    -------
    Pool
        A Pool yielding sequences with multiple markers inserted.
    """
    from ..operations.from_seq import from_seq
    from ..pool import Pool
    from ..party import get_active_party

    pool = from_seq(pool) if isinstance(pool, str) else pool
    
    # Register all markers with the Party
    party = get_active_party()
    marker_names = [markers] if isinstance(markers, str) else list(markers)
    registered_markers = []
    for marker_name in marker_names:
        registered_marker = party.register_marker(marker_name, marker_length)
        registered_markers.append(registered_marker)
    
    op = MarkerMultiScanOp(
        parent_pool=pool,
        markers=markers,
        num_insertions=int(num_insertions),
        positions=positions,
        strand=strand,
        marker_length=int(marker_length),
        insertion_mode=insertion_mode,
        mode=mode,
        num_hybrid_states=num_hybrid_states,
        name=op_name,
        iter_order=op_iter_order,
    )
    result_pool = Pool(operation=op, name=name, iter_order=iter_order)
    
    # Add all registered markers to the pool
    for registered_marker in registered_markers:
        result_pool.add_marker(registered_marker)
    
    return result_pool


class MarkerMultiScanOp(Operation):
    """Insert multiple XML markers at selected positions."""

    factory_name = "marker_multiscan"
    design_card_keys = ['positions', 'strands', 'marker_tags']

    def __init__(
        self,
        parent_pool,
        markers,
        num_insertions: int,
        positions: PositionsType = None,
        strand: str = '+',
        marker_length: int = 0,
        insertion_mode: str = 'ordered',
        mode: str = 'random',
        num_hybrid_states: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> None:
        if num_insertions < 1:
            raise ValueError(f"num_insertions must be >= 1, got {num_insertions}")
        if mode not in ('random', 'hybrid'):
            raise ValueError("marker_multiscan supports only mode='random' or 'hybrid'")
        if mode == 'hybrid' and num_hybrid_states is None:
            raise ValueError("num_hybrid_states is required when mode='hybrid'")
        if marker_length < 0:
            raise ValueError(f"marker_length must be >= 0, got {marker_length}")

        self._positions = positions
        self._mode = mode
        self._strand = strand
        self._marker_length = marker_length
        self._seq_length = parent_pool.seq_length
        self.num_insertions = num_insertions
        self.insertion_mode = insertion_mode
        self._marker_names = self._coerce_markers(markers)
        self._validate_marker_counts()

        num_states = 1 if mode == 'random' else num_hybrid_states
        super().__init__(
            parent_pools=[parent_pool],
            num_states=num_states,
            mode=mode,
            seq_length=None,
            name=name,
            iter_order=iter_order,
        )

    def _coerce_markers(
        self, markers: Union[TypingSequence[str], str]
    ) -> list[str]:
        """Normalize markers input to a list of marker names."""
        if isinstance(markers, str):
            markers = [markers]
        if not markers:
            raise ValueError("markers must not be empty")
        return list(markers)

    def _validate_marker_counts(self) -> None:
        """Validate marker counts against insertion_mode."""
        if self.insertion_mode not in ('ordered', 'unordered'):
            raise ValueError(
                "insertion_mode must be one of 'ordered', 'unordered'"
            )
        if self.insertion_mode == 'ordered' and len(self._marker_names) != self.num_insertions:
            raise ValueError(
                "insertion_mode='ordered' requires len(markers) == num_insertions"
            )
        if self.insertion_mode == 'unordered' and len(self._marker_names) < self.num_insertions:
            raise ValueError(
                "insertion_mode='unordered' requires len(markers) >= num_insertions"
            )

    def _get_valid_marker_positions(self, seq: str) -> list[int]:
        """Valid insertion positions excluding marker interiors."""
        valid_raw_positions = get_positions_without_markers(seq)
        
        if self._marker_length > 0:
            # For region markers, ensure room for content
            all_valid = []
            for i, raw_pos in enumerate(valid_raw_positions):
                if i + self._marker_length <= len(valid_raw_positions):
                    all_valid.append(raw_pos)
        else:
            all_valid = valid_raw_positions + [len(seq)]

        if self._positions is not None:
            indices = _validate_positions(
                self._positions,
                max_position=len(all_valid) - 1,
                min_position=0,
            )
            return [all_valid[i] for i in indices]

        return all_valid

    def _select_positions(
        self, valid_positions: list[int], rng: np.random.Generator
    ) -> list[int]:
        if len(valid_positions) < self.num_insertions:
            raise ValueError(
                f"Not enough valid positions ({len(valid_positions)}) "
                f"for {self.num_insertions} insertions"
            )
        chosen = rng.choice(
            valid_positions,
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

    def _select_marker_tags(
        self, seq: str, positions: list[int], strands: list[str], rng: np.random.Generator
    ) -> list[str]:
        """Build marker tags for each position."""
        if self.insertion_mode == 'ordered':
            names = self._marker_names
        else:
            # unordered: randomly select from available markers
            idxs = rng.choice(len(self._marker_names), size=self.num_insertions, replace=False)
            names = [self._marker_names[int(i)] for i in idxs]
        
        tags = []
        for i, (pos, strand, name) in enumerate(zip(positions, strands, names)):
            if self._marker_length > 0:
                content = seq[pos:pos + self._marker_length]
            else:
                content = ''
            tags.append(build_marker_tag(name, content, strand))
        return tags

    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        seq = parent_seqs[0]
        if rng is None:
            raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG")

        valid_positions = self._get_valid_marker_positions(seq)
        positions = self._select_positions(valid_positions, rng)
        strands = self._select_strands(rng)
        marker_tags = self._select_marker_tags(seq, positions, strands, rng)

        return {
            'positions': positions,
            'strands': strands,
            'marker_tags': marker_tags,
        }

    def compute_seq_from_card(
        self,
        parent_seqs: list[str],
        card: dict,
    ) -> dict:
        seq = parent_seqs[0]
        positions = list(card['positions'])
        marker_tags = list(card['marker_tags'])

        # Insert markers from right to left to preserve positions
        inserts = sorted(zip(positions, marker_tags), key=lambda x: x[0], reverse=True)
        for pos, tag in inserts:
            if self._marker_length > 0:
                # Region marker: replace content
                seq = seq[:pos] + tag + seq[pos + self._marker_length:]
            else:
                # Zero-length marker: insert
                seq = seq[:pos] + tag + seq[pos:]
        return {'seq_0': seq}

    def _get_copy_params(self) -> dict:
        return {
            'parent_pool': self.parent_pools[0],
            'markers': self._marker_names,
            'num_insertions': self.num_insertions,
            'positions': self._positions,
            'strand': self._strand,
            'marker_length': self._marker_length,
            'insertion_mode': self.insertion_mode,
            'mode': self.mode,
            'num_hybrid_states': self.num_states if self.mode == 'hybrid' else None,
            'name': None,
            'iter_order': self.iter_order,
        }
