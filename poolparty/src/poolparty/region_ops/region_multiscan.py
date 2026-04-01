"""Insert multiple XML region tags into a sequence."""

from numbers import Real

import numpy as np

from poolparty.types import CardsType, Integral, Literal, ModeType, MultiPositionsType, Optional, RegionType, Seq, Sequence, Union

from ..operation import Operation
from ..utils.dna_seq import DnaSeq
from ..utils.parsing_utils import build_region_tags, get_nontag_positions, nontag_pos_to_literal_pos, strip_all_tags
from ..utils.scan_utils import _is_valid_combo, _normalize_region_lengths, enumerate_multiscan_combinations
from ..utils.seq_utils import validate_positions


def _is_per_insert_positions(positions) -> bool:
    """Detect whether positions is per-insert (list of lists) vs shared."""
    if positions is None or isinstance(positions, slice):
        return False
    if isinstance(positions, (list, tuple)) and len(positions) > 0:
        return isinstance(positions[0], (list, tuple))
    return False


def region_multiscan(
    pool,
    tag_names,
    num_insertions: int,
    positions: MultiPositionsType = None,
    region: RegionType = None,
    remove_tags: Optional[bool] = None,
    region_length: int | Sequence[int] = 0,
    insertion_mode: Literal["ordered", "unordered"] = "ordered",
    min_spacing: Optional[int] = None,
    max_spacing: Optional[int] = None,
    prefix: Optional[str] = None,
    mode: ModeType = "random",
    num_states: Optional[Integral] = None,
    iter_order: Optional[Real] = None,
    cards: CardsType = None,
    _factory_name: Optional[str] = None,
):
    """
    Insert multiple XML-style region tags into a sequence.

    Parameters
    ----------
    pool : Pool or str
        Input Pool or sequence string to insert tags into.
    tag_names : Sequence[str] or str
        Tag name(s) to insert. If a single string, used for all insertions.
    num_insertions : Integral
        Number of region tags to insert.
    positions : PositionsType or list[PositionsType], default=None
        Valid insertion positions (0-based, nontag-relative). Flat list/slice/None
        for shared positions; list-of-lists for per-insert positions (one per insert).
    region : RegionType, default=None
        Region to constrain the scan to. Can be region name (str) or [start, stop].
    region_length : int or Sequence[int], default=0
        Length of sequence to encompass per region. Single int for uniform length,
        or a sequence of ints for per-region lengths (one per insertion).
    insertion_mode : str, default='ordered'
        How to assign tags to positions:
        - 'ordered': tag_names[i] goes to the i-th selected position (left to right)
        - 'unordered': all valid assignments of tags to positions are enumerated
    min_spacing : Optional[int], default=None
        Minimum gap between end of one region and start of next.
        Default: 0 (non-overlapping, touching OK).
    max_spacing : Optional[int], default=None
        Maximum gap between adjacent regions. None = unbounded.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.
    mode : ModeType, default='random'
        Position selection mode: 'random' or 'sequential'.
    num_states : Optional[Integral], default=None
        Number of states. If None, auto-determined for sequential mode.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.

    Returns
    -------
    Pool
        A Pool yielding sequences with multiple region tags inserted.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..party import get_active_party

    pool = from_seq(pool) if isinstance(pool, str) else pool

    party = get_active_party()
    region_names = [tag_names] if isinstance(tag_names, str) else list(tag_names)
    region_lengths = _normalize_region_lengths(region_length, num_insertions)

    op = RegionMultiScanOp(
        parent_pool=pool,
        tag_names=tag_names,
        num_insertions=int(num_insertions),
        positions=positions,
        region_constraint=region,
        remove_tags=remove_tags,
        region_length=region_length,
        insertion_mode=insertion_mode,
        min_spacing=min_spacing,
        max_spacing=max_spacing,
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        name=None,
        iter_order=iter_order,
        cards=cards,
        _factory_name=_factory_name,
    )

    registered_regions = []
    for i, region_name in enumerate(region_names):
        registered_regions.append(party.register_region(region_name, region_lengths[i]))
    pool_class = type(pool)
    result_pool = pool_class(operation=op)
    for registered_region in registered_regions:
        result_pool.add_region(registered_region)

    return result_pool


class RegionMultiScanOp(Operation):
    """Insert multiple XML region tags at selected positions."""

    factory_name = "region_multiscan"
    design_card_keys = ["combination_index", "starts", "ends", "names", "region_seqs"]

    def __init__(
        self,
        parent_pool,
        tag_names,
        num_insertions: int,
        positions: MultiPositionsType = None,
        region_constraint: RegionType = None,
        remove_tags: Optional[bool] = None,
        region_length: int | Sequence[int] = 0,
        insertion_mode: str = "ordered",
        min_spacing: Optional[int] = None,
        max_spacing: Optional[int] = None,
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        cards: CardsType = None,
        _factory_name: Optional[str] = None,
    ) -> None:
        if _factory_name is not None:
            self.factory_name = _factory_name

        if num_insertions < 1:
            raise ValueError(f"num_insertions must be >= 1, got {num_insertions}")
        if mode not in ("random", "sequential"):
            raise ValueError(f"mode must be 'random' or 'sequential', got '{mode}'")

        self._region_lengths = _normalize_region_lengths(region_length, num_insertions)
        for rl in self._region_lengths:
            if rl < 0:
                raise ValueError(f"region_length must be >= 0, got {rl}")

        self._positions = positions
        self._is_per_insert = _is_per_insert_positions(positions)
        if self._is_per_insert and len(positions) != num_insertions:
            raise ValueError(
                f"per-insert positions has {len(positions)} sublists, "
                f"but num_insertions={num_insertions}"
            )
        self._mode = mode
        self._min_spacing = min_spacing if min_spacing is not None else 0
        self._max_spacing = max_spacing
        self.num_insertions = num_insertions
        self.insertion_mode = insertion_mode
        self._region_names = self._coerce_tag_names(tag_names)
        self._validate_region_counts()

        self._sequential_cache: list[tuple[int, ...]] | None = None

        if isinstance(region_constraint, str):
            from ..party import get_active_party

            party = get_active_party()
            try:
                constraint_region = party.get_region_by_name(region_constraint)
                self._seq_length = constraint_region.seq_length
            except (ValueError, KeyError):
                self._seq_length = parent_pool.seq_length
        elif region_constraint is not None:
            self._seq_length = int(region_constraint[1]) - int(region_constraint[0])
        else:
            self._seq_length = parent_pool.seq_length

        natural_num_states = None
        if mode == "sequential":
            if self._seq_length is not None:
                natural_num_states = self._build_caches()
                if num_states is None:
                    num_states = natural_num_states
            else:
                raise ValueError(
                    "mode='sequential' requires known scan geometry. "
                    "Provide a parent with known seq_length, or a region "
                    "constraint with known length."
                )

        super().__init__(
            parent_pools=[parent_pool],
            num_states=num_states,
            mode=mode,
            seq_length=parent_pool.seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            region=region_constraint,
            remove_tags=remove_tags,
            _natural_num_states=natural_num_states,
            cards=cards,
        )

    def _coerce_tag_names(self, tag_names: Union[Sequence[str], str]) -> list[str]:
        """Normalize tag_names input to a list of names."""
        if isinstance(tag_names, str):
            tag_names = [tag_names]
        if not tag_names:
            raise ValueError("tag_names must not be empty")
        return list(tag_names)

    def _validate_region_counts(self) -> None:
        """Validate region counts against insertion_mode."""
        if self.insertion_mode not in ("ordered", "unordered"):
            raise ValueError("insertion_mode must be one of 'ordered', 'unordered'")
        if len(self._region_names) != self.num_insertions:
            raise ValueError(
                f"len(tag_names) ({len(self._region_names)}) must equal "
                f"num_insertions ({self.num_insertions})"
            )

    def _compute_valid_positions_for_insert(self, seq_length: int, insert_idx: int) -> list[int]:
        """Compute all valid start positions for a specific insert based on its region length."""
        rl = self._region_lengths[insert_idx]
        if rl > 0:
            max_start = seq_length - rl
            if max_start < 0:
                return []
            return list(range(max_start + 1))
        else:
            return list(range(seq_length + 1))

    def _build_per_insert_positions(self, seq_length: int) -> list[list[int]]:
        """Build per-insert valid position lists from seq_length and user positions."""
        result: list[list[int]] = []
        for i in range(self.num_insertions):
            all_valid_i = self._compute_valid_positions_for_insert(seq_length, i)
            if self._is_per_insert:
                p_list = list(self._positions[i])
                validated = validate_positions(
                    p_list, max_position=len(all_valid_i) - 1, min_position=0
                )
                result.append([all_valid_i[j] for j in validated])
            elif self._positions is not None:
                validated = validate_positions(
                    self._positions, max_position=len(all_valid_i) - 1, min_position=0
                )
                result.append([all_valid_i[j] for j in validated])
            else:
                result.append(all_valid_i)
        return result

    def _build_caches(self) -> int:
        """Enumerate valid combinations for sequential mode. Returns num_states."""
        seq_length = self._seq_length
        if seq_length is None:
            return 1

        valid = self._build_per_insert_positions(seq_length)

        self._sequential_cache = enumerate_multiscan_combinations(
            valid_positions=valid,
            num_insertions=self.num_insertions,
            region_length=self._region_lengths,
            insertion_mode=self.insertion_mode,
            min_spacing=self._min_spacing,
            max_spacing=self._max_spacing,
        )

        return len(self._sequential_cache)

    def _get_valid_region_indices(self, seq: str) -> list[list[int]]:
        """Return per-insert valid nontag indices for region tag insertion."""
        nontag_positions = get_nontag_positions(seq)
        num_nontag = len(nontag_positions)

        result: list[list[int]] = []
        for i in range(self.num_insertions):
            rl = self._region_lengths[i]
            if rl > 0:
                max_valid_idx = num_nontag - rl
                if max_valid_idx < 0:
                    result.append([])
                    continue
                all_valid_i = list(range(max_valid_idx + 1))
            else:
                all_valid_i = list(range(num_nontag + 1))

            if self._is_per_insert:
                p_list = list(self._positions[i])
                validated = validate_positions(
                    p_list, max_position=len(all_valid_i) - 1, min_position=0
                )
                result.append([all_valid_i[j] for j in validated])
            elif self._positions is not None:
                validated = validate_positions(
                    self._positions, max_position=len(all_valid_i) - 1, min_position=0
                )
                result.append([all_valid_i[j] for j in validated])
            else:
                result.append(all_valid_i)
        return result

    def _select_indices_random(
        self, valid_indices: list[list[int]], rng: np.random.Generator
    ) -> tuple[int, ...]:
        """Select an assignment tuple for random mode.

        Returns an assignment tuple where result[i] is the position for insert i.
        """
        for i, p_list in enumerate(valid_indices):
            if len(p_list) == 0:
                raise ValueError(
                    f"No valid positions for insert {i} (region_length={self._region_lengths[i]}). "
                    "The sequence may be too short or positions too constrained."
                )

        has_varying = len(set(self._region_lengths)) > 1
        valid_sets: list[set[int]] | None = None
        if has_varying:
            valid_sets = [set(vl) for vl in valid_indices]

        max_attempts = 1000
        for _ in range(max_attempts):
            combo = tuple(
                p_list[int(rng.integers(0, len(p_list)))]
                for p_list in valid_indices
            )

            if len(set(combo)) < self.num_insertions:
                continue

            if self.insertion_mode == "ordered":
                combo = tuple(sorted(combo))
                if has_varying and valid_sets is not None:
                    if not all(pos in valid_sets[i] for i, pos in enumerate(combo)):
                        continue

            if _is_valid_combo(combo, self._region_lengths, self._min_spacing, self._max_spacing):
                return combo

        raise ValueError(
            f"Cannot find valid {self.num_insertions}-position selection after "
            f"{max_attempts} attempts with min_spacing={self._min_spacing}, "
            f"max_spacing={self._max_spacing}"
        )

    def _get_names_for_combo(self, combo: tuple[int, ...]) -> list[str]:
        """Return region names for a given assignment combo.

        In ordered mode, names follow the original region order.
        In unordered mode, names follow the original region order too — the
        combo itself encodes which position each insert gets.
        """
        return list(self._region_names)

    def _build_tags(self, seq: str, combo: tuple[int, ...], names: list[str]) -> list[str]:
        """Build region tag strings for given assignment combo and names."""
        tags = []
        for i, (idx, tag_name) in enumerate(zip(combo, names)):
            rl = self._region_lengths[i]
            if rl > 0:
                literal_start = nontag_pos_to_literal_pos(seq, idx)
                literal_end = nontag_pos_to_literal_pos(seq, idx + rl)
                content = seq[literal_start:literal_end]
                content = strip_all_tags(content)
            else:
                content = ""
            tags.append(build_region_tags(tag_name, content))
        return tags

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        """Return Seq with region tags inserted and design card."""
        from ..utils.style_utils import SeqStyle

        seq = parents[0].string
        input_style = parents[0].style

        combination_index = None

        if self._mode == "sequential":
            state = self.state.value
            state = 0 if state is None else state

            cache = self._sequential_cache
            if cache is None:
                valid = self._get_valid_region_indices(seq)
                cache = enumerate_multiscan_combinations(
                    valid_positions=valid,
                    num_insertions=self.num_insertions,
                    region_length=self._region_lengths,
                    insertion_mode=self.insertion_mode,
                    min_spacing=self._min_spacing,
                    max_spacing=self._max_spacing,
                )

            combo_idx = state % len(cache)
            combination_index = combo_idx
            combo = cache[combo_idx]
            names = self._get_names_for_combo(combo)
        else:
            if rng is None:
                raise RuntimeError(f"{self._mode.capitalize()} mode requires RNG")
            valid = self._get_valid_region_indices(seq)
            combo = self._select_indices_random(valid, rng)
            names = self._get_names_for_combo(combo)

        tags = self._build_tags(seq, combo, names)

        # Build (position, tag, name, region_length, insert_index) tuples
        # sorted by position for left-to-right output construction
        inserts = sorted(
            zip(combo, tags, names, self._region_lengths, range(self.num_insertions)),
            key=lambda x: x[0],
        )

        result_parts: list[str] = []
        style_parts: list[SeqStyle] = []
        prev_end_idx = 0

        for nt_idx, tag, _, rl, _ in inserts:
            if prev_end_idx < nt_idx:
                start_literal = nontag_pos_to_literal_pos(seq, prev_end_idx)
                end_literal = nontag_pos_to_literal_pos(seq, nt_idx)
                result_parts.append(seq[start_literal:end_literal])
                if input_style is not None:
                    style_parts.append(input_style[start_literal:end_literal])

            result_parts.append(tag)

            if input_style is not None:
                if rl > 0:
                    opening_tag_len = tag.index(">") + 1
                    closing_tag_len = len(f"</{tag[1:tag.index('>')]}>")
                    content_start = nontag_pos_to_literal_pos(seq, nt_idx)
                    content_end = nontag_pos_to_literal_pos(seq, nt_idx + rl)
                    style_parts.append(SeqStyle.empty(opening_tag_len))
                    style_parts.append(input_style[content_start:content_end])
                    style_parts.append(SeqStyle.empty(closing_tag_len))
                else:
                    style_parts.append(SeqStyle.empty(len(tag)))

            if rl > 0:
                prev_end_idx = nt_idx + rl
            else:
                prev_end_idx = nt_idx

        nontag_positions = get_nontag_positions(seq)
        if prev_end_idx < len(nontag_positions):
            start_literal = nontag_pos_to_literal_pos(seq, prev_end_idx)
            result_parts.append(seq[start_literal:])
            if input_style is not None:
                style_parts.append(input_style[start_literal:])
        elif prev_end_idx == len(nontag_positions):
            last_nontag_literal = nontag_positions[-1] if nontag_positions else 0
            if last_nontag_literal + 1 < len(seq):
                result_parts.append(seq[last_nontag_literal + 1:])
                if input_style is not None:
                    style_parts.append(input_style[last_nontag_literal + 1:])

        result_seq = "".join(result_parts)
        output_style = SeqStyle.join(style_parts) if input_style is not None else None

        if self._party.suppress_cards:
            card = {}
        else:
            sorted_names = [n for _, _, n, _, _ in inserts]
            sorted_region_seqs = [t for _, t, _, _, _ in inserts]
            sorted_starts = [pos for pos, _, _, _, _ in inserts]
            sorted_stops = [pos + rl for pos, _, _, rl, _ in inserts]
            card = {
                "combination_index": combination_index,
                "starts": sorted_starts,
                "ends": sorted_stops,
                "names": sorted_names,
                "region_seqs": sorted_region_seqs,
            }

        output_seq = DnaSeq(result_seq, output_style)
        return output_seq, card
