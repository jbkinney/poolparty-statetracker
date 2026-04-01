"""State operation mixins for Pool class."""

from typing_extensions import Self

from ..types import CardsType, Integral, Optional, Pool_type, Real, Sequence, Union


class StateOpsMixin:
    """Mixin providing state operation methods for Pool."""

    def repeat(
        self,
        times: Integral,
        prefix: Optional[str] = None,
        iter_order: Optional[Real] = None,
        cards: CardsType = None,
    ) -> Self:
        """Repeat this pool's states a specified number of times.

        Parameters
        ----------
        times : Integral
            The number of times to repeat the pool's states. Must be >= 1.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        cards : list[str] or dict, optional
            Design card keys to include. Available keys: ``'repeat_index'``.

        Returns
        -------
        Pool
            A new Pool with ``times`` as many states as this pool.

        Raises
        ------
        ValueError
            If ``times`` is less than 1.
        """
        from ..state_ops.repeat import repeat

        return repeat(
            pool=self,
            times=times,
            prefix=prefix,
            iter_order=iter_order,
            cards=cards,
        )

    def sample(
        self,
        num_seqs: Optional[Integral] = None,
        seq_states: Optional[Sequence[Integral]] = None,
        seed: Optional[Integral] = None,
        with_replacement: bool = True,
        prefix: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> Self:
        """Sample states from this pool.

        Parameters
        ----------
        num_seqs : Optional[Integral], default=None
            Number of states to sample randomly. Mutually exclusive with
            ``seq_states``.
        seq_states : Optional[Sequence[Integral]], default=None
            Explicit list of state indices to select. Mutually exclusive with
            ``num_seqs``.
        seed : Optional[Integral], default=None
            Random seed for deterministic sampling. Only used with ``num_seqs``.
        with_replacement : bool, default=True
            Whether to sample with replacement. If False, ``num_seqs`` must be
            <= ``pool.num_states``.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.

        Returns
        -------
        Pool
            A Pool containing the sampled states.

        Raises
        ------
        ValueError
            If both ``num_seqs`` and ``seq_states`` are provided, or if neither
            is provided. If ``with_replacement`` is False and ``num_seqs``
            exceeds the pool's state count.
        """
        from ..state_ops.sample import sample

        return sample(
            pool=self,
            num_seqs=num_seqs,
            seq_states=seq_states,
            seed=seed,
            with_replacement=with_replacement,
            prefix=prefix,
            iter_order=iter_order,
        )

    def shuffle_states(
        self,
        seed: Optional[Integral] = None,
        permutation: Optional[Sequence[Integral]] = None,
        prefix: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> Self:
        """Randomly permute this pool's states.

        Parameters
        ----------
        seed : Optional[Integral], default=None
            Random seed for deterministic shuffling.
        permutation : Optional[Sequence[Integral]], default=None
            Custom permutation to use. If provided, ``seed`` must not be
            specified.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.

        Returns
        -------
        Pool
            A Pool with the same states in a randomly permuted order.
        """
        from ..state_ops.state_shuffle import state_shuffle

        return state_shuffle(
            pool=self,
            seed=seed,
            permutation=permutation,
            prefix=prefix,
            iter_order=iter_order,
        )

    def slice_states(
        self,
        key: Union[Integral, slice],
        prefix: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> Self:
        """Select a subset of this pool's states by index or slice.

        Parameters
        ----------
        key : Union[Integral, slice]
            Integer index or slice specifying which states to include.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.

        Returns
        -------
        Pool
            A Pool containing the selected states.
        """
        from ..state_ops.state_slice import state_slice

        return state_slice(
            pool=self,
            key=key,
            prefix=prefix,
            iter_order=iter_order,
        )
