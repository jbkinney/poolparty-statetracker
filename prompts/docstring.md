Write a concice but thorough docstring for this funciton using the following style:
    """
    Create a Pool containing the specified sequences.

    Parameters
    ----------
    seqs : Sequence[str]
        Sequence of string sequences to include in the pool.
    seq_names : Optional[Sequence[str]], default=None
        Optional sequence of names for each sequence. If not provided, names are auto-generated.
    mode : ModeType, default='random'
        Sequence selection mode: 'sequential', 'random', or 'hybrid'.
    num_hybrid_states : Optional[int], default=None
        Number of pool states when using 'hybrid' mode (ignored for other modes).
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying Operation.
    iter_order : Real, default=0
        Iteration order priority for the resulting Pool.
    op_iter_order : Real, default=0
        Iteration order priority for the underlying Operation.

    Returns
    -------
    Pool_type
        A Pool object yielding the provided sequences using the specified selection mode.
    """