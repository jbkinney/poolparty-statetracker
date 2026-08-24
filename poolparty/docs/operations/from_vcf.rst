from_vcf
========

Create a pool of reference-genome windows around each variant in a VCF file.
Every variant contributes a window carrying its alternate allele, and every
distinct site contributes one window carrying the reference allele, so a pool can
be scored as reference/alternate pairs.

.. code-block:: python

    import poolparty as pp
    pp.init()

.. note::

   Windows are always on the reference plus strand. Compose
   :func:`~poolparty.rc` for the reverse complement.

   Windows are uppercased, so soft-mask annotation is not preserved. Without
   this, a reference and its variant would differ in case as well as base at the
   variant position — an artifact of where each came from, not of the data.

   Sequence length is ``flank_left + len(allele) + flank_right``, so a pool
   containing indels holds sequences of differing lengths and its
   ``seq_length`` is ``None``. Operations that require a defined
   ``seq_length`` — among them :func:`~poolparty.recombine`,
   :func:`~poolparty.subseq_scan` and the scan operations — do not accept such a
   pool. Pass ``variant_types=['snv']`` for a uniform pool — substitutions vary in
   width, so including them does not give one. For a large file, restrict the
   VCF upstream (``bcftools view -v snps``).

----

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Parameter
     - Type
     - Default
     - Description
   * - ``vcf_path``
     - ``str``
     - *(required)*
     - Path to a VCF file. ``.gz`` files are read as gzip, which covers bgzip.
       BCF and indexed random access are not supported.
   * - ``fasta_path``
     - ``str``
     - *(required)*
     - Path to the reference FASTA, indexed with ``pyfaidx``.
   * - ``flank_left``
     - ``int``
     - *(required)*
     - Bases of reference sequence before the variant.
   * - ``flank_right``
     - ``int``
     - *(required)*
     - Bases of reference sequence after the variant.
   * - ``alleles``
     - ``str``
     - ``'both'``
     - ``'alt'`` emits one window per variant; ``'ref'`` one per distinct site;
       ``'both'`` emits both.
   * - ``variant_types``
     - ``list[str] | None``
     - ``None``
     - Keep only these types: ``'snv'``, ``'substitution'``, ``'insertion'``,
       ``'deletion'``. Only ``['snv']`` gives a uniform-length pool; a
       substitution's width varies with its allele. Excluded records contribute
       no window at all, not even a reference one.
   * - ``max_allele_length``
     - ``int | None``
     - ``100``
     - Skip records whose ``REF`` or ``ALT`` exceeds this many bases; ``None``
       disables the check. A single long deletion otherwise widens its own
       window by its length — ClinVar's longest ``REF`` is 9,983 bases.
   * - ``info_fields``
     - ``list[str] | None``
     - ``None``
     - ``INFO`` keys to expose as design cards, prefixed with ``info_``. Values
       are taken verbatim, not split per allele.
   * - ``prefix``
     - ``str | None``
     - ``None``
     - Prefix for generated sequence names.
   * - ``style``
     - ``str | None``
     - ``None``
     - Display style applied to every generated sequence.
   * - ``cards``
     - ``dict | list | None``
     - ``None``
     - Design card keys to include. Available keys: ``'chrom'``, ``'pos'``,
       ``'ref'``, ``'alt'``, ``'allele'``, ``'variant_type'``, ``'variant_id'``,
       ``'filter'``, ``'window_start'``, ``'window_stop'``, plus any
       ``info_``-prefixed keys named in ``info_fields``.

----

.. note::

   Only the most commonly used parameters are shown above. For the full
   parameter list, see :func:`~poolparty.from_vcf` in the
   :doc:`API Reference </api>`.

Examples
--------

Reference and alternate windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three records — two alternates at position 21 and an insertion at 31 — give five
windows. Position 21 yields a single reference window despite carrying two
alternate alleles.

.. code-block:: python

    pool = pp.from_vcf("variants.vcf", "ref.fa", 5, 5)
    pool.print_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">pool[0]: seq_length=None, num_states=5</em>
    chr1_21_A&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;GCAATACGTTG<br>
    chr1_21_A_G&nbsp;&nbsp;&nbsp;GCAATGCGTTG<br>
    chr1_21_A_T&nbsp;&nbsp;&nbsp;GCAATTCGTTG<br>
    chr1_31_A&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;GCAATACGTTG<br>
    chr1_31_A_ACG&nbsp;GCAATACGCGTTG
    </div>

``seq_length`` is ``None`` because the insertion at 31 makes that window two
bases longer than the others.

Variants only, with annotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``alleles='alt'`` drops the reference windows. ``info_fields`` carries values
from the VCF ``INFO`` column onto each sequence, which is how population
frequencies or clinical significance reach the design card.

.. code-block:: python

    pool = pp.from_vcf(
        "variants.vcf", "ref.fa", 5, 5,
        alleles="alt",
        info_fields=["AF"],
        cards={"pos": "pos", "alt": "alt",
               "variant_type": "variant_type", "info_AF": "AF"},
    )
    pool.generate_library()

.. raw:: html

    <div class="pp-pool">
    <em class="pp-header">df — 3 rows × 6 columns</em>
    <table class="pp-df">
    <tr><th>name</th><th>seq</th><th>pos</th><th>alt</th><th>variant_type</th><th>AF</th></tr>
    <tr><td>chr1_21_A_G</td><td>GCAATGCGTTG</td><td>21</td><td>G</td><td>snv</td><td>0.02</td></tr>
    <tr><td>chr1_21_A_T</td><td>GCAATTCGTTG</td><td>21</td><td>T</td><td>snv</td><td>0.01</td></tr>
    <tr><td>chr1_31_A_ACG</td><td>GCAATACGCGTTG</td><td>31</td><td>ACG</td><td>insertion</td><td>0.30</td></tr>
    </table>
    </div>

Uniform-length windows for model input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sequence models generally need a fixed input width, and so do PoolParty's
length-dependent operations. ``variant_types`` keeps only the classes that
preserve length, which restores ``seq_length``:

.. code-block:: python

    pool = pp.from_vcf(
        "variants.vcf.gz", "hg38.fa", 100, 100,
        alleles="alt", variant_types=["snv"],
    )
    pool.seq_length          # 201, so subseq_scan and recombine accept it

For a large VCF it is cheaper to restrict the file first — ``bcftools view -v
snps`` — since windows are then never cut for records you discard.

----

Naming
------

Sequences are named in GTEx style. A reference window omits the alternate
allele, because it is shared by every alternate at that site::

    chr1_21_A          the reference at chr1:21
    chr1_21_A_G        the A>G variant
    chr1_21_A_T        the A>T variant

.. note::

   The design cards follow the name: ``alt``, ``variant_type``,
   ``variant_id``, ``filter`` and every ``info_`` key are ``None`` on a
   reference row.

   ``window_start`` and ``window_stop`` give the genomic span the window
   covers, 0-based half-open. On an alternate row they describe the reference
   span the allele replaces, so ``window_stop - window_start`` differs from
   the sequence length for indels. ``chrom`` carries the VCF's own spelling,
   which may differ from the reference's.

   ``info_`` values are taken verbatim and are not split per allele: a
   ``Number=A`` field such as ``AF`` on a multi-allelic record gives every
   alternate the whole comma-separated value. Split such records upstream
   with ``bcftools norm -m-``.

----

Rejected input
--------------

Input that cannot be turned into a window is rejected, and the counts are
reported together in a single warning. The counts are not in one unit: a
record-level reason counts rejected records, an ALT-level reason counts rejected
alternate alleles, and ``allele too long`` counts both.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Reason
     - Detail
   * - ``non dna alt``
     - ``ALT`` is empty or contains anything but ``ACGT`` — missing values,
       spanning deletions, symbolic and breakend alleles, ``N`` runs and IUPAC
       ambiguity codes.
   * - ``non dna ref``
     - ``REF`` contains anything but ``ACGT``. An ``N`` inside an assembly gap
       matches the reference, so this is checked first.
   * - ``gap in window``
     - The flanking sequence contains anything but ``ACGT``. The flanks are most
       of the window, so a gap there is as unusable as one in the allele.
   * - ``allele too long``
     - ``REF`` or ``ALT`` longer than ``max_allele_length``. An over-long ``REF``
       removes the whole record, since it sets the window width; an over-long
       ``ALT`` removes only that allele.
   * - ``variant type``
     - Excluded by ``variant_types``.
   * - ``contig absent``
     - The contig is not in the FASTA under either spelling. ``chr1`` and ``1``
       are treated as the same contig.
   * - ``off contig``
     - The window would run past the start or end of the contig.
   * - ``ref mismatch``
     - The VCF ``REF`` disagrees with the FASTA. The comparison ignores letter
       case, so soft-masked references are fine.

The counts are also available afterwards as ``pool.operation.skipped``, under the
underscored key names (``non_dna_alt``, ``allele_too_long``, and so on).

A reference window depends only on the site, so an alternate allele that cannot be
represented — symbolic, over-long, not ACGT — leaves it in place. The exception is
``variant_types``: once the caller has filtered by type, a site with no surviving
alternate is one they excluded, and it contributes no window at all.

**Above 20% reference mismatches the call raises** rather than returning a pool, at
any file size. Partly-matching references would otherwise yield a plausible library
in which every window is displaced. Check that the VCF and the FASTA use compatible
reference sequences, assemblies and coordinates.

A malformed data line — fewer than eight tab-separated fields, or a ``POS`` that
is not an integer — raises rather than being skipped, as does a VCF from which no
record survives. Blank lines are ignored.

.. note::

   The whole VCF is read into memory, and memory scales with the number of
   records times the window width. Pre-filter large files: an unfiltered
   ClinVar release at 10 kb windows does not fit in memory.

See :func:`~poolparty.from_vcf`.
