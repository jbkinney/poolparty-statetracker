"""Restriction enzyme recognition sites database.

This module provides a curated collection of common restriction enzyme
recognition sequences for use in library design filtering.

Recognition sequences use IUPAC ambiguity codes:
    R = A or G (purine)
    Y = C or T (pyrimidine)
    S = G or C (strong)
    W = A or T (weak)
    K = G or T (keto)
    M = A or C (amino)
    B = C, G, or T (not A)
    D = A, G, or T (not C)
    H = A, C, or T (not G)
    V = A, C, or G (not T)
    N = any base
"""

# Dictionary mapping enzyme names to their recognition sequences.
# Recognition sequences are uppercase and may contain IUPAC ambiguity codes.
# Only the recognition site is provided (not cut positions).
ENZYME_SITES: dict[str, str] = {
    # ==========================================================================
    # Common 6-base cutters (Type II)
    # ==========================================================================
    "EcoRI": "GAATTC",
    "BamHI": "GGATCC",
    "HindIII": "AAGCTT",
    "XhoI": "CTCGAG",
    "NdeI": "CATATG",
    "XbaI": "TCTAGA",
    "SalI": "GTCGAC",
    "PstI": "CTGCAG",
    "SphI": "GCATGC",
    "SacI": "GAGCTC",
    "SacII": "CCGCGG",
    "KpnI": "GGTACC",
    "NcoI": "CCATGG",
    "NheI": "GCTAGC",
    "NotI": "GCGGCCGC",  # 8-base cutter
    "EcoRV": "GATATC",
    "ClaI": "ATCGAT",
    "BglII": "AGATCT",
    "SpeI": "ACTAGT",
    "AvrII": "CCTAGG",
    "MluI": "ACGCGT",
    "NruI": "TCGCGA",
    "SnaBI": "TACGTA",
    "StuI": "AGGCCT",
    "ScaI": "AGTACT",
    "PvuI": "CGATCG",
    "PvuII": "CAGCTG",
    "ApaI": "GGGCCC",
    "BspEI": "TCCGGA",
    "AgeI": "ACCGGT",
    "AflII": "CTTAAG",
    "AscI": "GGCGCGCC",  # 8-base cutter
    "FseI": "GGCCGGCC",  # 8-base cutter
    "PacI": "TTAATTAA",  # 8-base cutter
    "PmeI": "GTTTAAAC",  # 8-base cutter
    "SwaI": "ATTTAAAT",  # 8-base cutter
    "SfiI": "GGCCNNNNNGGCC",  # 13-base with internal N's
    "AatII": "GACGTC",
    "AccI": "GTMKAC",  # Degenerate
    "BstBI": "TTCGAA",
    "Eco53kI": "GAGCTC",
    "HpaI": "GTTAAC",
    "MfeI": "CAATTG",
    "NsiI": "ATGCAT",
    "SbfI": "CCTGCAGG",  # 8-base cutter
    "SgrAI": "CRCCGGYG",  # Degenerate 8-base
    "XmaI": "CCCGGG",
    "SmaI": "CCCGGG",  # Same as XmaI, different cut position
    # ==========================================================================
    # Golden Gate / Type IIS enzymes (cut outside recognition site)
    # ==========================================================================
    "BsaI": "GGTCTC",
    "BsmBI": "CGTCTC",
    "BbsI": "GAAGAC",
    "SapI": "GCTCTTC",
    "AarI": "CACCTGC",
    "BspQI": "GCTCTTC",
    "BtgZI": "GCGATG",
    "BsmFI": "GGGAC",
    "FokI": "GGATG",
    "BtsI": "GCAGTG",
    "BspMI": "ACCTGC",
    "Esp3I": "CGTCTC",  # Same as BsmBI (isoschizomer)
    "BpiI": "GAAGAC",  # Same as BbsI (isoschizomer)
    "LguI": "GCTCTTC",  # Same as SapI (isoschizomer)
    # ==========================================================================
    # 4-base cutters (frequent cutters)
    # ==========================================================================
    "DpnI": "GATC",  # Cuts methylated DNA only
    "DpnII": "GATC",  # Cuts unmethylated DNA only
    "Sau3AI": "GATC",
    "MboI": "GATC",
    "AluI": "AGCT",
    "HaeIII": "GGCC",
    "HhaI": "GCGC",
    "MspI": "CCGG",
    "HpaII": "CCGG",  # Methylation-sensitive
    "TaqI": "TCGA",
    "RsaI": "GTAC",
    "CviQI": "GTAC",  # Same as RsaI
    "HinP1I": "GCGC",
    "BfaI": "CTAG",
    "NlaIII": "CATG",
    "CviAII": "CATG",
    # ==========================================================================
    # Additional blunt-end cutters (not already listed above)
    # ==========================================================================
    "HincII": "GTYRAC",  # Degenerate blunt cutter
    "BalI": "TGGCCA",
    "DraI": "TTTAAA",
    "EcoICRI": "GAGCTC",
    "Ecl136II": "GAGCTC",
    "MscI": "TGGCCA",
    "NaeI": "GCCGGC",
    "PmlI": "CACGTG",
    "ZraI": "GACGTC",
    # ==========================================================================
    # Enzymes with degenerate/ambiguous recognition sequences
    # ==========================================================================
    "BstYI": "RGATCY",  # R=A/G, Y=C/T -> recognizes AGATCC, AGATCT, GGATCC, GGATCT
    "StyI": "CCWWGG",  # W=A/T -> recognizes CCAAGG, CCATGG, CCTAGG, CCTTGG
    "BstNI": "CCWGG",  # Recognizes CCAGG, CCTGG
    "SfcI": "CTRYAG",  # Recognizes CTRAAG, CTRGAG, CTRYAG patterns
    "BsrFI": "RCCGGY",
    "PspGI": "CCWGG",
    "AvaI": "CYCGRG",  # Recognizes multiple patterns
    "BanII": "GRGCYC",
    "HinfI": "GANTC",
    "DdeI": "CTNAG",
    "AflIII": "ACRYGT",
    "BanI": "GGYRCC",
    "AccB7I": "CCANNNNNTGG",
    "PflMI": "CCANNNNNTGG",  # Same as AccB7I
    "BstAPI": "GCANNNNNTGC",
    "DrdI": "GACNNNNNNGTC",
    "AhdI": "GACNNNNNGTC",
    "BsrDI": "GCAATG",
    "BsrGI": "TGTACA",
    "BsiWI": "CGTACG",
    "BlpI": "GCTNAGC",
    "Bsu36I": "CCTNAGG",
    "PsiI": "TTATAA",
    "MseI": "TTAA",
    "Tsp509I": "AATT",
}

# Preset groups of enzymes for common use cases
ENZYME_PRESETS: dict[str, list[str]] = {
    # Golden Gate assembly enzymes
    "golden_gate": [
        "BsaI",
        "BsmBI",
        "BbsI",
        "SapI",
        "AarI",
        "BspQI",
        "Esp3I",
        "BpiI",
    ],
    # Most commonly used cloning enzymes
    "common": [
        "EcoRI",
        "BamHI",
        "HindIII",
        "XhoI",
        "NdeI",
        "XbaI",
        "SalI",
        "PstI",
        "SacI",
        "KpnI",
        "NcoI",
        "NheI",
        "NotI",
        "SpeI",
        "EcoRV",
        "SmaI",
        "BglII",
        "ClaI",
        "ApaI",
        "MluI",
    ],
    # Standard multiple cloning site enzymes
    "mcs": [
        "EcoRI",
        "SacI",
        "KpnI",
        "SmaI",
        "BamHI",
        "XbaI",
        "SalI",
        "PstI",
        "SphI",
        "HindIII",
    ],
    # Gibson assembly - common enzymes to avoid in Gibson primers
    "gibson": [
        "EcoRI",
        "BamHI",
        "HindIII",
        "XhoI",
        "NdeI",
        "NotI",
        "XbaI",
        "SpeI",
        "PstI",
        "SacI",
    ],
    # Frequent (4-base) cutters
    "frequent_cutters": [
        "DpnI",
        "DpnII",
        "Sau3AI",
        "MboI",
        "AluI",
        "HaeIII",
        "HhaI",
        "MspI",
        "HpaII",
        "TaqI",
        "RsaI",
        "BfaI",
        "NlaIII",
        "MseI",
    ],
    # 8-base cutters (rare cutters)
    "rare_cutters": [
        "NotI",
        "AscI",
        "FseI",
        "PacI",
        "PmeI",
        "SwaI",
        "SbfI",
    ],
    # Blunt-end cutters
    "blunt": [
        "EcoRV",
        "SmaI",
        "StuI",
        "ScaI",
        "PvuII",
        "HpaI",
        "NruI",
        "SnaBI",
        "AluI",
        "HaeIII",
        "DraI",
        "NaeI",
        "PmlI",
    ],
}


def get_enzyme_site(enzyme_name: str) -> str:
    """Get recognition site for an enzyme by name.

    Parameters
    ----------
    enzyme_name : str
        Name of the restriction enzyme (case-insensitive).

    Returns
    -------
    str
        Recognition sequence (uppercase, may contain IUPAC codes).

    Raises
    ------
    ValueError
        If enzyme name is not found in the database.
    """
    # Try exact match first
    if enzyme_name in ENZYME_SITES:
        return ENZYME_SITES[enzyme_name]

    # Try case-insensitive match
    enzyme_lower = enzyme_name.lower()
    for name, site in ENZYME_SITES.items():
        if name.lower() == enzyme_lower:
            return site

    raise ValueError(
        f"Unknown enzyme: {enzyme_name!r}. "
        f"Use one of: {', '.join(sorted(ENZYME_SITES.keys())[:10])}... "
        f"({len(ENZYME_SITES)} total enzymes available)"
    )


def get_preset_enzymes(preset_name: str) -> list[str]:
    """Get list of enzyme names for a preset.

    Parameters
    ----------
    preset_name : str
        Name of the preset (case-insensitive).

    Returns
    -------
    list[str]
        List of enzyme names in the preset.

    Raises
    ------
    ValueError
        If preset name is not found.
    """
    # Try exact match first
    if preset_name in ENZYME_PRESETS:
        return ENZYME_PRESETS[preset_name]

    # Try case-insensitive match
    preset_lower = preset_name.lower()
    for name, enzymes in ENZYME_PRESETS.items():
        if name.lower() == preset_lower:
            return enzymes

    raise ValueError(
        f"Unknown preset: {preset_name!r}. "
        f"Available presets: {', '.join(sorted(ENZYME_PRESETS.keys()))}"
    )
