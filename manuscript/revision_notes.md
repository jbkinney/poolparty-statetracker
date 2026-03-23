# Revision notes: manuscript/main.tex vs manuscript_overleaf_26.03.17/main.tex

Sections revised: Abstract, Introduction, Methods, DMS example, Discussion.
Sections not yet revised: MPRA, SpliceAI.

## 1. Concrete over abstract

The old version leads with formal abstractions ("stateful sequence generators," "algebraic composition," "directed acyclic computation graph"). The revised version leads with what the tool *does* in practical terms — "generates sequences on demand," "chaining together a core set of operations." Jargon is introduced only when needed, not as framing.

## 2. Example-first exposition

The old Introduction paragraph 2 catalogues abstract categories of modifications (single-nucleotide changes, structural alterations, codon-level edits…). The revision replaces this with two vivid, concrete scenarios — a DMS library combining exhaustive singles, sampled higher-order mutants, and barcodes; an MPRA library tiling TFBSs in all permutations. The reader sees the *problem* before being told about the solution.

## 3. Shorter, more direct sentences

Sentence length is noticeably reduced throughout. Compound sentences with multiple subordinate clauses are broken apart. For example, the old Methods opening is a single dense paragraph mixing Pools, Operations, DAGs, and state tracking; the revision separates these into distinct, focused subsections (Pools and Operations → Sequence generation → Operation modes → StateTracker).

## 4. More granular section structure

The old "State Tracking and Sequence Generation" is one large subsection. The revision splits it into three: Sequence generation, Operation modes, and StateTracker. Similarly, "PoolParty Abstraction" becomes the more descriptive "Pools and Operations." This makes the Methods easier to navigate.

## 5. Active voice and reduced hedging

The old version uses more passive constructions ("Sequences were binned," "Models were weighted") and hedging ("can optionally carry"). The revision favors active voice and direct assertions: "PoolParty passes a state," "Each Operation decomposes the state it receives." The tone is more confident.

## 6. Simpler framing of technical concepts

The StateTracker description is a good example. The old version introduces "composition rules," "mixed-radix decomposition," and "disjoint union" up front in an abstract framework. The revision introduces the same ideas but grounds them in what the user experiences — sequential vs. random modes first, then the bookkeeping problem they create, then StateTracker as the solution.

## 7. Tighter Abstract

The old abstract front-loads implementation details (Pool objects, computation graph architecture). The revised abstract leads with *why* (libraries are essential, designing them is tedious) and *what changes* (declarative syntax, DAG, sequences on demand), deferring technical vocabulary.

## 8. Discussion is more focused

The old Discussion enumerates future extensions (constraint solvers, paired-end adapters, web interface) in a separate paragraph. The revised Discussion drops speculative roadmap items and instead keeps the focus on what PoolParty does now — design cards for surrogate modeling, the gap it fills — and acknowledges limitations (single-machine, not a sequence optimizer) without over-promising.

## 9. DMS section is more narrative

The old DMS section reads somewhat like a methods protocol. The revision integrates the code walkthrough into a narrative — defining the template, branching into four operations, merging, then inspecting design cards — making it easier to follow as a reader who hasn't used the tool.

## Overall pattern

The revision moves from a *specification-style* voice (here is what the system is, formally) to a *demonstration-style* voice (here is the problem, here is how you solve it). It trusts the reader less with abstractions and more with concrete examples.
