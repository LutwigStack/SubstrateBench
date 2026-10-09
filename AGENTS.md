# Canonical SubstrateBench

This repository owns reusable benchmark code, public contracts, documentation,
and focused tests. Exploratory experiments and their evidence belong in
`LutwigStack/substratebench-lab`.

- Promote the smallest complete rewrite from reviewed evidence. Record its
  bounded scope and limitations in a document such as
  `docs/promotion/SB-001.md` before adding code.
- Keep the package installable from a fresh checkout and usable from any working
  directory. Do not import lab modules or use machine-specific absolute paths.
- Do not commit raw runs, checkpoints, binary arrays, credentials, generated
  results, virtual environments, or editor state.
- Keep reference/oracle results separate from learned-source evidence. State
  caller contracts, measured cost boundaries, failed qualification, and
  unverified generalization honestly.
- Run focused tests and an installed-package CLI/API check. Obtain independent
  code and methodology review before committing a promotion.
- Prefer Python's standard library and small concrete owners. Add dependencies
  or abstraction only for a demonstrated public use.

The original research vision is a specification, not evidence that its planned
components exist. No licensing terms have been selected for this initial import.
