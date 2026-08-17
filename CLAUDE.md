# CLAUDE.md - scaleinvariance

## Agent Skill

The agent skill file lives at `agent-skills/scaleinvariance/SKILL.md` in this repo. This is the source of truth.

**When publishing a new version**, always update the skill file to reflect any API changes, then install it locally:

```bash
cp agent-skills/scaleinvariance/SKILL.md ~/.claude/skills/scaleinvariance/
cp agent-skills/scaleinvariance/SKILL.md ~/.codex/skills/scaleinvariance/
```

Do both agents. The installed Claude copy sat at 0.11.0 through the 0.12.0 and
0.13.x work, i.e. codex and Claude were reviewing against a skill that predated
the wavelet framework and the N-D C1 fix.

### Development Guidelines

1. **Function Naming**: Use descriptive names like `structure_function_hurst()` rather than generic interfaces or abbreviations
2. **Dependencies**: Keep minimal - use numpy/scipy for core functionality, torch for FIF
3. **Testing**: Add tests in `tests/` directory as methods are implemented

### Coding Practices

1. **No unnecessary filler**: For unimplemented functions, just use `raise NotImplementedError` - no docstrings, comments, or placeholder code
2. **NO FALLBACKS unless EXPLICITELY REQUESTED**:

## Tests

Run functional tests from the repo root:

```bash
python -m pytest tests/functional/ -v
```

## Publishing

PyPI is the distribution channel; Zenodo is the archive. Do both.

Before tagging, bump the version in **all four** places — `pyproject.toml`,
`setup.py`, `scaleinvariance/__init__.py`, and the `Version **X.Y.Z**` line in
`agent-skills/scaleinvariance/SKILL.md` — and bump `version:`/`date-released:`
in `CITATION.cff`, which is what Zenodo reads for record metadata. `docs/conf.py`
reads `scaleinvariance.__version__`, so it needs no bump. Add the release's
`CHANGELOG.md` section at the same time; the release notes are sliced from it.

```bash
python -m build && twine upload dist/*
```

Then cut a GitHub release, which is what triggers the archive:

```bash
git tag -a vX.Y.Z -m "scaleinvariance X.Y.Z" && git push origin vX.Y.Z
```

```bash
gh release create vX.Y.Z --title "scaleinvariance vX.Y.Z" --notes-file <(sed -n "/## \[X.Y.Z\]/,/## \[/p" CHANGELOG.md | sed '$d')
```

The Zenodo–GitHub webhook then deposits a new version under the concept DOI
automatically — no manual upload. It fires on **releases**, not bare tags, so a
tag alone archives nothing. The toggle for `thomasdewitt/scaleinvariance` must be
on at https://zenodo.org/account/settings/github/ **before** the release is cut;
the OAuth grant can lapse silently, so re-check it each time.
