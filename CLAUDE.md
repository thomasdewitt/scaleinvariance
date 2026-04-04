# CLAUDE.md - scaleinvariance

## Agent Skill

The agent skill file lives at `agent-skills/scaleinvariance/SKILL.md` in this repo. This is the source of truth.

**When publishing a new version**, always update the skill file to reflect any API changes, then install it locally:

```bash
cp agent-skills/scaleinvariance/SKILL.md ~/.claude/skills/scaleinvariance/
```

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

```bash
python -m build && twine upload dist/*
```
