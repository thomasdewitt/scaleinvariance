---
name: No silent fallbacks
description: Never silently fall back to a different kernel or method - raise an error instead
type: feedback
---

Never implement silent fallbacks. If a requested configuration is incompatible (e.g., spectral kernel with custom scale_metric), raise a ValueError rather than silently switching to a different method.

**Why:** Thomas's CLAUDE.md explicitly says "NO FALLBACKS unless EXPLICITLY REQUESTED". Silent fallbacks hide incorrect behavior and make debugging harder.

**How to apply:** When a parameter combination is invalid, always raise with a clear error message. Let the user explicitly choose the alternative.
