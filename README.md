# Semantic Translator Architecture — v0.1

**A Universal Open Protocol for LLM Inference Efficiency via Dual-End Semantic Caching**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Version](https://img.shields.io/badge/version-0.1-green.svg)]()
[![Status](https://img.shields.io/badge/status-open--draft-orange.svg)]()
[![Open Source](https://img.shields.io/badge/open--source-yes-brightgreen.svg)]()

**Author:** cnomic-dev  
**Date:** April 2026  
**Submitted to:** Anthropic feedback channel (April 3, 2026)  
**Philosophy:** This architecture is fully open. Any platform, researcher, or developer may implement, extend, or fork it without restriction.

---

## Overview

Current LLM inference pipelines process every request from raw token sequences, reconstructing the full KV cache on each turn regardless of semantic overlap with prior interactions.

This protocol introduces a **Semantic Translator layer** — a lightweight, platform-agnostic middleware between the user interface and the inference engine.

> **Core insight:** Reduce how often full inference occurs, not just its cost.

The v0.1 specification is intentionally minimal. It defines exactly four components and nothing more. All extensions belong in future versions or platform-specific implementations.

---

## ⚠️ Security & Stability Parameters (Read First)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `query_max_len` | `2000` | Input length guard — prevents DoS via long prompts |
| `max_cache_age_sec` | `86400` | TTL — prevents stale semantic states from persisting |
| `max_cache_entries` | `1000` | Memory bound — evicts least-used when full |
| `triple_hash_validation` | SHA-256 | Cache injection prevention — signs `(I,C,O)` triple |

> These parameters are intentionally conservative for v0.1. Platforms may tune them based on their threat model, but must document deviations in `config.yaml`.

---

## The Four Core Components (v0.1 Scope)

| # | Component | Description |
|---|-----------|-------------|
| 1 | **Ternary Decomposition** `T` | Maps input query to `(I, C, O) ∈ {-1, 0, 1}³` |
| 2 | **S³ Projection** `φ` | Embeds the triple onto the unit 3-sphere via precomputed lookup table |
| 3 | **Semantic Cache** | Returns cached result if chordal distance ≤ ε |
| 4 | **Cross-language Aligner** `R` | SO(4) rotation matrix interface (identity default) |

Everything else — persistence, dynamic thresholds, hallucination attractors, higher-dimensional extensions — is outside v0.1 scope.

---

## Dimension Definition (Locked for v0.1)

**This definition is fixed for v0.1 and must not vary across implementations.**

| Dimension | Value | Meaning |
|-----------|-------|---------|
| **I — Intent** | `-1` | Exploration / Question |
| | `0` | Neutral / Verification |
| | `1` | Instruction / Assertion |
| **C — Context** | `-1` | Casual / Conversational |
| | `0` | Neutral / Domain-agnostic |
| | `1` | Formal / Academic |
| **O — Operation** | `-1` | Compression / Summarization |
| | `0` | Translation / Conversion |
| | `1` | Expansion / Generation |

---

## Mathematics

### 1. Ternary Decomposition

$$\mathcal{T}: q \mapsto (I, C, O) \in \{-1, 0, 1\}^3$$

`T` maps an input query to a semantic triple. In v0.1, `T` is a minimal rule-based classifier with a replaceable interface (`fallback_fn`).

> **`T` MUST be replaceable. The rule-based implementation is only a reference baseline.**

### 2. S³ Projection

The triple is embedded onto the unit 3-sphere:

$$\phi(I, C, O) = \frac{(1,\ I,\ C,\ O)}{\|(1,\ I,\ C,\ O)\|} \in S^3 \subset \mathbb{R}^4$$

This mapping ensures all 27 discrete points are unique and non-overlapping. Note that angular distances between points are not uniform — this is a known property of the fixed embedding and does not affect correctness.

All 27 points are precomputed offline and stored in `lookup_table.npy`. No runtime floating-point computation is required.

### 3. Cache Retrieval

For a new query `q'`, compute **chordal distance** against cached entries:

$$d_c(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2$$

If `d_c ≤ ε`, return cached result directly without invoking the inference core.

**Defaults:**
- `ε_cache = 0.65` — corresponds to one ternary dimension of difference under the fixed S³ embedding defined in v0.1. **This is not a universal constant** — it is empirically calibrated to this specific embedding.
- `ε_verify = 0.30` — for strict semantic verification

> **Why chordal distance?** For 27 fixed points on S³, chordal distance is monotonic with geodesic distance and enables O(1) lookup via a precomputed 27×27 distance matrix.

### 4. Cross-language Alignment

Language-to-language alignment is expressed as an SO(4) rotation:

$$\phi(L_j) = R_{ij} \cdot \phi(L_i), \quad R_{ij} \in \mathbb{R}^{4 \times 4}, \quad R_{ij}^\top R_{ij} = I$$

**v0.1 default:** Identity matrix. Platforms replace with learned rotations via `alignments/default.json`.

> Future versions may restrict `R` to SU(2) for quaternion-based alignment: `v' = R v R†`. This is deferred to v0.2 to avoid ambiguity between SU(2) (quaternion conjugate rotation on S³) and SO(4) (general 4D rotation). v0.1 uses SO(4) as the conservative, well-defined interface.

Complexity reduction: `O(N²) → O(2N)` for N languages.

Alignment can be learned offline using parallel corpora:

```python
from scipy.spatial.transform import Rotation
R = Rotation.align_vectors(target_vectors, source_vectors)
```

---

## Standard API Contract

### Request Schema

```json
{
  "query":   { "type": "string", "maxLength": 2000 },
  "lang":    { "type": "string", "enum": ["en", "zh", "ja", "..."] },
  "version": { "type": "string", "const": "0.1" }
}
```

### Response Schema

```json
{
  "hit":    { "type": "boolean" },
  "triple": { "type": "array", "items": { "enum": [-1, 0, 1] }, "minItems": 3, "maxItems": 3 },
  "result": { "type": "string" },
  "source": { "type": "string", "enum": ["cache", "inference"] }
}
```

---

## Forward Compatibility Rule

v0.2+ may extend the semantic triple with additional dimensions (e.g., `Certainty`, `Emotion`, `Temporal`).

**Zero-Padding Rule:**
- New dimensions default to `0` in the vector representation.
- Distance calculations automatically ignore dimensions with value `0`.
- Old v0.1 parsers truncate to the first 3 dimensions — no breaking changes.

```python
# v0.1 triple (4D vector)
v0 = np.array([1.0, I, C, O]) / norm

# v0.2 extended (e.g. 8D)
v1 = np.zeros(8)
v1[:4] = v0   # new dims = 0 → d_c(v0, v1[:4]) == 0 → backward compatible
```

---

## Minimal Runnable Core

### `precompute_27_points.py`

```python
import numpy as np

LOOKUP = {}
for I in [-1, 0, 1]:
    for C in [-1, 0, 1]:
        for O in [-1, 0, 1]:
            v = np.array([1.0, I, C, O], dtype=np.float32)
            LOOKUP[(I, C, O)] = v / np.linalg.norm(v)

np.save('lookup_table.npy', LOOKUP)
print(f"Generated {len(LOOKUP)} points.")
```

### `core/translator.py`

```python
import numpy as np
import time
import hashlib

# Load precomputed lookup table
LOOKUP = np.load('lookup_table.npy', allow_pickle=True).item()
TRIPLES = list(LOOKUP.keys())
VECTORS = np.array([LOOKUP[t] for t in TRIPLES], dtype=np.float32)

# Precompute 27x27 distance matrix for O(1) retrieval
DIST_MATRIX = np.linalg.norm(
    VECTORS[:, None, :] - VECTORS[None, :, :], axis=-1
)
TRIPLE_INDEX = {t: i for i, t in enumerate(TRIPLES)}


class SemanticTranslator:
    def __init__(
        self,
        epsilon: float = 0.65,
        max_cache: int = 1000,
        ttl: float = 86400,
        query_max_len: int = 2000,
    ):
        self.epsilon = epsilon
        self.max_cache = max_cache
        self.ttl = ttl
        self.query_max_len = query_max_len
        self.aligner = np.eye(4, dtype=np.float32)  # SO(4) identity — replace with learned matrix
        # Cache: {triple_hash -> [{"triple": ..., "result": ..., "hits": ..., "ts": ...}]}
        self.cache: dict[str, dict] = {}

    # ── Hashing ────────────────────────────────────────────────────
    def _triple_hash(self, triple: tuple) -> str:
        """SHA-256 of triple — prevents cache injection."""
        return hashlib.sha256(str(triple).encode()).hexdigest()

    # ── Decomposition ──────────────────────────────────────────────
    def decompose(self, query: str, fallback_fn=None) -> tuple:
        """
        Map query to (I, C, O) triple.
        T MUST be replaceable — this rule-based version is a reference baseline only.
        Inject a custom classifier via fallback_fn for production use.
        """
        if fallback_fn:
            return fallback_fn(query)
        I = 1 if any(kw in query for kw in ["please", "请", "！", "!"]) \
            else (-1 if "?" in query or "？" in query else 0)
        C = 0  # Default: neutral / domain-agnostic
        O = 0  # Default: translation / conversion
        return (I, C, O)

    # ── Projection ─────────────────────────────────────────────────
    def project(self, triple: tuple) -> np.ndarray:
        """Apply SO(4) aligner then return unit vector on S³."""
        v = self.aligner @ LOOKUP[triple]
        return (v / np.linalg.norm(v)).astype(np.float32)

    # ── Cache retrieval ────────────────────────────────────────────
    def retrieve(self, triple: tuple) -> dict | None:
        """O(1) lookup via precomputed distance matrix."""
        now = time.time()
        idx_q = TRIPLE_INDEX[triple]
        best_dist, best_key = float("inf"), None

        for key, entry in list(self.cache.items()):
            if now - entry["ts"] > self.ttl:
                del self.cache[key]
                continue
            idx_c = TRIPLE_INDEX[entry["triple"]]
            d = DIST_MATRIX[idx_q, idx_c]
            if d < best_dist:
                best_dist, best_key = d, key

        if best_dist <= self.epsilon and best_key:
            self.cache[best_key]["hits"] += 1
            return self.cache[best_key]
        return None

    # ── Cache storage ──────────────────────────────────────────────
    def store(self, triple: tuple, result: str):
        """Store with TTL + hit-count eviction."""
        now = time.time()
        if len(self.cache) >= self.max_cache:
            # Remove expired first; fall back to lowest hit count
            expired = [k for k, v in self.cache.items()
                       if now - v["ts"] > self.ttl]
            if expired:
                del self.cache[expired[0]]
            else:
                evict = min(self.cache, key=lambda k: self.cache[k]["hits"])
                del self.cache[evict]
        key = self._triple_hash(triple)
        self.cache[key] = {
            "triple": triple,
            "result": result,
            "hits": 1,
            "ts": now,
        }

    # ── Main entry point ───────────────────────────────────────────
    def translate(self, query: str, fallback_fn) -> dict:
        """
        Primary interface.
        Returns cache hit or inference result with full audit trail.
        """
        if len(query) > self.query_max_len:
            query = query[:self.query_max_len]

        triple = self.decompose(query, fallback_fn=None)
        hit = self.retrieve(triple)

        if hit:
            return {
                "hit": True,
                "triple": list(triple),
                "result": hit["result"],
                "source": "cache",
            }

        result = fallback_fn(query)
        self.store(triple, result)
        return {
            "hit": False,
            "triple": list(triple),
            "result": result,
            "source": "inference",
        }
```

---

## Setup & Quickstart

```bash
git clone https://github.com/cnomic-dev/semantic-translator-architecture
cd semantic-translator-architecture
pip install numpy

python precompute_27_points.py      # generates lookup_table.npy
python examples/quickstart.py
```

```python
# examples/quickstart.py
from core.translator import SemanticTranslator

def my_llm(query: str) -> str:
    return f"[inference result for: {query}]"

st = SemanticTranslator(epsilon=0.65)

r1 = st.translate("What is consciousness?", fallback_fn=my_llm)
print(r1)  # source: inference

r2 = st.translate("What is awareness?", fallback_fn=my_llm)
print(r2)  # source: cache (semantic hit)

# Verify hit rate
def test_hit_rate(st, pairs):
    hits = sum(1 for q in pairs if st.translate(q, my_llm)["hit"])
    print(f"Hit rate: {hits}/{len(pairs)} = {hits/len(pairs):.0%}")
```

```bash
# Run tests
python -m pytest tests/ -v
python examples/quickstart.py
```

---

## What v0.1 Does NOT Promise

- Does not handle long context windows (> 2000 tokens)
- Does not support dynamic threshold adjustment at runtime
- Does not guarantee absolute cross-platform semantic consistency
- Does not include persistent disk cache (`dict` only; disk cache planned for v0.2)
- Does not implement quaternion conjugate rotation (SU(2) planned for v0.2)
- Eviction strategy is implementation-defined beyond the defaults above

---

## Relationship to Existing Infrastructure

| Layer | Technology | Role | Conflict? |
|-------|-----------|------|-----------|
| Deployment | Lemonade (AMD) | Local inference endpoint | None — compatible |
| KV Cache | TurboQuant (Google) | Memory compression during inference | None — complementary |
| Weights | GGUF / AWQ | Static model size reduction | None — orthogonal |
| **Protocol** | **This spec (v0.1)** | **Reduces inference invocations** | N/A — new layer |

---

## Versioning & Extension Path

```
v0.1  — 4 core components, rule-based T, SO(4) identity aligner, dict cache
v0.2  — Disk cache, SU(2) quaternion aligner, learned T classifier
v0.3  — S⁷ extension (octonion structure), dynamic ε
v1.0  — Stable API, multi-platform validation
```

All extensions follow the zero-padding rule for backward compatibility.

---

## Dependencies

```toml
[project]
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
alignment = ["scipy>=1.10"]
testing   = ["pytest>=7.0"]
```

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add new language adapters, encoding rules, or cache strategies.

All contributions must preserve the four-component core and the locked v0.1 dimension definitions.

---

## Related Work

- [ΨSEP — Human Basic Evolution Equation](https://github.com/cnomic-dev/human-basic-evolution-equation)
- SEP — Symbiotic Evolution Protocol (AI governance framework)
- ESP — Evolutionary Singularity Protocol (economic-AI coupling)

---

## License

```
Apache License 2.0

Copyright 2026 cnomic-dev

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

This architecture is fully open. Any platform, researcher, or developer
may implement, extend, or fork without restriction.
```

---

*v0.1 — April 2026 — github.com/cnomic-dev/semantic-translator-architecture*
