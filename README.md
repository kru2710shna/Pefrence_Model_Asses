# Preference Model Assessment — KV-Cache RL Environment

An RL environment for training LLMs on a realistic ML engineering task: implement KV-caching in a transformer generation loop. The candidate must produce outputs that are **bitwise identical** to a naive reference implementation while running **at least 2× faster**.

The environment is dual-gate by design — correctness *and* speed must both be satisfied — which makes reward hacking nearly impossible. You can't pass correctness by copying the naive code (fails the speed gate), can't pass speed by hardcoding outputs (fails correctness on hidden prompts), can't pass either by returning early (fails the memory shape check).

---

## Why this task

Most RL-for-code environments either (a) check correctness only, which is reward-hackable by memorization, or (b) check a single metric, which is gameable by adversarial submissions that satisfy the metric without doing the underlying work. This environment is built around the observation that **two independent gates with different failure modes compose into a much stronger guarantee than either alone**.

KV-caching specifically was chosen because:

- It's a real task every inference engineer does. Realistic, not contrived.
- It has a clean correctness oracle (bitwise equality with the naive implementation). No fuzzy matching, no "close enough."
- It has a clean speed oracle (wall-clock ratio against the same naive implementation on the same hardware). No external benchmarks to drift.
- The two oracles can't be satisfied by the same shortcut. Each gate closes a different attack surface.
- It scales: the same dual-gate pattern generalizes to flash attention, paged attention, speculative decoding, quantization, and dozens of other inference optimizations.

---

## Repository layout

```