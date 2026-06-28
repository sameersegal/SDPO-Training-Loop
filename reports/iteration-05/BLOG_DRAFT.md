# Some bugs have no line number — the blind spot in self-improving code models

*A research note on what self-distillation can and can't teach a model to fix.*

---

There's a clean idea gaining traction for getting LLMs to improve themselves: let the model
attempt a problem, then **feed back a critique aimed at the exact step it got wrong**, and have the
model learn from its own corrected attempt. No human labels, no reference solutions — the model
teaches itself, and the feedback just points the flashlight. On math, "aim the feedback at the wrong
step" beats handing over a full reference solution by a wide margin.

We're applying this to **competitive programming**, and we hit something the math results can't show
you.

**The mechanism, in one line.** Self-distillation's entire learning signal is the *per-token
advantage*: how much seeing the feedback shifts the model's mind about each token it wrote. The model
only learns where that shift is **real and localized** — concentrated on the tokens that were
actually wrong. Diffuse the signal across the whole answer and you teach nothing (or worse, you punish
the parts it got right).

> 📊 **[GRAPH 1 — the per-token signal]** advantage along a failed attempt: a sharp negative spike at
> the wrong region (good, learnable) vs. a flat smear across the whole rollout (no signal).

**The blind spot.** Aiming the feedback assumes the error *has* a location. In competitive
programming, the most common way a hard problem is failed isn't a wrong line — it's a **timeout**. The
code is *correct*; it's just too slow. The algorithm is O(n²) when the input size demands O(n log n).
There is no buggy token to point at, no step to rewrite — **the entire approach is wrong, globally.**

Per-token credit assignment has nothing to grab onto. We measured it directly: on these
timeout-dominated hard problems, the feedback signal was **~20× weaker** than on a localizable bug,
and what little there was landed on the model's *prose*, not its code.

> 📊 **[GRAPH 2 — error geometry]** WA (wrong answer, a *local* error) produces a localizable signal;
> TLE (timeout, a *global* error) produces a flat, near-zero one — same model, same method.

**The reframe.** The lesson everyone's taking from the math results is *"aim the feedback better."*
We think the real lesson is **the feedback has to match the *geometry* of the error**:

| Error | Where it lives | Feedback that teaches |
|---|---|---|
| Wrong answer / off-by-one | a **local** step | localize it: "your code is right through line N; the bug is after" |
| Timeout (too slow) | the **global** approach | re-architect it: "this is O(n²); n≤1e6 needs O(n log n)" |

A single recipe — "anchor the correct prefix, rewrite the wrong step" — only covers the top row. The
bottom row, an entire class of real engineering failures, needs a *different kind* of feedback
altogether. **Localization is the answer only when the error is local.**

**A second trap, worth stating plainly.** A *more capable* teacher doesn't help if it doesn't fix
*this attempt*. Hand the teacher a full reference solution and it writes a correct answer — in its own
style — that disagrees with the student even where the student was right. The result is a signal that
fights the correct parts of the rollout. **Capability to generate ≠ ability to correct.** The
precondition for self-improvement isn't "is the teacher good," it's "does the feedback resolve *this
model's specific gap*, in a form it can localize."

**What we're building toward.** If self-distillation only teaches what the feedback can localize, then
*before* spending GPU hours you should be able to **measure whether a usable signal even exists** — a
fuel gauge for self-improvement. We gate every training run on exactly that: does the feedback make
the model assign higher probability to the right tokens, and is that shift localized? If it's faint or
diffuse, we don't train — we fix the feedback, or we report the null.

> 📊 **[GRAPH 3 — the fuel gauge]** signal strength × localization across difficulty, with the
> go/no-go line that decides whether a run is worth running.

**Takeaway.** Self-improving models are only as good as the geometry of their feedback. The exciting
recipe — point the model at its mistake — quietly assumes every mistake is a point. The most
expensive failures aren't points; they're the whole approach. Teaching a model to fix *those* is a
different problem, and it's the one we think matters most for real code.

*Part of an open research log — building [SparkyCoder]: post-training an 8B model to solve
competitive-programming problems by teaching itself.*
