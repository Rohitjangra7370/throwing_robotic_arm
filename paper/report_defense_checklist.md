# AR525 MC-PILOT Report-Defense Checklist

This document is a defense checklist for presenting the project strongly while staying fully truthful about what was implemented, what was adapted, and what still remains limited.

## Core Positioning

Use this as the main story:

We recreated the core MC-PILOT idea in a pure Python ballistic simulator, starting from `MC-PILCO`, then extended it to study how the learned throwing policy behaves under different release heights. The main contribution is not just "it runs", but that we identified and fixed several non-obvious failure modes that were preventing learning, and we documented why certain design choices worked or failed.

That framing is both impressive and safe. It emphasizes:

- faithful algorithmic adaptation from `MC-PILCO` to a throwing task
- serious debugging and ablation work
- a meaningful geometry extension study
- honest boundaries on what was and was not implemented

## What You Can Claim Confidently

Say these clearly and directly:

1. You implemented the core MC-PILOT-style throwing setup on top of `MC-PILCO`:
   - augmented state with target position
   - single-shot policy `pi(P) -> speed`
   - GP model of free-flight ball dynamics
   - landing-based cost

2. You reproduced a strong simulation baseline in your own environment, rather than only reading the paper:
   - recreated the throwing environment
   - connected it to GP model learning and MC policy optimization
   - got successful learning after fixing real implementation issues

3. You did meaningful diagnosis, not blind tuning:
   - fixed broken gradient flow
   - fixed landing handling during GP rollout
   - identified unreachable-target issues
   - showed when exploration diversity matters
   - discovered that policy lengthscale resolution must match target-range width

4. Your extension already produced a useful scientific observation:
   - changing release height changes the effective geometry and difficulty of the learning problem
   - the same nominal policy settings do not transfer equally well across all geometries
   - narrow target domains can make the RBF policy effectively blind unless policy lengthscales are scaled appropriately

## What You Should NOT Claim

Avoid these unless you qualify them carefully:

1. Do not say you implemented "elevated targets" in the full original sense.
   What you actually implemented in `mc-pilot-elevated` is a release-height variation study with ground-plane targets still sampled as `(Px, Py)` on `z=0`.

2. Do not say the project is a full end-to-end reproduction of the paper.
   It is a strong simulation recreation/adaptation of the core algorithmic idea, but not the full real-system pipeline.

3. Do not say the current system optimizes release height and launch angle jointly.
   Right now release height is configured per experiment and launch angle is fixed.

4. Do not say every paper parameter was reproduced exactly.
   Some parameters were intentionally adapted for your simulator so learning would be physically meaningful and numerically stable.

5. Do not overclaim statistical certainty from a small number of seeds.
   Present the results as strong empirical evidence, not final proof of universal robustness.

## Best Truthful Framing

If someone asks "What exactly is your contribution?", use something close to this:

> We first recreated the core MC-PILOT simulation logic from `MC-PILCO` in a custom ballistic environment. Then we validated the baseline and performed a structured extension study where we changed the release height and analyzed how exploration strategy, reachable target range, and policy resolution affected learning. A major contribution of our work is the diagnosis of why certain settings fail, not just reporting which ones fail.

If someone asks "Did you implement the original elevated-target plan from your proposal?", say:

> Not fully. Our implemented extension is a narrower but still meaningful geometry generalization study: we varied release height while keeping targets on the ground plane. So the current extension studies how the same throwing framework adapts to different launcher geometries, rather than full 3D elevated targets.

That answer is honest and still sounds strong.

## Most Impressive Parts to Emphasize

These are the strongest parts of the project and should appear early in the defense:

1. The project moved beyond simple reproduction.
   You did not just port code. You identified why training originally failed and fixed multiple hidden issues.

2. The debugging was conceptually deep.
   The best examples are:
   - gradient path to policy parameters was broken
   - landing freeze was necessary because post-impact GP rollout produced nonsense states
   - target domain and reachable range had to be physically consistent
   - narrow target ranges required smaller policy lengthscales

3. The failure analysis is unusually strong.
   Your notes do a good job separating:
   - exploration failure
   - GP data coverage failure
   - cost-shaping failure
   - policy representation failure

4. Config E is a good showcase case study.
   Present it as the hardest geometry and your best example of scientific debugging:
   - standard settings failed repeatedly
   - multiple plausible hypotheses were tested
   - the final resolution came from identifying RBF blindness, not just changing random hyperparameters

## Main Limitations to Admit Early

Admitting these early makes the rest of the defense more credible.

1. The current extension is release-height variation, not full 3D elevated targets.

2. The current simulator is a custom ballistic simulator, not the paper's full robot plus delay-estimation setup.

3. Launch angle is fixed, so the learned policy only outputs release speed.

4. Experimental coverage is still limited compared with a full paper-level study:
   - not many seeds
   - not a full statistical sweep
   - no hardware validation

5. There is one notable implementation caveat still worth acknowledging:
   training-time particle landing in GP rollout is frozen after crossing the ground threshold, but not explicitly interpolated to the exact landing point the same way the real simulator does. If asked, present this as a known train-test mismatch and a reasonable next cleanup.

## Questions You Should Be Ready For

### 1. Why is this still a good project if the full original extension was not completed?

Suggested answer:

> Because the implemented part is still a nontrivial algorithmic extension and validation study. We recreated the core method, demonstrated successful learning, and produced real insights about failure modes, geometry dependence, exploration design, and policy resolution. So the contribution is smaller in scope than the original full 3D plan, but technically solid and well understood.

### 2. What is genuinely new in your work, beyond the paper?

Suggested answer:

> The main new part is the controlled study across different release heights in our simulator and the resulting analysis of what breaks across geometries. In particular, we found that narrow target domains can make the policy effectively constant unless the RBF lengthscales are matched to the target-range width.

### 3. Why didn't paper defaults work directly?

Suggested answer:

> Because algorithm settings interact with geometry and simulator details. Some values that are acceptable in the paper setup led to gradient saturation, unreachable targets, or poor policy resolution in ours. We treated that as a research question and analyzed which assumptions were geometry-dependent.

### 4. Did you just tune hyperparameters until it worked?

Suggested answer:

> No. We tested specific hypotheses and documented why each change helped or failed. For example, some failures were due to poor exploration coverage, some due to physically unreachable targets, and one important failure was due to the policy basis functions being too smooth to distinguish narrow target ranges.

### 5. Why keep launch angle fixed?

Suggested answer:

> We deliberately kept one degree of freedom fixed so we could first validate the core MC-PILOT adaptation and study generalization across target geometry without exploding the action space. It makes the current system narrower than the full proposal, but also makes the conclusions cleaner.

### 6. How is your work different from just using an analytical ballistic formula?

Suggested answer:

> The point is not only to compute one ideal trajectory. The GP model learns the dynamics from sampled throws, and the policy learns to generalize over a target distribution while accounting for model uncertainty. Also, our results show that learning behavior depends strongly on exploration data and policy representation, which a simple closed-form baseline does not address well.

## How to Present the "Elevated" Work Truthfully

Use this wording:

- "release-height generalization study"
- "varying platform height"
- "geometry extension under different launcher heights"
- "same MC-PILOT framework tested under different release geometries"

Avoid saying:

- "full elevated-target implementation"
- "3D target extension completed"
- "joint optimization of height, angle, and velocity"

## Recommended Slide/Report Structure

Use this order because it makes the work look systematic rather than messy.

1. Problem and paper context
   - what MC-PILOT is
   - why model-based RL is interesting for throwing

2. What you implemented
   - `MC-PILCO` to throwing adaptation
   - custom ballistic simulator
   - single-shot policy
   - GP model and landing cost

3. Why reproduction was nontrivial
   - broken gradients
   - wrong horizon bug
   - landing handling bug
   - cost saturation

4. Baseline results
   - successful learning on the recreated baseline

5. Release-height extension study
   - configs A-E
   - what transferred and what failed
   - why stratified exploration helped

6. Config E case study
   - repeated failure
   - false leads
   - actual root cause: policy resolution / lengthscale mismatch

7. Limitations and next steps
   - current elevated work is release-height variation, not full 3D targets
   - fixed launch angle
   - more seeds and cleaner evaluation still needed

That structure makes your project look rigorous and mature.

## Defense Checklist

Before the defense, make sure you can show or say all of the following:

- one clean diagram of the current implemented pipeline
- one slide that separates paper method, your baseline recreation, and your extension
- one table of configs A-E with release height, target range, exploration strategy, and headline result
- one slide on "major bugs/failure modes we discovered and fixed"
- one slide on "what did not work and why"
- one slide on Config E as a case study in diagnosis
- one limitations slide with truthful boundaries
- one short verbal explanation of why your extension is still valuable even though it is narrower than the original full plan

## One-Sentence Safe Summary

If you need a short high-quality summary during the defense, use this:

> We recreated the core MC-PILOT throwing framework in a custom simulator, validated baseline learning, and then used a release-height extension study to show that geometry, exploration coverage, and policy resolution critically affect data-efficient throwing performance.

## One-Sentence Limitation Statement

If you need a short honest limitation statement, use this:

> The current extension varies release geometry rather than implementing full 3D elevated targets, so our strongest conclusions are about geometry-dependent learning behavior under fixed-angle throwing, not the full target-height problem.

## Final Advice

The best way to sound impressive is not to pretend the scope is larger than it is.

What makes this project strong is:

- you actually built and validated a hard algorithmic adaptation
- you found real bugs and conceptual failure modes
- you documented negative results properly
- you can explain why the system works when it works, and why it fails when it fails

That combination is usually more convincing in a defense than claiming a bigger scope with a weaker understanding.
