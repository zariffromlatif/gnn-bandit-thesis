# Publication Roadmap: GNN-Bandit Framework
## How to Target A* Conferences and Q1 Journals

**Paper Title:** Graph-Enhanced Causal Reinforcement Learning for Proactive Customer Retention  
**Subtitle:** A Prescriptive Framework using Off-Policy Evaluation on the Open Bandit Dataset

---

## PART 1: TARGET VENUE SELECTION

### A* Conferences (CORE Ranking A*)

These are the most prestigious venues. Acceptance rates are 15–25%. Your work sits at the intersection of RL, GNNs, and recommender systems.

| Conference | Full Name | Deadline (Typical) | Fit Score | Notes |
|---|---|---|---|---|
| **KDD** | ACM Knowledge Discovery & Data Mining | Feb | ⭐⭐⭐⭐⭐ | Best fit — covers bandits, OPE, graphs, retention |
| **WWW** | The Web Conference | Oct | ⭐⭐⭐⭐⭐ | Strong recommender + causal track |
| **NeurIPS** | Neural Information Processing Systems | May | ⭐⭐⭐ | Very competitive; needs strong theory |
| **AAAI** | AAAI Conference on AI | Aug | ⭐⭐⭐⭐ | Applied AI track — good for your framing |
| **SIGIR** | ACM SIGIR (Information Retrieval) | Jan | ⭐⭐⭐⭐ | Strong OPE + recommendation research |
| **ICLR** | International Conference on Learning Representations | Oct | ⭐⭐ | Needs very strong theory/novelty |

**Primary Recommendation: KDD or WWW first.**  
Your GNN + bandit + OPE combination maps perfectly to their applied data science and causal inference tracks.

---

### Q1 Journals (Scimago/JCR Q1)

| Journal | Publisher | Impact Factor | Fit | Turnaround |
|---|---|---|---|---|
| **Expert Systems with Applications** | Elsevier | ~8.5 | ⭐⭐⭐⭐⭐ | 3–6 months |
| **IEEE Trans. on Neural Networks & Learning Systems** | IEEE | ~14.3 | ⭐⭐⭐⭐ | 6–12 months |
| **Knowledge-Based Systems** | Elsevier | ~8.1 | ⭐⭐⭐⭐⭐ | 3–5 months |
| **Information Sciences** | Elsevier | ~8.1 | ⭐⭐⭐⭐ | 4–6 months |
| **IEEE Trans. on Knowledge and Data Engineering** | IEEE | ~8.9 | ⭐⭐⭐ | 8–14 months |
| **ACM Trans. on Recommender Systems** | ACM | New journal | ⭐⭐⭐⭐⭐ | 4–6 months |
| **Neurocomputing** | Elsevier | ~6.0 | ⭐⭐⭐ | 3–5 months |

**Primary Recommendation: Expert Systems with Applications or Knowledge-Based Systems.**  
Both are Q1, respect applied ML contributions, and have faster turnaround than IEEE journals.

---

## PART 2: HONEST GAP ANALYSIS

This is the most important section. What your current paper has versus what Q1/A* reviewers will demand.

### What You Have (Strengths)

- Novel combination: LightGCN + BCQ + OPE in a unified retention framework
- Correct problem framing: churn prediction → prescriptive intervention
- Principled dataset choice: OBD is a real, peer-reviewed bandit dataset
- Sound methodology: IPW and DR estimators are the right tools

---

### Critical Gaps That Will Cause Rejection

#### Gap 1: Dataset Scale — HIGH SEVERITY

The preprocessing pipeline revealed only **481 unique users and 80 items**. This is a fatal weakness for most Q1 venues. Reviewers will immediately flag this.

The OBD "All" campaign has 13.7M rows but only 481 distinct user profiles because the hashed features repeat heavily. This is a fundamental limitation.

**Fixes:**
- Use all three campaigns (All + Men + Women) and treat them as separate user populations — tripling of scope is legitimate
- Report results per-campaign separately (shows robustness)
- Supplement with a synthetic dataset generated from OBD's distributions with a larger user count (must be clearly disclosed)
- Cite that OBD's user features are anonymized hash strings — frame this as a privacy-preserving real-world constraint

---

#### Gap 2: No Baseline Comparisons — HIGH SEVERITY

You cannot publish with only your method's results. Q1 reviewers require at least 4–6 baselines.

Required baselines:

| Baseline | What It Tests |
|---|---|
| Random Policy | Proves any structured method beats random |
| Thompson Sampling (BTS) | The existing logging policy — you must beat this |
| Standard DQN (no constraint) | Shows BCQ constraint is necessary |
| Matrix Factorization + Bandit | GNN vs. simpler embedding |
| LightGCN + Greedy (no bandit) | Proves RL component adds value |
| Uplift-only (no GNN) | Proves graph component adds value |

---

#### Gap 3: No Ablation Study — HIGH SEVERITY

An ablation study removes one component at a time and measures the drop in performance. This is mandatory for every ML paper at A* and Q1 level.

Required ablations:

```
Full GNN-Bandit          → your main result
GNN-Bandit (no graph)    → replaces LightGCN with MF embeddings
GNN-Bandit (no causal)   → replaces BCQ with vanilla DQN
GNN-Bandit (no OPE)      → evaluates with direct method only
GNN-Bandit (IPW only)    → doubly robust vs. IPW-only
```

---

#### Gap 4: No Statistical Significance Testing — HIGH SEVERITY

Single-run results are not accepted. You need:
- Multiple random seeds (at least 5)
- Confidence intervals (95%) on all metrics
- Paired t-test or Wilcoxon test comparing your method to baselines
- Report mean ± standard deviation for every metric

---

#### Gap 5: Theoretical Contribution is Weak — MEDIUM SEVERITY

For NeurIPS/ICML/ICLR this is fatal. For KDD/WWW/Expert Systems it is acceptable if your empirical contribution is very strong. At minimum, include:
- Formal problem definition (MDP/bandit formulation in mathematical notation)
- Proof that DR estimator is doubly robust in your setting
- Convergence guarantee or regret bound citation from BCQ paper applied to your setting

---

#### Gap 6: Cold-Start Analysis is Missing — MEDIUM SEVERITY

You found 205 cold-start users (42.6%). This is your strongest selling point. You need a dedicated experiment showing:
- Performance gap: GNN-Bandit vs. baselines specifically for cold-start users
- Embedding quality visualization (t-SNE plot of LightGCN embeddings)
- Performance as a function of graph connectivity (degree)

---

#### Gap 7: Framing Mismatch — LOW-MEDIUM SEVERITY

OBD is a fashion recommendation dataset, not a customer retention/churn dataset. Reviewers will notice this. You must:
- Explicitly define your "retention" proxy: a user is "retained" if they click (engagement = retention signal)
- Justify why click-through rate is a valid retention surrogate in e-commerce
- Either rename the problem to "engagement optimization" or add a section defending the retention interpretation
- Cite papers that use CTR as a retention proxy (there are many in RecSys literature)

---

## PART 3: REQUIRED EXPERIMENTAL ADDITIONS

### Metrics to Report

| Metric | Definition | Why Reviewers Expect It |
|---|---|---|
| **IPS (Inverse Propensity Score)** | Unbiased reward estimate | Standard OPE metric |
| **DR Estimate** | Doubly robust reward | Your main metric |
| **SNIPW** | Self-normalized IPW | More stable than raw IPW |
| **DM (Direct Method)** | Model-based estimate | Comparison baseline for OPE |
| **Relative Lift (%)** | (GNN-Bandit - BTS) / BTS | Business-interpretable |
| **Cold-Start Lift** | Reward for degree-0 users only | Your key differentiator |

---

### Experiment Structure

#### Experiment 1: Main OPE Comparison
- Compare all methods on DR estimated reward
- Report across all three campaigns (All, Men, Women)
- Table format with mean ± std across 5 seeds

#### Experiment 2: Ablation Study
- Remove each component, report DR reward degradation
- Bar chart showing contribution of each module

#### Experiment 3: Cold-Start Performance
- Filter test set to users with degree 0 (205 users)
- Compare GNN-Bandit vs. non-graph baselines
- This is your showpiece result — GNN should win here clearly

#### Experiment 4: Embedding Analysis
- t-SNE visualization of LightGCN embeddings
- Color by campaign (Men/Women) or by click rate
- Shows the graph is learning meaningful structure

#### Experiment 5: Sensitivity Analysis
- Vary LightGCN embedding dimension (16, 32, 64, 128)
- Vary number of GNN layers (1, 2, 3, 4)
- Vary BCQ batch constraint threshold
- Line plots showing robustness

#### Experiment 6: OPE Estimator Comparison
- Compare IPW, SNIPW, DR, DM on your policy
- Shows your awareness of OPE literature and that DR is the right choice

---

## PART 4: PAPER STRUCTURE FOR Q1 SUBMISSION

Q1 journals expect 10–15 pages (double column) or 8,000–12,000 words.

```
1. Introduction (1.5 pages)
   - Hook: the cost of customer churn
   - Problem: churn prediction ≠ retention optimization
   - Gap: cold-start problem in intervention selection
   - Contributions: numbered list (4–5 bullet points)
   - Paper organization

2. Related Work (1.5 pages)
   - Customer churn prediction (classic ML)
   - Recommender systems & GNNs (LightGCN, NGCF)
   - Contextual bandits for marketing
   - Off-policy evaluation methods
   - Causal inference in ML
   → Show how YOUR work differs from each

3. Problem Formulation (0.5 pages)
   - Formal bandit/MDP definition
   - Context, action, reward space
   - OPE objective function

4. Methodology (3 pages)
   4.1 Graph Construction
   4.2 LightGCN Encoder
   4.3 BCQ Policy Learning
   4.4 OPE with DR Estimator
   4.5 Full Algorithm (pseudocode box)

5. Experimental Setup (1 page)
   - Dataset description (OBD stats)
   - Baselines
   - Implementation details (hyperparameters)
   - Evaluation metrics

6. Results & Analysis (3 pages)
   6.1 Main results table
   6.2 Ablation study
   6.3 Cold-start analysis
   6.4 Embedding visualization
   6.5 Sensitivity analysis

7. Discussion (0.5 pages)
   - Why the GNN helps
   - Limitations (honest: dataset scale, simulation vs. live)
   - Ethical considerations

8. Conclusion (0.5 pages)

References (1–1.5 pages)
   → Aim for 40–60 references
```

### Contributions Section Template (Critical)

Reviewers read this first. It must be specific and falsifiable. Write it as:

> We make the following **four contributions**:
> 1. We formalize customer retention as a causal contextual bandit problem and propose the GNN-Bandit framework — the first integration of LightGCN graph embeddings with BCQ off-policy learning for prescriptive retention.
> 2. We demonstrate that LightGCN embeddings improve policy performance by **X%** for cold-start users (42.6% of the population) compared to non-graph baselines.
> 3. We conduct a rigorous off-policy evaluation using Doubly Robust estimators on the Open Bandit Dataset, showing our policy achieves **X%** lift over the Thompson Sampling baseline without live deployment.
> 4. We provide an open-source preprocessing pipeline converting OBD into a retention-framed bandit environment, enabling reproducible research.

*Fill in X% after running experiments.*

---

## PART 5: RELATED WORK YOU MUST CITE

### GNNs for Recommendation
- He et al. (2020) — **LightGCN** (SIGIR 2020) — your GNN backbone
- Wang et al. (2019) — **NGCF** — predecessor to LightGCN
- Ying et al. (2018) — **PinSage** — industrial-scale GNN recommendation

### Bandits & Off-Policy Learning
- Fujimoto et al. (2019) — **BCQ** (ICML 2019) — your RL backbone
- Saito et al. (2020) — **Open Bandit Dataset** — your dataset paper
- Dudik et al. (2011) — **Doubly Robust OPE** — your evaluator
- Precup et al. (2000) — Eligibility traces for off-policy evaluation
- Lattimore & Szepesvári (2020) — Bandit Algorithms (textbook)

### Causal Inference in Marketing
- Radcliffe & Surry (2011) — Real-world uplift modeling
- Gutierrez & Gérardy (2017) — Causal inference survey for data scientists
- Athey & Imbens (2017) — Recursive partitioning for heterogeneous treatment effects

### Customer Churn / Retention
- Hadden et al. (2007) — Churn prediction literature review
- Verbeke et al. (2012) — Profit-driven churn prediction

---

## PART 6: SUBMISSION STRATEGY & TIMELINE

### Recommended 12-Month Plan

```
Month 1–2 :  Run all experiments, collect results with 5 seeds
Month 3   :  Write full paper draft (all sections)
Month 4   :  Internal review with supervisor + revision
Month 5   :  Submit to KDD (Feb deadline) OR Expert Systems with Applications

--- IF REJECTED ---
Week 1    :  Read all reviewer comments carefully
Week 2–5  :  Address every reviewer comment (write a response letter)
Week 6    :  Resubmit to next target venue

--- IF ACCEPTED ---
Week 1–3  :  Camera-ready preparation
```

### Journal vs. Conference: Which First?

**If you want faster publication:**  
Submit to **Expert Systems with Applications** (Q1) first — 3–6 month turnaround, no rebuttal phase, revisions are iterative.

**If you want prestige on your CV:**  
Submit to **KDD or WWW** first. Even a rejection from these with good reviews strengthens your journal submission.

**Recommended for a thesis student:**  
Journal first (Expert Systems or Knowledge-Based Systems). Conference deadlines are rigid; journals allow rolling submission and iterative revision.

### Rejection is Normal

- First-submission acceptance rate at A* is under 25%
- Treat reviewer comments as free expert feedback
- A paper rejected from KDD with constructive reviews is routinely accepted at WWW or Expert Systems after revision
- Never abandon a paper after one rejection

---

## PART 7: COMMON REJECTION REASONS TO AVOID

| Reviewer Complaint | How to Preempt It |
|---|---|
| "Dataset is too small" | Use all 3 campaigns; report per-campaign; add synthetic supplement |
| "No baselines" | Include all 6 baselines listed in Part 3 |
| "No significance testing" | 5 seeds, 95% CI, t-tests — on every result table |
| "Framing mismatch (recommendation ≠ retention)" | Explicit justified mapping in Section 3 |
| "Incremental — just LightGCN + BCQ glued together" | Frame the OPE evaluation framework as the novel contribution, not just the architecture |
| "No theoretical analysis" | Add formal regret bound from BCQ applied to your setting |
| "Reproducibility concerns" | Release code on GitHub before submission |

---

## PART 8: CODE RELEASE (MANDATORY FOR Q1)

Most Q1 venues now require or strongly encourage code release. Before submitting:

1. Create a public GitHub repository with your full pipeline
2. Include: preprocessing script, GNN training code, BCQ agent, OPE evaluator
3. Add a `README.md` with exact reproduction steps and environment requirements
4. Tag the repository commit that corresponds to the submitted paper version
5. Link the repository in your paper's abstract or footnote

This alone can swing a borderline review from reject to accept.

---

## PART 9: PRE-SUBMISSION CHECKLIST

Go through every item before hitting submit:

### Experiments
- [ ] All 6 baselines implemented and compared
- [ ] Full ablation study (minimum 4 ablations)
- [ ] 5 random seeds with mean ± std reported on every table
- [ ] Statistical significance test (t-test or Wilcoxon) vs. best baseline
- [ ] Cold-start analysis as a dedicated experiment section
- [ ] t-SNE embedding visualization included
- [ ] Sensitivity analysis on LightGCN layers, embedding dim, BCQ threshold

### Writing
- [ ] Formal problem definition in mathematical notation (Section 3)
- [ ] Contributions list is specific with numbers (not vague claims)
- [ ] "Retention" framing explicitly justified with citations
- [ ] Related work covers all 5 areas listed in Part 5
- [ ] Discussion section honestly addresses limitations
- [ ] 40–60 references, all key prior work included

### Logistics
- [ ] GitHub repository created and linked in the paper
- [ ] Paper reviewed and approved by your thesis supervisor
- [ ] Full grammar check (Grammarly, LanguageTool, or similar)
- [ ] Paper formatted to target journal/conference template
- [ ] All figures are high resolution (300 DPI minimum)
- [ ] Author list and affiliations confirmed

---

## KEY NUMBERS FROM YOUR DATASET (Reference)

From the preprocessing pipeline output:

| Statistic | Value |
|---|---|
| Total rows (All campaigns) | ~13.7M (random) + ~12.4M (bts) |
| Unique users | 481 |
| Unique items | 80 |
| Positive interaction edges | 4,571 |
| Graph density | 11.88% |
| Cold-start users (degree = 0) | 205 (42.6%) |
| Mean user degree | 9.5 |
| Mean uplift (BTS vs Random) | +0.0003 |
| Pairs with positive uplift | 3,206 / 18,349 |
| Data timespan | ~7 days (Nov 24–30, 2019) |
| Train / Val / Test split | 70% / 15% / 15% (chronological) |

---

*Report generated: 2026-05-04*  
*Dataset: Open Bandit Dataset v1.0 (Saito et al., 2020)*
