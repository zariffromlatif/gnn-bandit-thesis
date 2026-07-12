# Researcher Action Plan: From Thesis to A* / Q1 Publication
## Personal Step-by-Step Guide for the GNN-Bandit Paper

> This file is about **what YOU personally need to do** — not just what the paper needs.  
> Follow this as your weekly/monthly operational guide from now until acceptance.

---

## PHASE 0: MINDSET & REALITY CHECK

Before anything else, understand these facts:

**Fact 1: This will take 8–18 months.**  
Writing, submitting, getting rejected, revising, and resubmitting is the normal path. Plan for it.

**Fact 2: Your supervisor is your most important resource.**  
Every major decision — venue choice, paper framing, when to submit — should be made with your supervisor. Do not submit anywhere without their approval.

**Fact 3: Rejection is not failure.**  
The majority of papers that eventually get published were rejected at least once. Each rejection comes with free expert feedback. Use it.

**Fact 4: The code is as important as the paper.**  
Reviewers increasingly check GitHub repositories. Your preprocessing pipeline (already done) is a head start. The modeling code must be clean and reproducible.

**Fact 5: Reading papers is part of the job.**  
You need to read at least 30–40 papers in your area before writing your related work. Budget 2–3 weeks for this alone.

---

## PHASE 1: FOUNDATION WORK (Weeks 1–4)

This phase is about preparation before you write a single line of the paper.

---

### Step 1.1: Lock Down Your Supervisor Relationship

**What to do:**
- Schedule a dedicated 1-hour meeting with your supervisor to present your GNN-Bandit framework
- Show them the PUBLICATION_ROADMAP.md file (the gap analysis section especially)
- Get their agreement on: (a) the target venue, (b) the timeline, (c) their expected involvement
- Agree on a weekly/biweekly check-in meeting schedule

**Why this matters:**  
Without supervisor buy-in, you cannot submit. Many students lose months waiting for feedback. Establish a clear communication schedule now.

**Deliverable:** A shared Google Doc or email thread confirming the target venue and timeline.

---

### Step 1.2: Do a Deep Literature Review

**What to do:**
- Use Google Scholar, Semantic Scholar, and ACM Digital Library
- Search for papers on these exact topics:
  - "off-policy evaluation bandit"
  - "LightGCN recommendation"
  - "customer retention reinforcement learning"
  - "uplift modeling marketing"
  - "batch constrained Q-learning"
  - "contextual bandit customer churn"
- For each paper you find: read the abstract, skim the methodology, read the conclusion
- Keep a spreadsheet: Paper Title | Authors | Year | Venue | Key Contribution | Relevance to Your Work

**Target:** Find and log 50+ papers. You will cite 40–60 of them.

**Tools to use:**
- [Semantic Scholar](https://www.semanticscholar.org) — free, shows citation counts
- [Connected Papers](https://www.connectedpapers.com) — visual graph of related papers
- [Zotero](https://www.zotero.org) — free reference manager, saves you hours formatting citations
- Google Scholar alerts — set alerts for "off-policy evaluation" and "contextual bandit retention"

**Deliverable:** A Zotero library with 50+ papers organized by topic.

---

### Step 1.3: Read the Papers of Your Target Venue

**What to do:**
- Go to the proceedings of your target conference/journal for the last 3 years
- For KDD: [dl.acm.org/conference/kdd](https://dl.acm.org/conference/kdd)
- For Expert Systems with Applications: search on ScienceDirect
- Find 5–10 papers that are similar in scope to yours (GNNs + RL + applied ML)
- Analyze their structure: How long? How many experiments? How many baselines? What figures?

**Why this matters:**  
You need to write a paper that looks like it belongs in your target venue. Read what gets accepted there and mirror the style and depth.

**Deliverable:** A list of 5 "model papers" you will use as structural reference while writing.

---

### Step 1.4: Understand the Review Criteria

**For KDD:** Research track papers are evaluated on novelty, technical quality, and significance of results.  
**For Expert Systems with Applications:** Reviewers focus on practical applicability, experimental rigor, and clarity of contribution.  
**For Knowledge-Based Systems:** Strong emphasis on the knowledge representation angle and system design.

Download the Call for Papers (CFP) for your target venue and read the review criteria word by word.

---

## PHASE 2: IMPLEMENT THE FULL FRAMEWORK (Weeks 3–10)

This is the most important and most time-consuming phase. You need working code and real results.

---

### Step 2.1: Set Up Your Development Environment

**What to install:**
```
Python 3.10+
PyTorch 2.x
PyTorch Geometric (PyG) — for LightGCN
NumPy, Pandas, SciPy — already used in preprocessing
scikit-learn — for baselines (MF, logistic regression)
d3rlpy or custom BCQ — for the bandit agent
matplotlib, seaborn — for plots
Weights & Biases (wandb) — for experiment tracking (free for students)
```

**Project folder structure to create:**
```
gnn-bandit/
├── data/
│   └── processed/          ← output from preprocess.py
├── src/
│   ├── graph/
│   │   └── lightgcn.py     ← LightGCN encoder
│   ├── agent/
│   │   └── bcq.py          ← BCQ bandit agent
│   ├── ope/
│   │   ├── ipw.py          ← IPS estimator
│   │   ├── dr.py           ← Doubly Robust estimator
│   │   └── dm.py           ← Direct Method estimator
│   ├── baselines/
│   │   ├── random_policy.py
│   │   ├── bts_policy.py
│   │   ├── dqn.py
│   │   └── mf_bandit.py
│   └── utils/
│       └── metrics.py
├── experiments/
│   ├── run_main.py         ← main experiment script
│   ├── run_ablation.py
│   └── run_sensitivity.py
├── notebooks/
│   └── analysis.ipynb      ← for t-SNE, plots
├── preprocess.py           ← already done
├── README.md
└── requirements.txt
```

**Deliverable:** Clean, modular code structure on GitHub (private repo for now).

---

### Step 2.2: Implement LightGCN

The LightGCN encoder takes your `lightgcn_adj.npz` (already computed) as input and produces user/item embeddings.

**Key implementation details:**
- Input: sparse adjacency matrix A (561×561), embedding dimension K
- Forward pass: multi-layer graph convolution, no activation, no transformation
- Output: final user embeddings = mean of all layer embeddings
- Use the normalized adjacency: D^{-1/2} A D^{-1/2}
- Start with K=64, L=2 layers (tune later in sensitivity analysis)

**Reference implementation:** [https://github.com/gusye1234/LightGCN-PyTorch](https://github.com/gusye1234/LightGCN-PyTorch)  
Adapt it to your user-item graph — do not copy it wholesale; understand each component.

**Test it works:** After training, verify that users with similar click histories have similar embeddings (cosine similarity > 0.7 for same-campaign users).

---

### Step 2.3: Implement BCQ Agent

BCQ (Batch-Constrained Q-Learning) is designed for offline/batch RL — exactly your setting.

**Key implementation details:**
- Input: user embedding (from LightGCN) concatenated with context features
- Output: Q-value for each action (item_id), constrained to actions seen in training data
- The constraint: only select actions where the behavioral cloning model assigns probability > threshold τ
- Reward: click signal (0 or 1) adjusted by propensity score

**Reference:** Fujimoto et al. (2019) "Off-Policy Deep Reinforcement Learning without Exploration"  
The `d3rlpy` library has a BCQ implementation: `pip install d3rlpy`

---

### Step 2.4: Implement All Baselines

Each baseline must be implemented cleanly so reviewers can verify your comparison is fair.

| Baseline | Implementation |
|---|---|
| Random Policy | Select item_id uniformly at random |
| BTS (Thompson Sampling) | Use the propensity scores from the bts/ data |
| Vanilla DQN | Standard Q-network, no batch constraint |
| MF + Bandit | Matrix Factorization embeddings fed into same BCQ |
| LightGCN + Greedy | Use GNN embeddings, pick highest Q-value with no constraint |
| Uplift-only | Select action with highest uplift from `uplift_estimates.csv` |

---

### Step 2.5: Implement the OPE Estimators

These are what you use to evaluate your policy without live deployment.

**IPS Estimator:**
```
V_IPS(π) = (1/n) Σ [ r_t * π(a_t|x_t) / π_0(a_t|x_t) ]
```
where π_0 is the logging policy propensity score (already in your dataset).

**Doubly Robust Estimator:**
```
V_DR(π) = V_DM(π) + (1/n) Σ [ (r_t - r_hat_t) * π(a_t|x_t) / π_0(a_t|x_t) ]
```
where r_hat_t is the direct model's reward prediction.

**SNIPW (Self-Normalized):**
```
V_SNIPW(π) = Σ [w_t * r_t] / Σ [w_t]
```
where w_t = π(a_t|x_t) / π_0(a_t|x_t)

**Deliverable:** A `metrics.py` file that takes a policy and test data and returns all four OPE estimates.

---

### Step 2.6: Run All Experiments

Run every experiment 5 times with different random seeds (0, 1, 2, 3, 4).

**Experiment execution order:**
1. Train all baselines → record OPE metrics
2. Train GNN-Bandit (full) → record OPE metrics
3. Run ablations → record OPE metrics for each variant
4. Filter to cold-start users (degree=0) → re-run all comparisons
5. Sensitivity sweep (embedding dim, layers, threshold)
6. Generate t-SNE embeddings from trained LightGCN

**Track everything in Weights & Biases (wandb).**  
Log: loss curves, OPE estimates per epoch, hyperparameters, random seed.  
This makes writing the experimental setup section trivial.

**Deliverable:** A results spreadsheet with mean ± std for every method on every metric.

---

## PHASE 3: WRITE THE PAPER (Weeks 9–14)

Do not start writing the full paper until you have results. Sections 1–4 can be drafted earlier; Sections 5–6 require final numbers.

---

### Step 3.1: Write in This Order

This is the most efficient writing order (not the reading order):

```
1. Methodology (Section 4)       ← you know this best; write it first
2. Experimental Setup (Section 5) ← document what you actually did
3. Results (Section 6)            ← fill in the tables and figures
4. Introduction (Section 1)       ← write last; you now know what you proved
5. Related Work (Section 2)       ← place your work in context
6. Problem Formulation (Section 3) ← formalize what you already did
7. Discussion (Section 7)         ← honest reflection
8. Conclusion (Section 8)         ← summarize contributions + future work
9. Abstract                       ← written absolutely last
```

**Why this order?**  
Most students write the introduction first and get stuck because they don't yet know what they've proven. Write what you know (methodology, results) first.

---

### Step 3.2: Writing Rules for Academic ML Papers

**The Abstract (150–250 words):**
- Sentence 1: The problem (one sentence)
- Sentence 2–3: Why existing methods fail
- Sentence 4–5: What you propose
- Sentence 6–7: How you evaluate it
- Sentence 8: Key result with a number ("our method achieves X% lift over the baseline")
- Sentence 9: What this enables

**The Introduction:**
- Do NOT start with "In recent years, machine learning has..."
- Start with a compelling statistic: "Customer churn costs companies an estimated $1.6 trillion annually..."
- End with a clear, numbered contributions list
- The contributions must match exactly what you prove in the paper

**The Related Work:**
- Never write a list of summaries ("Paper A does X. Paper B does Y.")
- Write it as a narrative: "While prior work on GNN-based recommendation [X, Y, Z] has shown strong performance, these methods do not address the intervention selection problem. Separately, contextual bandit approaches [A, B] have been applied to marketing but without graph-based user representations. Our work bridges these two lines..."

**The Methodology:**
- Every equation must be numbered
- Every variable must be defined when first introduced
- Include a system architecture figure (a diagram showing the full pipeline)
- Include a pseudocode algorithm box

**The Results:**
- Every table needs: baseline rows, your method row (bold), improvement column
- Every table needs a caption that states the main finding
- Every figure needs a caption that explains what to observe
- Never present a result without discussing what it means

---

### Step 3.3: Figures You Must Create

| Figure | What It Shows | How to Make It |
|---|---|---|
| System Architecture | Full pipeline from OBD → GNN → BCQ → OPE | Draw.io or PowerPoint |
| Bipartite Graph Visualization | User-item interaction graph (sample) | NetworkX + matplotlib |
| Main Results Table | DR reward: all methods × all campaigns | LaTeX table |
| Ablation Bar Chart | DR reward per ablation variant | matplotlib |
| Cold-Start Performance | GNN-Bandit vs. baselines for degree-0 users | grouped bar chart |
| t-SNE Embeddings | User embeddings colored by engagement level | scikit-learn TSNE |
| Sensitivity Line Plots | DR reward vs. embedding dim / layers | matplotlib |
| OPE Estimator Comparison | IPW vs. SNIPW vs. DR vs. DM | grouped bar chart |

**Figure quality requirements:**
- All figures must be vector format (PDF or SVG) for journal submission
- Minimum 300 DPI if using PNG
- Font size in figures must be readable when printed at column width (~8cm)
- Use a consistent color palette across all figures

---

### Step 3.4: LaTeX Setup

Almost all A* conferences and Q1 journals require LaTeX. If you are not familiar with it, start now.

**Tools:**
- [Overleaf](https://www.overleaf.com) — free online LaTeX editor, no installation needed
- Download the template for your target venue from their official website
  - KDD: uses ACM SigConf template
  - Expert Systems with Applications: uses Elsevier template (elsarticle)

**Overleaf tips:**
- Use `\cite{}` for references — Zotero exports `.bib` files directly to Overleaf
- Use `\label{}` and `\ref{}` for all figures, tables, and equations
- Use `booktabs` package for professional-looking tables
- Never use `\textbf{}` for emphasis in body text — use it only for results in tables

---

## PHASE 4: INTERNAL REVIEW (Weeks 14–16)

Before submitting to any journal or conference, the paper must go through multiple rounds of review within your own team.

---

### Step 4.1: Self-Review Checklist

Read your paper as if you are a hostile reviewer. Ask yourself:

- [ ] Is every claim supported by an experiment or a citation?
- [ ] Is the contribution clearly different from all cited papers?
- [ ] Are there any numbers in the paper that cannot be reproduced from your code?
- [ ] Does every figure have a clear message?
- [ ] Is the abstract self-contained (can someone understand your work from the abstract alone)?
- [ ] Have you honestly stated the limitations?
- [ ] Does your conclusion match what you actually proved?

---

### Step 4.2: Supervisor Review

**How to request a review from your supervisor:**
- Send the paper with a specific list of questions: "Is the contribution framing strong enough? Is the related work coverage sufficient? Is Experiment 3 convincing?"
- Give them at least 2 weeks to review
- Incorporate all feedback before asking for the next round
- Expect 2–3 rounds of revision before they approve submission

---

### Step 4.3: Peer Review (Optional but Recommended)

Ask a lab colleague or friend in a related field to read Section 1 (Introduction) and tell you:
- What do they think the paper is about?
- What do they think the main contribution is?
- What is confusing?

If their answers don't match what you intended, your writing is not clear enough.

---

## PHASE 5: SUBMISSION (Week 16–18)

---

### Step 5.1: Prepare Submission Materials

Most venues require:
- Main paper (PDF, formatted to their template)
- Supplementary material (optional but recommended — put extra experiments here)
- Code repository link (GitHub)
- A cover letter (journals only — 1 page explaining what you're submitting and why it fits the journal)

**Cover letter template for journals:**
```
Dear Editor,

We submit our manuscript titled "[Title]" for consideration in [Journal Name].

This paper proposes [1-2 sentence summary]. We believe this work is a strong 
fit for [Journal Name] because [specific reason related to the journal's scope].

Our key contributions are:
1. [Contribution 1]
2. [Contribution 2]
3. [Contribution 3]

This manuscript has not been submitted elsewhere and all authors have approved 
the submission.

Sincerely,
[Your Name]
```

---

### Step 5.2: Submission Process

**For journals (e.g., Expert Systems with Applications):**
1. Go to the journal's submission portal (usually Editorial Manager or EviseS)
2. Create an account
3. Upload: manuscript PDF, cover letter, highlights (3–5 bullet points), graphical abstract (optional)
4. Suggest 3–5 potential reviewers (optional but helps — pick authors of papers you cite)
5. You will receive a decision in 3–6 months: Accept / Minor Revision / Major Revision / Reject

**For conferences (e.g., KDD):**
1. Go to the conference submission portal (usually OpenReview or HotCRP)
2. Submit your paper PDF before the abstract deadline (usually 1 week before full paper)
3. You will receive reviews in 6–8 weeks
4. Most A* conferences have a rebuttal phase (48–72 hours to respond to reviews)
5. Final decision comes 2–4 weeks after rebuttal

---

### Step 5.3: The Rebuttal (Conferences Only)

The rebuttal is your chance to correct factual errors in reviews and clarify misunderstandings. It is NOT a place to promise future experiments.

**Rules for a good rebuttal:**
- Address every reviewer concern directly, in order
- Be polite — never argue aggressively
- Provide numbers if a reviewer questions a result
- If a reviewer suggests an experiment you can run in 48 hours, run it and include the result
- Acknowledge valid criticisms: "Reviewer 2 correctly points out that..."
- Keep it concise — reviewers read many rebuttals quickly

**Example rebuttal structure:**
```
We thank all reviewers for their thorough and constructive feedback.

== Response to Reviewer 1 ==
[R1.1] "The dataset is small..."
We acknowledge this limitation. However, [your response with evidence].

[R1.2] "No statistical significance..."
We ran significance tests (Wilcoxon, p < 0.05) which confirm...

== Response to Reviewer 2 ==
...
```

---

## PHASE 6: REVISION & RESUBMISSION

---

### Step 6.1: How to Handle a "Major Revision" Decision

A major revision is NOT a rejection. It means the editor believes the paper has potential and is giving you a chance to fix it.

**What to do:**
1. Read all reviews carefully 3 times before responding
2. Create a "Response to Reviewers" document
3. For every single reviewer comment, write:
   - The original comment (quoted)
   - Your response
   - What you changed in the paper (with page/line numbers)
4. Make every requested change, even ones you disagree with (unless they are factually wrong)
5. Re-submit within the given deadline (usually 60–90 days)

**The Response to Reviewers document is as important as the revised paper.** Make it thorough.

---

### Step 6.2: How to Handle a Rejection

**Immediate actions:**
- Do not resubmit the same paper to the same venue
- Read the reviews. Categorize each comment as: (a) valid criticism, (b) misunderstanding, (c) scope mismatch
- For valid criticisms: fix them before the next submission
- For misunderstandings: rewrite the relevant section more clearly
- For scope mismatch: choose a more appropriate venue next time

**Next submission:**
- Choose a venue one tier below (e.g., KDD rejected → try WWW or Expert Systems)
- Address ALL reviewer concerns before resubmitting
- Do not resubmit within 2–3 weeks — take time to genuinely improve the paper

---

### Step 6.3: How to Handle an Acceptance

1. Read the acceptance email carefully — there will be conditions (minor revisions, copyright forms)
2. Complete all requested revisions promptly
3. Sign the copyright transfer agreement
4. Prepare the camera-ready version (final formatted PDF)
5. Upload all final materials by the camera-ready deadline
6. Post a preprint to arXiv (optional but standard practice — makes your work immediately visible)
7. Update your GitHub repository with the final paper version

---

## PHASE 7: TOOLS & RESOURCES

### Writing Tools
| Tool | Purpose | Cost |
|---|---|---|
| Overleaf | LaTeX paper writing | Free (student plan) |
| Grammarly | Grammar and style checking | Free tier available |
| Hemingway Editor | Clarity and readability | Free web version |
| Zotero | Reference management + citation export | Free |
| Connected Papers | Visualize related work | Free |

### Experiment Tools
| Tool | Purpose | Cost |
|---|---|---|
| Weights & Biases (wandb) | Experiment tracking and visualization | Free for academics |
| Google Colab Pro | GPU training if local GPU is insufficient | ~$10/month |
| PyTorch Geometric | GNN implementation (LightGCN) | Free |
| d3rlpy | BCQ and offline RL implementations | Free |
| scikit-learn | Baselines and preprocessing | Free |

### Staying Current
| Resource | What It Provides |
|---|---|
| arXiv cs.LG / cs.IR | Daily preprints in your area |
| Papers With Code | Papers + code + benchmarks |
| Google Scholar Alerts | Email alerts for new citations of key papers |
| Twitter/X @paperswithcode | Daily ML paper highlights |
| ResearchGate | Networking with other researchers |

---

## WEEKLY ACTION SCHEDULE

Use this as a rough template. Adjust based on your course load and supervisor schedule.

### Weeks 1–2: Foundation
- Set up Zotero, read 20 key papers
- Set up GitHub repo with folder structure
- Meet supervisor: confirm target venue and timeline

### Weeks 3–4: Literature & Setup
- Read 30 more papers, complete literature spreadsheet
- Install all tools, set up development environment
- Read 5 accepted papers from target venue

### Weeks 5–6: LightGCN Implementation
- Implement and test LightGCN encoder
- Verify embeddings make sense (similarity check)
- Train on train.csv, evaluate on val.csv

### Weeks 7–8: BCQ + Baselines
- Implement BCQ agent
- Implement all 6 baselines
- Implement OPE estimators (IPS, DR, SNIPW, DM)

### Weeks 9–10: Run All Experiments
- Main experiment: all methods × all campaigns × 5 seeds
- Ablation study
- Cold-start analysis

### Weeks 11–12: Additional Experiments
- Sensitivity analysis
- t-SNE visualization
- OPE estimator comparison

### Weeks 13–14: Write Methodology + Results
- Write Sections 4 and 5 (methodology + setup)
- Fill in all result tables and create figures

### Weeks 15–16: Write Full Paper
- Write Introduction, Related Work, Problem Formulation
- Write Discussion and Conclusion
- Write Abstract last

### Weeks 17–18: Review + Revise
- Self-review pass
- Supervisor review round 1
- Revise

### Weeks 19–20: Final Preparation + Submit
- Supervisor review round 2
- Grammar check
- Prepare submission materials
- Submit

---

## FINAL HONEST ADVICE

**On perfectionism:** Your paper will never feel ready. There will always be one more experiment you could run, one more baseline you could add. At some point you have to submit. A paper submitted is infinitely more valuable than a perfect paper that never leaves your hard drive.

**On comparison to others:** Do not compare your timeline to PhD students at top universities with full-time research positions, multiple supervisors, and GPU clusters. Your path is different. Measure your progress against your own previous work, not against others.

**On solo vs. collaborative work:** If possible, add a co-author (classmate, lab member) who can contribute to the implementation or writing. Two people catch more errors than one. Reviewers also view multi-author papers as more credible for a first submission.

**On the Open Bandit Dataset:** The small user count (481) is a known limitation. Be the first to acknowledge it in your paper — in the Discussion or Limitations section. Reviewers respect honesty about limitations far more than they respect attempts to hide them.

**On your thesis vs. the paper:** Your thesis and your journal paper are different documents with different audiences. The thesis is for your university committee. The paper is for the research community. They will overlap significantly, but do not copy-paste your thesis chapters directly into the paper — rewrite for the academic audience.

---

*This action plan was generated based on your GNN-Bandit framework and Open Bandit Dataset project.*  
*Generated: 2026-05-04*
