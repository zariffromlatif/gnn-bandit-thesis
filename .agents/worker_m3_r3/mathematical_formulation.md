# Formal Mathematical Formulation: Graph-Enhanced Causal Reinforcement Learning (GNN-Bandit)

---

## 1. Problem Setup: Offline Contextual Bandit Formulation

Let $\mathcal{M} = (\mathcal{X}, \mathcal{A}, \mathcal{R}, \mathcal{P}, \pi_0)$ define an offline contextual bandit environment under batch logging constraints:

- $\mathcal{X} \subseteq \mathbb{R}^d$ denotes the context space representing user state profiles, historical behaviors, and relational topological signals.
- $\mathcal{A} = \{a_1, a_2, \dots, a_K\}$ denotes the finite, discrete action space of size $|\mathcal{A}| = K$, corresponding to candidate recommendation items or marketing interventions.
- $\mathcal{R}: \mathcal{X} \times \mathcal{A} \to [0, 1]$ represents the stochastic reward distribution, where $r \sim \mathcal{R}(x, a)$ represents the observed feedback (e.g., click, conversion, retention event) with expected reward $r(x, a) = \mathbb{E}[r \mid x, a]$.
- $\mathcal{P}(x)$ denotes the exogenous distribution over user contexts.
- $\pi_0(a \mid x) = \mathbb{P}(A = a \mid X = x)$ denotes the stochastic **logging policy** (behavior policy) that collected the historical logged dataset $\mathcal{D}_0$.

### Historical Logged Dataset
The agent is restricted to an offline logged dataset $\mathcal{D}_0$ comprising $N$ independent interaction trajectories:
$$\mathcal{D}_0 = \left\{ (x_i, a_i, r_i, p_i) \right\}_{i=1}^N$$
where $x_i \sim \mathcal{P}$, $a_i \sim \pi_0(\cdot \mid x_i)$, $r_i \sim \mathcal{R}(x_i, a_i)$, and $p_i = \pi_0(a_i \mid x_i)$ is the logging propensity score.

### Target Objective
The objective is to find a target policy $\pi: \mathcal{X} \to \Delta(\mathcal{A})$ that maximizes the expected counterfactual value (expected policy return):
$$V(\pi) = \mathbb{E}_{x \sim \mathcal{P}, a \sim \pi(\cdot \mid x)} \left[ r(x, a) \right] = \int_{\mathcal{X}} \sum_{a \in \mathcal{A}} \pi(a \mid x) r(x, a) \mathcal{P}(x) \, dx$$
subject to batch offline safety constraints and without access to online environmental interaction.

---

## 2. Causal Potential Outcomes & Uplift Identification

To formalize the prescriptive intervention mechanism, we ground the decision process in the **Neyman-Rubin Potential Outcomes Framework**.

### 2.1 Potential Outcomes & Treatment Effects
For each user context $x \in \mathcal{X}$ and candidate action $a \in \mathcal{A}$:
- Let $Y(a) \in \{0, 1\}$ denote the potential outcome that would have been observed had the user received action $a$.
- The observed reward is given by the consistency equation:
  $$r = \sum_{a \in \mathcal{A}} \mathbb{I}(A = a) Y(a)$$
- For a focal intervention $a$ relative to a null/baseline control action $a_0$, the **Individual Treatment Effect (ITE)** is defined as:
  $$\tau(x, a) = Y(a) - Y(a_0)$$
- The **Conditional Average Treatment Effect (CATE)** (or heterogeneous uplift) is given by:
  $$\tau_{\mathrm{CATE}}(x, a) = \mathbb{E}[Y(a) - Y(a_0) \mid X = x] = \mu(x, a) - \mu(x, a_0)$$
  where $\mu(x, a) = \mathbb{E}[Y(a) \mid X = x]$.

### 2.2 Causal Identification Assumptions
To guarantee identifiability of $V(\pi)$ and $\tau_{\mathrm{CATE}}(x, a)$ from observational logged data $\mathcal{D}_0$, three fundamental assumptions must hold:

1. **Stable Unit Treatment Value Assumption (SUTVA)**:
   The potential outcomes for any user context $x_i$ are independent of the treatment assigned to user $x_j$ ($i \ne j$), and there are no multiple versions of treatment action $a$.
2. **Unconfoundedness (Conditional Ignorability)**:
   Given the observed context $X \in \mathcal{X}$, treatment assignment is conditionally independent of potential outcomes:
   $$\{Y(a)\}_{a \in \mathcal{A}} \perp A \mid X$$
3. **Positivity (Overlap / Common Support)**:
   Every action has a non-zero probability of assignment across the entire support of the context distribution:
   $$\forall x \in \mathrm{supp}(\mathcal{P}), \forall a \in \mathcal{A}, \quad \pi_0(a \mid x) \ge \epsilon > 0$$

### 2.3 User Heterogeneity & Sleeping Dogs Partitioning
Under the causal uplift framing, the user population $\mathcal{X}$ is partitioned into four distinct behavioral quadrants based on potential outcomes $(Y(a), Y(a_0))$:
- **Persuadables ($Y(a)=1, Y(a_0)=0$)**: Positive uplift ($\tau > 0$). These users convert *only* if treated.
- **Sure Things ($Y(a)=1, Y(a_0)=1$)**: Zero uplift ($\tau = 0$). These users convert regardless of treatment (marketing waste).
- **Lost Causes ($Y(a)=0, Y(a_0)=0$)**: Zero uplift ($\tau = 0$). These users never convert regardless of treatment.
- **Sleeping Dogs ($Y(a)=0, Y(a_0)=1$)**: Negative uplift ($\tau < 0$). These users are harmed by intervention (e.g., unsubscribing upon receiving marketing messages).

The prescriptive objective of GNN-Bandit is to selectively intervene on Persuadables while strictly avoiding Sleeping Dogs.

---

## 3. Graph Representation Theory & Relational Convolutions

### 3.1 Bipartite User-Item Topology
Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ denote the bipartite interaction graph, where $\mathcal{V} = \mathcal{U} \cup \mathcal{I}$ comprises the disjoint sets of user nodes $\mathcal{U}$ ($|\mathcal{U}| = N_u$) and item/action nodes $\mathcal{I}$ ($|\mathcal{I}| = N_i$). 
The adjacency matrix $\mathbf{A} \in \mathbb{R}^{(N_u + N_i) \times (N_u + N_i)}$ is defined as:
$$\mathbf{A} = \begin{bmatrix} \mathbf{0} & \mathbf{R} \\ \mathbf{R}^\top & \mathbf{0} \end{bmatrix}$$
where $\mathbf{R} \in \mathbb{R}^{N_u \times N_i}$ represents the historical positive interaction matrix ($R_{ui} = 1$ if user $u$ interacted with item $i$, and $0$ otherwise).

### 3.2 Symmetrically Normalized Graph Laplacian
Let $\mathbf{D} \in \mathbb{R}^{(N_u + N_i) \times (N_u + N_i)}$ be the diagonal degree matrix with $D_{vv} = \sum_{v'} A_{vv'}$. The symmetrically normalized adjacency matrix $\tilde{\mathbf{A}}$ is:
$$\tilde{\mathbf{A}} = \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$$
For isolated nodes ($D_{vv} = 0$), $D_{vv}^{-1/2} \triangleq 0$.

### 3.3 LightGCN Layer-Wise Message Passing
LightGCN discards non-linear feature transformations and activations during neighborhood aggregation, retaining only pure linear diffusion:
$$\mathbf{e}_u^{(k+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u| |\mathcal{N}_i|}} \mathbf{e}_i^{(k)}$$
$$\mathbf{e}_i^{(k+1)} = \sum_{u \in \mathcal{N}_i} \frac{1}{\sqrt{|\mathcal{N}_i| |\mathcal{N}_u|}} \mathbf{e}_u^{(k)}$$
where $\mathbf{e}_v^{(0)} \in \mathbb{R}^d$ is the learnable 0-th layer ID embedding for node $v \in \mathcal{V}$.

In matrix notation across all $L$ layers:
$$\mathbf{E}^{(k+1)} = \tilde{\mathbf{A}} \mathbf{E}^{(k)}$$
The final representation $\mathbf{E}^*$ is computed via uniform layer combination:
$$\mathbf{E}^* = \sum_{k=0}^L \alpha_k \mathbf{E}^{(k)}, \quad \text{where } \alpha_k = \frac{1}{L + 1}$$

### 3.4 Spectral Filtering & Causal Smoothing Property
**Theorem 1 (Low-Pass Causal Smoothing)**:
Let $\mathbf{L}_{\mathrm{sym}} = \mathbf{I} - \tilde{\mathbf{A}} = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^\top$ denote the eigendecomposition of the normalized graph Laplacian, where eigenvalues $\lambda_m \in [0, 2]$. The $L$-layer LightGCN operator acts as a spectral polynomial filter $g(\lambda) = \sum_{k=0}^L \alpha_k (1 - \lambda)^k$.

*Proof*:
For any initial signal $\mathbf{x} \in \mathbb{R}^{|\mathcal{V}|}$, the $k$-th diffusion step is:
$$\tilde{\mathbf{A}}^k \mathbf{x} = (\mathbf{I} - \mathbf{L}_{\mathrm{sym}})^k \mathbf{x} = \mathbf{U} (\mathbf{I} - \mathbf{\Lambda})^k \mathbf{U}^\top \mathbf{x}$$
Summing over all layers $k \in \{0, \dots, L\}$ with weights $\alpha_k$:
$$\mathbf{E}^* = \sum_{k=0}^L \alpha_k \tilde{\mathbf{A}}^k \mathbf{E}^{(0)} = \mathbf{U} \left( \sum_{k=0}^L \alpha_k (\mathbf{I} - \mathbf{\Lambda})^k \right) \mathbf{U}^\top \mathbf{E}^{(0)} = \mathbf{U} g(\mathbf{\Lambda}) \mathbf{U}^\top \mathbf{E}^{(0)}$$
Since $g(\lambda) = \frac{1}{L+1} \sum_{k=0}^L (1 - \lambda)^k$ is strictly monotonically decreasing on $\lambda \in [0, 2]$ with $g(0) = 1$ and $g(2) \approx 0$, high-frequency graph noise is attenuated while low-frequency homophilic causal response patterns are preserved and diffused across multi-hop neighbors. $\blacksquare$

---

## 4. Batch-Constrained Reinforcement Learning (BCQ) Formulation

### 4.1 Distributional Shift & Extrapolation Error
In offline policy optimization, standard Q-learning updates evaluate:
$$\max_{a' \in \mathcal{A}} Q(s, a')$$
When an action $a'$ has low or zero support under the logging policy ($\pi_0(a' \mid s) \approx 0$), the neural network approximator $\hat{Q}(s, a')$ produces arbitrary, unconstrained extrapolation errors. Subsequent maximization greedily selects these phantom high-value actions, leading to catastrophic offline policy failure.

### 4.2 State-Conditioned Action Plausibility Filtering
To eliminate extrapolation error, GNN-Bandit implements a state-conditioned Batch-Constrained action mask.
1. **Behavior Cloning Network**: A parameterized behavioral policy $\hat{P}_\beta(a \mid s)$ is trained via maximum likelihood over logged trajectories:
   $$\mathcal{L}_{\mathrm{BC}}(\theta_\beta) = -\frac{1}{N} \sum_{i=1}^N \log \hat{P}_\beta(a_i \mid s_i; \theta_\beta)$$
2. **Dynamic Action Constraint**: The set of admissible actions $\hat{\mathcal{A}}(s) \subseteq \mathcal{A}$ is restricted to those exceeding an adaptive threshold:
   $$\hat{\mathcal{A}}(s) = \left\{ a \in \mathcal{A} \;\middle|\; \hat{P}_\beta(a \mid s) \ge \tau \cdot \max_{a' \in \mathcal{A}} \hat{P}_\beta(a' \mid s) \right\}$$
   where $\tau = \frac{\rho}{|\mathcal{A}|}$ is calibrated via the threshold ratio $\rho \in (0, 1]$.
3. **Safety Fallback Guarantee**: If $|\hat{\mathcal{A}}(s)| < K_{\min}$, $\hat{\mathcal{A}}(s)$ is populated with the top-$K_{\min}$ actions ranked by $\hat{P}_\beta(\cdot \mid s)$ to prevent empty action sets.

### 4.3 Quantile Q-Learning & CVaR Risk-Sensitive Objective
Rather than estimating scalar expected returns $\mathbb{E}[Q(s, a)]$, GNN-Bandit models the full return distribution via Quantile Regression with $M$ quantiles:
$$\theta_m(s, a) = \text{MLP}_Q(s, a; \theta_Q)_m, \quad m \in \{1, \dots, M\}$$
The network is trained using the Huber quantile loss at quantile targets $\xi_m = \frac{2m - 1}{2M}$:
$$\mathcal{L}_{\mathrm{QR}}(\theta_Q) = \sum_{m=1}^M \mathbb{E} \left[ \rho_{\xi_m}^\kappa \left( r - \theta_m(s, a) \right) \right]$$
where $\rho_\xi^\kappa(u) = |\xi - \mathbb{I}(u < 0)| \frac{\mathcal{L}_\kappa(u)}{\kappa}$.

To ensure risk-averse safe optimization in customer retention, we evaluate the **Conditional Value-at-Risk (CVaR)** at tail level $\alpha \in (0, 1]$:
$$\text{CVaR}_\alpha(s, a) = \frac{1}{\alpha} \int_0^\alpha F_{Q(s, a)}^{-1}(u) \, du \approx \frac{1}{\lfloor \alpha M \rfloor} \sum_{m=1}^{\lfloor \alpha M \rfloor} \theta_m(s, a)$$

### 4.4 Hybrid Collaborative Policy Scoring
When item embeddings $\mathbf{e}_a^*$ are available from the GNN encoder, the decision score combines the CVaR-adjusted Q-value with structural collaborative dot-product similarity:
$$\text{score}(s, a) = \mathcal{Z}\Big( \text{CVaR}_\alpha(s, a) \Big) + \lambda_{\mathrm{hybrid}} \cdot \mathcal{Z}\Big( \mathbf{e}_u^{*\top} \mathbf{e}_a^* \Big)$$
where $\mathcal{Z}(v) = \frac{v - \mu_v}{\sigma_v + \epsilon}$ denotes sample-wise z-score normalization across candidate actions $a \in \hat{\mathcal{A}}(s)$.

The target evaluation policy $\pi(a \mid s)$ applies temperature-scaled softmax over the constrained action space:
$$\pi(a \mid s) = \begin{cases} \frac{\exp\left( \text{score}(s, a) / T \right)}{\sum_{a' \in \hat{\mathcal{A}}(s)} \exp\left( \text{score}(s, a') / T \right)}, & \text{if } a \in \hat{\mathcal{A}}(s) \\ 0, & \text{otherwise} \end{cases}$$

---

## 5. Off-Policy Evaluation (OPE) Theory & Double Robustness

### 5.1 OPE Estimators
To evaluate the target policy $\pi$ using offline data $\mathcal{D}_0$ generated by $\pi_0$, we employ four standard estimators:

1. **Inverse Propensity Weighting (IPW)**:
   $$\hat{V}_{\mathrm{IPW}}(\pi) = \frac{1}{N} \sum_{i=1}^N w_i r_i, \quad \text{where } w_i = \frac{\pi(a_i \mid x_i)}{\pi_0(a_i \mid x_i)}$$
2. **Self-Normalized IPW (SNIPW)**:
   $$\hat{V}_{\mathrm{SNIPW}}(\pi) = \frac{\sum_{i=1}^N w_i r_i}{\sum_{i=1}^N w_i}$$
3. **Direct Method (DM)**:
   $$\hat{V}_{\mathrm{DM}}(\pi) = \frac{1}{N} \sum_{i=1}^N \sum_{a \in \mathcal{A}} \pi(a \mid x_i) \hat{r}(x_i, a)$$
   where $\hat{r}(x, a)$ is a supervised reward regression model.
4. **Doubly Robust (DR) Estimator**:
   $$\hat{V}_{\mathrm{DR}}(\pi) = \frac{1}{N} \sum_{i=1}^N \left[ \sum_{a \in \mathcal{A}} \pi(a \mid x_i) \hat{r}(x_i, a) + w_i \Big( r_i - \hat{r}(x_i, a_i) \Big) \right]$$

---

### 5.2 Double Robustness: Formal Theorem & Proof

**Theorem 2 (Unbiasedness of the Doubly Robust Estimator)**:
Under SUTVA, Unconfoundedness, and Positivity, if **either** the reward model is correctly specified ($\hat{r}(x, a) = r(x, a)$) **or** the propensity model is correctly specified ($\hat{\pi}_0(a \mid x) = \pi_0(a \mid x)$), then $\hat{V}_{\mathrm{DR}}(\pi)$ is an unbiased estimator of the true policy value $V(\pi)$:
$$\mathbb{E}_{\mathcal{D}_0} \left[ \hat{V}_{\mathrm{DR}}(\pi) \right] = V(\pi)$$

*Proof*:
Taking the expectation of a single sample $i$ under the data generating process:
$$\mathbb{E} \left[ \hat{V}_{\mathrm{DR}}^{(i)}(\pi) \right] = \mathbb{E}_{X} \left[ \sum_{a \in \mathcal{A}} \pi(a \mid X) \hat{r}(X, a) + \mathbb{E}_{A, R \mid X} \left[ \frac{\pi(A \mid X)}{\pi_0(A \mid X)} (R - \hat{r}(X, A)) \;\middle|\; X \right] \right]$$

Expanding the inner conditional expectation over treatment assignment $A \sim \pi_0(\cdot \mid X)$ and reward $R$:
$$\mathbb{E}_{A, R \mid X} \left[ \frac{\pi(A \mid X)}{\pi_0(A \mid X)} (R - \hat{r}(X, A)) \;\middle|\; X \right] = \sum_{a \in \mathcal{A}} \pi_0(a \mid X) \frac{\pi(a \mid X)}{\pi_0(a \mid X)} \Big( \mathbb{E}[R \mid X, A=a] - \hat{r}(X, a) \Big)$$
$$= \sum_{a \in \mathcal{A}} \pi(a \mid X) \Big( r(X, a) - \hat{r}(X, a) \Big)$$

Substitute this back into the outer expectation:
$$\mathbb{E} \left[ \hat{V}_{\mathrm{DR}}^{(i)}(\pi) \right] = \mathbb{E}_{X} \left[ \sum_{a \in \mathcal{A}} \pi(a \mid X) \hat{r}(X, a) + \sum_{a \in \mathcal{A}} \pi(a \mid X) \Big( r(X, a) - \hat{r}(X, a) \Big) \right]$$
$$= \mathbb{E}_{X} \left[ \sum_{a \in \mathcal{A}} \pi(a \mid X) r(X, a) \right] = V(\pi)$$

**Case 1 (Correct Propensity $\pi_0$, Arbitrary Reward Model $\hat{r}$)**:
The $\hat{r}(X, a)$ terms cancel identically regardless of the functional form of $\hat{r}(X, a)$, yielding $\mathbb{E}[\hat{V}_{\mathrm{DR}}] = V(\pi)$.

**Case 2 (Correct Reward Model $\hat{r}(x, a) = r(x, a)$, Misspecified Propensity $\hat{\pi}_0$)**:
If $\hat{r}(x, a) = r(x, a)$, then $R - \hat{r}(X, A) = 0$ in expectation, so the second term vanishes:
$$\mathbb{E} \left[ \hat{V}_{\mathrm{DR}}^{(i)}(\pi) \right] = \mathbb{E}_{X} \left[ \sum_{a \in \mathcal{A}} \pi(a \mid X) r(X, a) \right] + 0 = V(\pi)$$
Thus, the estimator is doubly robust. $\blacksquare$

---

### 5.3 Asymptotic Variance Derivation

**Theorem 3 (Asymptotic Variance of $\hat{V}_{\mathrm{DR}}$)**:
The asymptotic variance of $\hat{V}_{\mathrm{DR}}(\pi)$ is given by:
$$\mathbb{V}\left[ \hat{V}_{\mathrm{DR}}(\pi) \right] = \frac{1}{N} \left( \mathbb{V}_X \left[ \sum_{a \in \mathcal{A}} \pi(a \mid X) r(X, a) \right] + \mathbb{E}_X \left[ \sum_{a \in \mathcal{A}} \frac{\pi^2(a \mid X)}{\pi_0(a \mid X)} \sigma^2(X, a) \right] + \mathbb{E}_X \left[ \sum_{a \in \mathcal{A}} \frac{\pi^2(a \mid X)}{\pi_0(a \mid X)} \Big( r(X, a) - \hat{r}(X, a) \Big)^2 \right] \right)$$
where $\sigma^2(x, a) = \mathbb{V}[R \mid X=x, A=a]$.

*Significance*: 
When the reward model $\hat{r}(x, a)$ is accurate, the third variance term approaches zero, substantially reducing variance compared to standard IPW whose variance includes the raw second moment $\mathbb{E}[R^2]$ without baseline subtraction.

---

## 6. End-to-End GNN-Bandit Algorithmic Flowchart

```
[ Bipartite Graph G = (U, I, E) ]
              │
              ▼
   [ Symmetrically Normalized Laplacian A_tilde ]
              │
              ▼
   [ LightGCN Multi-Hop Linear Diffusion ] ──► Node Embeddings E* = [e_u*, e_a*]
              │
              ├────────────────────────────────────────┐
              ▼                                        ▼
   [ Augmented State s = (x_u, e_u*) ]        [ CATE / Uplift Network ]
              │                                        │
              ├────────────────────────┐               │
              ▼                        ▼               │
   [ Behavior Cloning P_beta(a|s) ]   [ Quantile Q-Network (CVaR_alpha) ]
              │                        │
              ▼                        │
   [ Action Mask A_hat(s) ] ◄──────────┘
              │
              ▼
   [ Hybrid Policy score(s, a) = z(CVaR) + lambda * z(e_u^T e_a) ]
              │
              ▼
   [ Constrained Softmax Target Policy pi(a|s) ]
              │
              ▼
   [ Off-Policy Evaluation (Doubly Robust V_DR) ]
```

