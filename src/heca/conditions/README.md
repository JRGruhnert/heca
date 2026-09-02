# Scoring math: `score_single` and `containment_score`

This document is about the **mathematics** of the two scoring functions used
by `heca.conditions.condition` (implemented in
[`entity.py`](../data/entity.py)):

- **`score_single`** — the *membership* problem: is a given value inside one
  condition?
- **`containment_score`** — the *agreement* problem: do two conditions share a
  value that can occur under both?

For each problem we first survey the standard methods from the literature and
then explain, formula by formula, which method the code implements and why.

---

## Notation

| symbol | meaning |
|---|---|
| $v$ | full entity value, $[\text{pos}(3), \text{aa}(3), \text{extra}(D), \text{ste}(1)]$ |
| $v_p$ | continuous pose part, dimension $d$ (pos + rot + extra, excluding the state) |
| $s$ | discrete state, $s \in \{1,\dots,S\}$ |
| $K$ | mixture components, weights $w_k$ ($\sum_k w_k = 1$) |
| $\mu_k$, $\Sigma_k = \operatorname{diag}(\sigma^2_{k,1},\dots,\sigma^2_{k,d})$ | component mean / diagonal covariance |
| $\pi_k$ | categorical state distribution of component $k$ |
| $q$ | quantile, `cfg.z_quantile_joint` (default $0.99$) — joint ellipsoid level for membership *and* containment; per-dim cap `cfg.z_quantile_dim` (default $0.999 \Leftrightarrow 3.29\sigma$) |
| $\chi^2_q(d)$ | $q$-quantile of the chi-squared distribution with $d$ dof |

**Model.** Each condition fits, per entity, a Gaussian-mixture × categorical:

$$
p(v) = \sum_{k=1}^{K} w_k\,
\mathcal{N}\!\big(v_p; \mu_k, \Sigma_k\big)\,
\operatorname{Cat}(s; \pi_k),
\qquad
\mathcal{N}(v_p;\mu_k,\Sigma_k)
= \prod_{d'=1}^{d} \frac{1}{\sqrt{2\pi\sigma^2_{k,d'}}}
  \exp\!\left(-\frac{(v_{p,d'}-\mu_{k,d'})^2}{2\sigma^2_{k,d'}}\right).
$$

All scoring happens in **model space** (`model_value` drops the rotation
columns when `cfg.fit_rotation = False`).

---

# 1. `score_single` — the membership problem

**Problem.** Decide whether a value $v$ "belongs to" the condition: is
$v_p$ plausible under the pose model **and** is $s$ the state the model
predicts?

## 1.1 Methods that exist

| method | test | references / notes |
|---|---|---|
| per-dimension $z$-score / "$3\sigma$ rule" | $\max_d \frac{|v_{p,d}-\mu_d|}{\sigma_d} \le c$ | classic; simple, but a *box*, not the joint region; ignores that deviations can compensate |
| **joint Mahalanobis + $\chi^2$ quantile** | $z^2 = \sum_d \frac{(v_{p,d}-\mu_d)^2}{\sigma^2_d} \le \chi^2_q(d)$ | the standard multivariate membership test — [Hotelling's $T^2$](https://www.statmod.org/smij/Vol13/Iss5&6/Romer/Abstract.html), [scikit-learn `EllipticEnvelope`](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html), [Mahalanobis-based outlier detection](https://link.springer.com/article/10.1186/s13677-024-00682-0), [R `mvoutlier`](https://cran.r-universe.dev/mvoutlier/doc/manual.html) |
| likelihood ratio | $p(v)/\max_x p(x)$ against a cutoff | graded; but the cutoff has no statistical calibration |
| **joint $\chi^2$ + per-dimension cap** | $z^2 \le \chi^2_q(d)$ **and** $\max_d z_d \le z_{\max}$ | the joint test with the masking problem (below) explicitly capped |

Because $\Sigma$ is diagonal, $z^2$ is a sum of independent squared standard
normals under the model, hence **$z^2 \sim \chi^2_d$** — the $\chi^2$ quantile
is the exact $q$-ellipsoid of the component, not an approximation.

## 1.2 What the code does (formula by formula)

**Step 1 — best component** (pose posterior, state ignored):

$$
k^* = \arg\max_k \Big[ \ln w_k + \ln \mathcal{N}(v_p; \mu_k, \Sigma_k) \Big].
$$

**Step 2 — pose gate.** With $z_d = \frac{|v_{p,d} - \mu_{k^*,d}|}{\sigma_{k^*,d}}$
and $z^2 = \sum_d z_d^2$:

$$
\text{valid\_pose}
= \underbrace{\big(z \le \sqrt{\chi^2_q(d)}\big)}_{\text{joint } \chi^2 \text{ ellipsoid}}
\;\wedge\;
\underbrace{\big( \text{if } \texttt{z\_max\_sigma} \text{ set: } \max_d z_d \le \texttt{z\_max\_sigma} \big)}_{\text{per-dimension cap}} .
$$

**Step 3 — state gate** (hard equality with the component's most likely
state):

$$
\text{valid\_state} = \big(s = \arg\max_j \pi_{k^*,j}\big).
$$

**Result:** $\texttt{score\_single}(v) = \text{valid\_pose} \wedge
\text{valid\_state}$.

## 1.3 Why this method, and its known limitation

- **Why the joint $\chi^2$ test:** it is the textbook multivariate
  membership test (Hotelling $T^2$ / `EllipticEnvelope`), and — unlike the
  likelihood-ratio variant — it needs **no tunable cutoff**: $q$ is a
  statistical quantile (default $0.99$).
- **Why the per-dimension cap:** the joint ellipsoid is *not a box* — with
  $d=6$, $\sqrt{\chi^2_{0.99}(6)} \approx 3.81$, so one dimension can be
  $3.8\sigma$ off while five perfect dimensions mask it. `z_quantile_dim`
  (a two-sided normal quantile, converted via
  $c = \Phi^{-1}\big((1+q)/2\big)$; default $0.999 \Leftrightarrow 3.29\sigma$)
  explicitly forbids any single dimension from deviating more than that many
  $\sigma$ — the standard remedy for the masking problem of joint tests.
- **Why the hard state gate:** the state is discrete; a value whose state
  differs from the model's prediction is simply *not* in the condition (no
  partial credit).

**Limitation:** $q$ is fixed per entity; in very high $d$ the ellipsoid is
large, so the test is lenient unless `z_quantile_dim` is tightened.

---

# 2. `containment_score` — the agreement problem

**Problem.** Do two conditions $A$ and $B$ share a value that can occur under
*both*? (Symmetric — neither condition is the "target".) This gates whether a
subgoal connection between two skills is created.

## 2.1 Methods that exist

| method | definition | references / notes |
|---|---|---|
| **Bhattacharyya coefficient** | $\operatorname{BC}(A,B) = \int \sqrt{p_A p_B}$; closed form for Gaussians, factorizes over dims | [BC and Bhattacharyya distance](https://www.db-thueringen.de/servlets/MCRFileNodeServlet/dbt_derivate_00042924/Brehm-pdfa.pdf), [BC as a generalization of Mahalanobis](https://www.mdpi.com/2078-2489/14/3/164) — symmetric, graded; needs a cutoff |
| **KL divergence** | per-Gaussian KL is closed form; GMM KL has no closed form → approximations: matching (Goldberger), **variational**, unscented, MC | [Hershey & Olsen 2007](https://www.semanticscholar.org/paper/Approximating-the-Kullback-Leibler-Divergence-Hershey-Olsen/831780b12cb41a9905c3d4f58831a2ea6d09223b) — asymmetric (direction must be chosen), needs a cutoff |
| **Overlap coefficient (OVL)** | $\operatorname{OVL}(A,B) = \int \min(p_A, p_B)$ | [OVL.CI](https://cran.r-project.org/web/packages/OVL.CI/index.html), [LPS::OVL](https://search.r-project.org/CRAN/refmans/LPS/html/OVL.html), [covariate-specific OVL](https://www.math.bas.bg/~statlab/ISCPS_2025_Presentations/Christos%20Nakas.pdf) — the honest "shared mass", but no closed form for mixtures → numeric/MC |
| **Earth mover's distance** | optimal transport between the mixtures | a *metric*, not a "common value" statement; expensive |
| **ellipsoid intersection** | does $\exists x$ with $Q_A(x) \le c_A$ and $Q_B(x) \le c_B$? | geometric feasibility; exact test is convex but heavier |
| **symmetric percentile overlap (used)** | does a common value exist — **exact ellipsoid-intersection test** $\min_x \max(Q_1/c, Q_2/c) \le 1$ | this code — threshold-free, matches the "value in both" semantics (see below) |

## 2.2 What the code does (formula by formula)

**Step 1 — exact ellipsoid-intersection test.** For component $i$ of $A$ and
component $j$ of $B$, the pair agrees on the pose iff the two $q$-ellipsoids
intersect — i.e. iff there exists a point $x$ satisfying *both* quadratic
constraints

$$
Q_1(x) = \sum_{d'} \frac{(x_{d'} - \mu_{1,i,d'})^2}{\sigma^2_{1,i,d'}}
\le \chi^2_q(d)
\qquad\text{and}\qquad
Q_2(x) = \sum_{d'} \frac{(x_{d'} - \mu_{2,j,d'})^2}{\sigma^2_{2,j,d'}}
\le \chi^2_q(d),
$$

optionally restricted to the per-dimension cap $|x_{d'} - \mu_{k,d'}| \le
z_{\max}\,\sigma_{k,d'}$ for both $k$.

This is decided **exactly** by minimizing the convex function

$$
\min_x\; \max\!\Big(\tfrac{Q_1(x)}{\chi^2_q(d)},\ \tfrac{Q_2(x)}{\chi^2_q(d)}\Big)
\;\le\; 1
\;\;\Longleftrightarrow\;\;
E_1 \cap E_2 \ne \varnothing,
$$

over the axis-aligned box given by the intersection of the per-dimension
projections of both ellipsoids (and caps) — any common point must lie in that
box. Fast paths (one component center inside the other ellipsoid; disjoint
per-dimension projections) resolve most pairs without the optimizer.

**Step 2 — state gate** (hard equality of the most likely states, aligned to
the union of state spaces):

$$
\text{agrees}(i,j)
= \big(E_1 \cap E_2 \ne \varnothing\big)
\wedge \big(\arg\max \pi_{1,i} = \arg\max \pi_{2,j}\big).
$$

**Step 3 — covered mass.** The fraction of each side's weight that has at
least one agreeing partner, then the minimum:

$$
\text{covered}_1 = \sum_i w_{1,i}\,\mathbf{1}\big[\exists j:\ \text{agrees}(i,j)\big],
\qquad
\text{covered}_2 = \sum_j w_{2,j}\,\mathbf{1}\big[\exists i:\ \text{agrees}(i,j)\big],
$$

$$
\boxed{\;\texttt{containment\_score}(A,B) = \min\big(\text{covered}_1, \text{covered}_2\big)\;}
$$

## 2.3 Why this method, and its limitations

- **Why not BC / KL / OVL:** all three are graded overlap *magnitudes* that
  still need a hand-tuned cutoff to decide "compatible or not". The
  percentile-overlap formulation instead embeds the cutoff in the **statistical
  quantile** $q$: the decision is simply *"does any value lie in both
  $q$-ellipsoids"* ($\text{score} > 0$). It is symmetric (both conditions must
  produce the value) and threshold-free.
- **The covered-mass `min`** makes self-agreement $1.0$ by construction and
  avoids the $\sum_i w_i^2$ under-counting a plain pair-weight sum would have.
- **Why the exact intersection test instead of the earlier agreement-value
  test:** the previous version only checked whether the posterior mean
  $x^*_{ij}$ (the mode of the product, precision-weighted toward the smaller
  variance) lay inside both ellipsoids. When the variances differ a lot,
  $x^*$ could fall outside the wider ellipsoid even though the two ellipsoids
  still overlap elsewhere — the score then **under-reported** agreement and
  missed valid connections. The exact test answers the actual question — "does
  *any* common value exist?" — and removes that conservative bias: if the
  ellipsoids intersect anywhere, the pair agrees.
- **Consistency with the subgoal value:** the value put into the subgoal is
  still `best_sample`'s posterior mean $x^*$ — the most jointly plausible
  point. Whenever the ellipsoids intersect, that point lies in (or arbitrarily
  close to) the intersection region, so score and value stay consistent.

---

## 3. Configuration

| field | default | used in | role |
|---|---|---|---|
| `z_quantile_joint` | `0.99` | both | joint ellipsoid $z \le \sqrt{\chi^2_q(d)}$ — membership (`score_single`) and "value in both" (`containment`) |
| `z_quantile_dim` | `0.999` | both | per-dimension cap, two-sided normal quantile ($0.999 \Leftrightarrow 3.29\sigma$) |

---

## 4. How the two scores are used

- `score_single` gates whether a scene value is accepted as satisfying a
  condition (`Condition.test`, graph node updates).
- `containment_score` gates whether two conditions can be chained:
  `Condition.make_subgoal` creates a connection only if the score is `> 0` for
  every shared entity, and the subgoal value is `best_sample` (the agreement
  value $x^*$ of the winning component pair).

---

## 5. References

1. Mahalanobis, P. C. (1936). *On the generalised distance in statistics*.
2. [Hotelling's $T^2$ statistic (finite-sample multivariate inference)](https://www.statmod.org/smij/Vol13/Iss5&6/Romer/Abstract.html).
3. [Mahalanobis-distance / $\chi^2$ outlier detection (ASOD)](https://link.springer.com/article/10.1186/s13677-024-00682-0); [R `mvoutlier`](https://cran.r-universe.dev/mvoutlier/doc/manual.html); [scikit-learn `EllipticEnvelope`](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html).
4. Bhattacharyya, A. (1943). *On a measure of divergence between two statistical populations*; [Bhattacharyya distance / coefficient formulas](https://www.db-thueringen.de/servlets/MCRFileNodeServlet/dbt_derivate_00042924/Brehm-pdfa.pdf); [BC as generalization of Mahalanobis](https://www.mdpi.com/2078-2489/14/3/164).
5. [Hershey, J. R., & Olsen, P. A. (2007). *Approximating the Kullback–Leibler divergence between Gaussian mixture models*.](https://www.semanticscholar.org/paper/Approximating-the-Kullback-Leibler-Divergence-Hershey-Olsen/831780b12cb41a9905c3d4f58831a2ea6d09223b)
6. [Overlap coefficient (OVL)](https://www.math.bas.bg/~statlab/ISCPS_2025_Presentations/Christos%20Nakas.pdf); [OVL.CI](https://cran.r-project.org/web/packages/OVL.CI/index.html); [LPS::OVL](https://search.r-project.org/CRAN/refmans/LPS/html/OVL.html).
7. [GMM similarity measures / Gaussian mixture reduction](https://ieeexplore.ieee.org/search/searchresult.jsp?matchBoolean=true&queryText=%22Index%20Terms%22:Gaussian%20mixture%20reduction%20(GMR)&newsearch=true).
