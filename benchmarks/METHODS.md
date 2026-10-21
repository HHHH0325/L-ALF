# Methods, modules and sources

All methods use the finite-library oracle described in the benchmark README.
The production L-ALF reference is the main package, and its five historical
paper trajectories are not supplied or reconstructed here. `full_llm` and
`llm_beta0` are excluded because they were removed from the retained comparison.

## R1-7: nine module comparisons

| Method ID | Changed module | Retained modules | Source |
|---|---|---|---|
| `llm_gen_rf` | LLM property evaluator → RF | LLM target generation, MMR and paper ranking | Breiman 2001 |
| `llm_gen_xgb` | LLM evaluator → bootstrap XGBoost | Same | Chen & Guestrin 2016 |
| `llm_gen_gp` | LLM evaluator → fingerprint GP | Same | Srinivas et al. 2010, GP/UCB background |
| `llm_gen_graph` | LLM evaluator → GCN ensemble | Same | Kipf & Welling 2017, architecture background |
| `similarity_gen_llm` | LLM target generation → three observed anchors | MMR, LLM evaluator and paper ranking | Local implementation; ECFP background, Rogers & Hahn 2010 |
| `rule_gen_llm` | Target generation → valence-checked terminal substitutions or aromatic C/N swaps | Same | Local rule control |
| `evolution_gen_llm` | Target generation → six mutations per anchor, top three observed-fitness interpolation proposals | Same | Local mutation/selection control; no graph crossover |
| `llm_no_context` | Clear global/local evaluator references and simplify generation prompt | LLMs, retrieval and ranking | Local ablation; affects more than one prompt component |
| `llm_minimal_guidance` | Remove expert roles and explicit chemical edit heuristics | Numerical targets, reference-selection rules, trade-off focus, retrieval and ranking | Local ablation; `guidance_ablation.py` |

Observed anchors maximize f, ratio, and common score. Mutational proxy fitness
is a similarity-to-the-fourth-power weighted interpolation of queried-history
scores. It never queries labels for newly generated target structures.

The common comparison evaluator predicts S1, EST, f, ratio. RF has 96 trees,
`max_features=0.7`, `min_samples_leaf=2`. XGBoost uses three bootstrap members,
100 estimators, depth 4, learning rate 0.07, subsample 0.8 and column sampling
0.7. Independent regressors fit each property.

Fingerprint GP uses Morgan radius 2/2048 bits, PCA up to 128 components,
standardization, Matérn-2.5 plus white-noise 0.05, jitter 1e-6, no kernel
optimization, and a length scale equal to the median queried-training distance.
It produces 32 independent marginal ranking draws, not a joint batch acquisition.
PCA is rebuilt from structures only, with fixed feature seed 0; a small user
library uses fewer components. This replaces the historical external feature cache.

GCN uses three 48-dimensional normalized graph convolutions, mean pooling,
atom features without bond-type features, and three bootstrap ensemble members.
AdamW uses learning rate 0.002 and weight decay 1e-4, up to 80 epochs, a queried
validation split and 12 stale epochs for early stopping. The original validation
rule is retained for manuscript-size histories; small histories keep at least
one training row. These are graph regressors, not graph GP/BO.

Surrogates fit f and ratio after `log1p(max(value,0))`, then invert predictions.
Historical negative f labels are not rewritten. LLM evaluation makes three
batched responses at temperatures 0.15, 0.20 and 0.25. Each candidate receives
at most three global, two local-positive and four local-negative references.
Traditional surrogates fit all queried labels; this differs from LLM context
selection. These comparisons do not isolate intrinsic model-family superiority
under identical information representations. No prompt control removes a
model's pretraining or proves that its internal reasoning is disabled.

## R1-9: twelve whole-search baselines

| Method ID | Actual selection | Literature / implementation |
|---|---|---|
| `random` | Single uniform sample without replacement, sliced into 20-member batches | Original figure's random protocol; no surrogate |
| `similarity_search` | Maximum Morgan/Tanimoto similarity to three observed anchors, select top 20 | Local finite-library implementation; Rogers & Hahn 2010 |
| `evolutionary_search` | Top 40 observed parents; uniform bit crossover, four flipped bits, 15 children each mapped to three neighbors, five random immigrants; select by observed-fitness interpolation | Local fingerprint GA; distinct from molecular Graph-GA |
| `rf_search` | RF predict remaining library, common-score UCB prescreen 500, paper ranking select 20 | Breiman 2001; task-specific combination |
| `xgb_search` | Same search with bootstrap XGBoost | Chen & Guestrin 2016; task adaptation |
| `gp_search` | Same search with fingerprint GP | Srinivas et al. 2010 background; no transfer of its theoretical guarantees |
| `graph_search` | Same search with GCN ensemble | Kipf & Welling 2017 architecture background |
| `graph_gp_search` | Bond-labeled WL subtree kernel GP, same 500→20 search | Shervashidze et al. 2011; Griffiths et al. 2023 background; GAUCHE software not run |
| `graphga_search` | Jensen graph crossover/mutation; map offspring to 50 library candidates; uniform sample 20 | Jensen 2019, actual vendored operators; finite-library adaptation |
| `molleo_search` | LLM two-parent offspring, Graph-GA fallback for invalid structures, graph mutation, map 50 candidates, select 20 nearest to best observed parent | Wang et al. 2025 and Jensen 2019; paper-driven adaptation, not official MolLEO/PMO execution |
| `qpots` | Five GP models, prescreen 500, eight joint posterior draws, sampled feasible Pareto union capped at 80, maximin feature-distance batch of 20 | Renganathan & Carlson 2025; retained historical discrete-pool adaptation |
| `qlognparego` | Same five GP models/prescreen, random Chebyshev scalarization plus qLogNEI, eight MC samples, sequential greedy 20 with pending points | Knowles 2006; Daulton et al. 2020; Ament et al. 2023; BoTorch qLogNParEGO |

WL kernel counts three bond-labeled refinement iterations, L2-normalizes sparse
counts and uses their dot product. GP noise variance is 0.05 with jitter 1e-6,
fixed kernel parameters and 32 independent marginal ranking draws. The kernel
omits stereochemistry; this is not execution of the GAUCHE benchmark.

Graph-GA/MolLEO use up to 100 best queried parents, fitness-proportional distinct
parent sampling, 60 offspring attempts and mutation probability 0.067. Offspring
map to nearest unused library structures. If fewer than 50 map, best-parent
neighbors fill the panel; this is structural filling without extra labels.
MolLEO packs ten independent two-parent tasks per response. Invalid generated
structures may fall back to graph crossover; API/JSON failures cannot. Generated
offspring can differ from the structures actually queried. Neither method uses
the common Pareto/UCB rule for final selection.

MOBO represents two log1p nonnegative objectives and three constraint residuals
using five Matérn-2.5 SingleTaskGPs, outcome standardization and fixed observation
noise `(1e-4,1e-3,1e-4,1e-4,1e-4)`. EST uses `est-nextafter(0.30,-inf)` so equality
at 0.30 is infeasible. Other residuals are `1.3-s1` and `s1-3.2`. Fit cap is 20
SciPy iterations, with the retained Torch optimizer fallback for fitting failure.
Prescreen multiplies feasibility probability by the sum of nonnegative objective
UCBs, beta 0.5. qPOTS uses a discrete posterior Pareto union, not the original
paper's evolutionary optimization of posterior paths. qLogNParEGO is not an
exact reproduction of original ParEGO, and qEHVI was not added to the retained
method registry. Acquisition failure stops a run instead of silently changing
its selection rule.

## Paper ranking, budget and evaluation

Retrieved methods use 15/15/20 candidates for f/ratio/balance directions and
MMR lambda increasing linearly from 0.5 to 0.8 across 20 rounds. MMR follows
Carbonell & Goldstein 1998, adapted to molecular fingerprints. Every selected
batch has 20 distinct, unseen molecules; the first random batch counts toward
the total 400 queries. Methods may screen different candidate sets.

For methods using paper ranking, predicted feasibility precedes the first
Pareto front; other feasible candidates follow, then infeasible fillers.
Within each group the rule is mean common score + 0.6 times score deviation:

```text
H = 1 when 1.3 <= S1 <= 3.2 and EST < 0.30; otherwise 0
O = 3*clip(log1p(max(f,0))/log1p(0.3),0,1)
  + 3*clip(log1p(max(ratio,0))/log1p(70),0,1)
common_score = 4*H + O*(1 if H else 0.3)
```

This preserves the retained comparison's first-front rule instead of replacing
it with the production algorithm's successive Pareto layers. Tree/ensemble
deviation, LLM response variation and GP posterior SD have different statistical
meanings. No uncertainty calibration is claimed.

HV is computed only after acquisition from feasible `log1p(max(f,0))` and
`log1p(max(ratio,0))` objectives, normalized to the feasible reference-library
maxima, reference point `(0,0)`. Descriptor hits are computational thresholds,
not measured ISC rates, ROS yields or experimental photosensitizer performance.
Five historical L-ALF runs are not paired with new comparison seeds; report
their selection rule and run variability without inventing paired significance
or assuming identical first batches and total computation costs.

Full bibliography and stable links are in `references.bib`. Method registry keys
match those citations; local controls without a specific source are identified
as local implementations.
