# xRFM Algorithm

xRFM (tree-structured Recursive Feature Machine) combines kernel ridge regression with supervised tree partitioning so that each region of the feature space learns its own kernel-adapted representation. The sections below describe every component necessary to implement the method without referencing external documents.

## Preliminaries

- **Dataset notation**: let `X ∈ ℝⁿˣᵈ` be training inputs with rows `x^(i)` and `y ∈ ℝⁿˣᶜ` the corresponding targets. Validation data `(X_val, y_val)` is available for hyperparameter selection inside each leaf.
- **Kernel ridge regression**: given a positive semi-definite kernel `K`, the predictor trained on `(X, y)` is  
  `f(x) = K(x, X) α`, where the coefficients solve `(K(X, X) + λ I) α = y`. Brute-force training scales super-quadratically with `n`.
- **Average Gradient Outer Product (AGOP)**: for a differentiable predictor `f`,  
  `AGOP(f, S) = (1 / |S|) Σ_{x ∈ S} ∇f(x) ∇f(x)ᵀ`.  
  Directions with large AGOP eigenvalues correspond to coordinates along which the predictor varies strongly.
- **Recursive Feature Machines (RFM)**: iteratively fit a predictor, build an AGOP-based feature transform, and refit. Kernel-RFM adapts the kernel to data but still trains on the full dataset each iteration and cannot specialize to local structure.

xRFM addresses these limitations by (a) refining the kernel-RFM procedure inside bounded-size leaves and (b) using supervised tree splits so that each leaf captures a coherent subset of the training distribution.

## Leaf RFM

Each leaf trains a modified kernel-RFM on the samples assigned to it. The procedure adapts both the kernel bandwidth and the feature transform learned from the AGOP.

- **Kernel family**: `K_{p,q}(x, z) = exp(-‖x - z‖ₚᵩ / Lᵩ)` with `0 < q ≤ p ≤ 2`, allowing different distance exponents for tabular data.
- **Feature transform**: maintain a positive semi-definite matrix `M_t` that scales coordinates either diagonally (`use_diag = true`) or via the full matrix. Replacing `x` with `M_t^{1/2} x` adapts the kernel to the learned feature subspace.
- **Bandwidth adaptation**: optionally rescale the kernel bandwidth `L` per leaf using a heuristic search routine.
- **Gradient handling**: when forming the AGOP, omit self-kernel derivatives to stabilize gradients for kernels that are non-differentiable at zero distance (e.g., Laplace).

### Pseudocode: LeafRFM

```
LeafRFM(X, y, X_val, y_val, τ, λ, ε, use_diag, adapt_bandwidth):
    M₀ ← I_d
    if adapt_bandwidth:
        L ← AdaptBandwidth(X)
    for t = 0 … τ - 1:
        if use_diag:
            X_M ← X ⊙ sqrt(diag(M_t))      # element-wise scaling
            α_t ← SolveKRR(X_M, y, λ, L, p, q)
            f_t(x) = K(x ⊙ sqrt(diag(M_t)), X_M) α_t
        else:
            X_M ← X M_t^{1/2}
            α_t ← SolveKRR(X_M, y, λ, L, p, q)
            f_t(x) = K(M_t^{1/2} x, X_M) α_t
        E_t ← ValidationError(f_t, X_val, y_val)
        G_t ← (1 / |X|) Σ_{i=1}^n ∇f_t(x^(i)) ∇f_t(x^(i))ᵀ
        M_{t+1} ← G_t / (ε + max_{i,j} G_t[i, j])   # entry-wise normalization
    Choose t* with minimum E_t
    return α_{t*}, M_{t*}, L, use_diag
```

The returned quadruple is stored in the leaf for later inference.

## Tree-Based Partitioning

The tree stratifies the dataset using supervised directions derived from AGOP eigenvectors. Internal nodes store both the splitting direction and the median threshold.

### Pseudocode: TreePartition

```
TreePartition(D, TreeHyp):
    # D holds all (x, y) pairs available at the node
    if |D| ≤ L_max:
        return LeafNode(data=D)
    Sample N examples S ⊂ D
    α_split, M_split, L_split, use_diag ← LeafRFM(S_X, S_y, X_val=∅, y_val=∅,
                                                   τ=1, λ_split, ε, use_diag_split,
                                                   adapt_bandwidth_split)
    v ← TopEigenvector(AGOP from split model)
    project each (x, y) ∈ D onto s = vᵀ x
    m ← median of all projections s
    D_left  ← {(x, y) ∈ D : vᵀ x ≤ m}
    D_right ← {(x, y) ∈ D : vᵀ x >  m}
    left_child  ← TreePartition(D_left,  TreeHyp)
    right_child ← TreePartition(D_right, TreeHyp)
    return InternalNode(vector=v, threshold=m,
                        children=(left_child, right_child))
```

Key properties:

- The split model uses a lightweight (often single-iteration) LeafRFM to discover a direction that matters for prediction.
- Splitting at the median guarantees both children contain at most `⌈|D| / 2⌉` points, ensuring balanced depth.
- Using supervised directions groups points by how the current model’s predictions change, which outperforms unsupervised projections on the benchmarks considered by the authors.

## Global Training Procedure

Once the tree structure is fixed, each leaf is trained on its local subset (with its own validation data).

### Pseudocode: xRFM-fit

```
xRFM-fit(D_train, D_val, TreeHyp, LeafHyp):
    T ← TreePartition(D_train, TreeHyp)
    for leaf ℓ in Leaves(T):
        D_train^ℓ ← points from D_train routed to ℓ
        D_val^ℓ   ← {(x, y) ∈ D_val : Route(x, T) = ℓ}
        α_ℓ, M_ℓ, L_ℓ, use_diag_ℓ ← LeafRFM(D_train^ℓ.X, D_train^ℓ.y,
                                             D_val^ℓ.X,   D_val^ℓ.y,
                                             LeafHyp.τ, LeafHyp.λ_leaf,
                                             LeafHyp.ε, LeafHyp.use_diag,
                                             LeafHyp.adapt_bandwidth)
        store (α_ℓ, M_ℓ, L_ℓ, use_diag_ℓ) in ℓ
    return T
```

- If a leaf contains too little validation data, reserve a subset of its training points for validation so that `LeafRFM` can select hyperparameters reliably.
- Because each leaf holds at most `L_max` samples, the overall training cost is `O(n log n)` (logarithmic number of levels times linear work per point).

### Routing Helper

```
Route(x, T):
    node ← T.root
    while node is internal:
        if node.vᵀ x ≤ node.threshold:
            node ← node.left_child
        else:
            node ← node.right_child
    return node
```

## Prediction Procedure

### Pseudocode: xRFM-predict

```
xRFM-predict(T, x):
    ℓ ← Route(x, T)
    α_ℓ, M_ℓ, L_ℓ, use_diag_ℓ ← parameters stored in ℓ
    if use_diag_ℓ:
        x_M ← x ⊙ sqrt(diag(M_ℓ))
        X_support ← ℓ.X_train ⊙ sqrt(diag(M_ℓ))
    else:
        x_M ← M_ℓ^{1/2} x
        X_support ← ℓ.X_train M_ℓ^{1/2}
    return K_{p,q}(x_M, X_support; L_ℓ) α_ℓ
```

Prediction requires `O(log n)` comparisons to find the leaf plus one kernel evaluation against the support points stored in that leaf.

### Soft Routing (Optional)

When a global split temperature `T > 0` is provided, xRFM replaces hard routing with a softened mixture over every leaf in the tree. Each internal node stores a splitting vector `v_j`, threshold `b_j`, and optional temperature scaling factor `σ_j`. For an input matrix `X`, let `ℒ` be the leaves in deterministic order and `Path(ℓ)` the ordered list of internal nodes visited on the way to leaf `ℓ`, paired with a boolean indicating whether the left branch was taken.

```
SoftRoutePredict(T, X):
    for each internal node j:
        logits_j ← (X v_j) - b_j
        τ_j ← T · σ_j                  # σ_j defaults to 1
        z_j ← logits_j / τ_j
    for each leaf ℓ ∈ ℒ:
        log_p_ℓ ← 0
        for (j, went_left) in Path(ℓ):
            if went_left:
                log_p_ℓ ← log_p_ℓ + log σ(-z_j)
            else:
                log_p_ℓ ← log_p_ℓ + log σ(z_j)
    log_p ← stack(log_p_ℓ over ℓ)      # shape (n_samples, |ℒ|)
    w ← softmax(log_p, dim=1)          # normalized mixture weights
    for each leaf ℓ:
        ŷ_ℓ ← LeafPredict(ℓ, X)        # regression or probability vector
    return Σ_{ℓ} w[:, ℓ] ⊙ ŷ_ℓ
```

The softmax is implemented stably by subtracting per-row maxima before exponentiation. Leaf predictions reuse the standard `LeafPredict` routine. Setting `T → 0⁺` sharpens the distribution toward the single most probable leaf, while larger `T` averages over many leaves and improves robustness on ambiguous samples.

## Practical Notes

- **Hyperparameters**: `TreeHyp` typically includes the sample size `N`, maximum leaf size `L_max`, and ridge parameter `λ_split` for the split model. `LeafHyp` covers the number of RFM iterations `τ`, the ridge penalty `λ_leaf`, the normalization constant `ε`, and booleans controlling diagonal AGOP usage and bandwidth adaptation.
- **Categorical variables**: implementation-specific optimizations pre-process categorical columns so that leaf RFMs can handle them efficiently before kernel evaluation.
- **Local feature learning**: tree partitioning lets different leaves specialize in disjoint feature subsets (e.g., different coordinates matter depending on the initial split direction), which vanilla kernel-RFM cannot disentangle.
- **Complexity profile**: training scales as `O(n log n)` due to balanced splitting; inference scales as `O(log n)` plus the cost of a single leaf-level kernel prediction.

Together, these steps define the full xRFM algorithm: construct a supervised median-split tree, train enhanced kernel-RFMs inside leaves, and route test points through the tree to query the appropriate leaf predictor.
