"""Inter-rater agreement panel for the 100-pair benchmark.

Computes, on the same 98 pairwise-complete pairs, a panel of estimators so the
choice of metric is transparent rather than cherry-picked:

  - Observed agreement (exact, within-1)
  - Cohen's kappa: unweighted, linear-, quadratic-weighted
  - PABAK (prevalence-adjusted bias-adjusted kappa)
  - Gwet's AC1 (nominal) and AC2 (linear / quadratic ordinal weights)
  - Krippendorff's alpha (nominal + ordinal; uses ALL data, incl. the 2 pairs
    Rater 1 left blank, via its native missing-data handling)
  - ICC two-way single-rater: ICC(2,1) absolute agreement & ICC(3,1) consistency

All estimators implemented from first principles (no pingouin/krippendorff dep).

Run: python3 agreement_panel.py
"""
import numpy as np
import compute_kappa_complete as base

Q = 5  # rating scale 0..4


# ----------------------------------------------------------- weight matrices
def agree_weights(q, kind):
    W = np.zeros((q, q))
    for i in range(q):
        for j in range(q):
            d = abs(i - j) / (q - 1)
            if kind == "identity":
                W[i, j] = 1.0 if i == j else 0.0
            elif kind == "linear":
                W[i, j] = 1.0 - d
            elif kind == "quadratic":
                W[i, j] = 1.0 - d * d
    return W


# --------------------------------------------------------------- estimators
def confusion(a, b, q):
    cm = np.zeros((q, q))
    for x, y in zip(a, b):
        cm[x, y] += 1
    return cm


def cohen(cm, W):
    n = cm.sum()
    p = cm / n
    r, c = p.sum(1), p.sum(0)
    po = (W * p).sum()
    pe = (W * np.outer(r, c)).sum()
    return (po - pe) / (1 - pe)


def pabak(cm):
    q = cm.shape[0]
    po = np.trace(cm) / cm.sum()
    return (q * po - 1) / (q - 1)


def gwet(cm, W):
    n = cm.sum()
    p = cm / n
    q = cm.shape[0]
    pa = (W * p).sum()
    pi = (p.sum(1) + p.sum(0)) / 2.0          # mean marginal per category
    Tw = W.sum()
    pe = (Tw / (q * (q - 1))) * (pi * (1 - pi)).sum()
    return (pa - pe) / (1 - pe)


def krippendorff(units, level, labels):
    """units: list of tuples of ratings per item (1 or 2 values, missing dropped).
    level: 'nominal' or 'ordinal'. labels: ordered list of possible values."""
    L = list(labels)
    idx = {v: i for i, v in enumerate(L)}
    q = len(L)
    o = np.zeros((q, q))
    for vals in units:
        vals = [v for v in vals if v is not None]
        m = len(vals)
        if m < 2:
            continue
        for x in vals:
            for y in vals:
                if x is not y or vals.count(x) > 1:
                    pass
        # pairable: each ordered pair contributes 1/(m-1)
        for ix, x in enumerate(vals):
            for jy, y in enumerate(vals):
                if ix != jy:
                    o[idx[x], idx[y]] += 1.0 / (m - 1)
    nc = o.sum(1)
    n = nc.sum()
    if n < 2:
        return float("nan")

    def delta2(ci, ki):
        if level == "nominal":
            return 0.0 if ci == ki else 1.0
        lo, hi = (ci, ki) if ci <= ki else (ki, ci)
        s = nc[lo:hi + 1].sum() - (nc[ci] + nc[ki]) / 2.0
        return s * s

    Do = sum(o[c, k] * delta2(c, k) for c in range(q) for k in range(q)) / n
    De = sum(nc[c] * nc[k] * delta2(c, k)
             for c in range(q) for k in range(q)) / (n * (n - 1))
    return 1 - Do / De if De > 0 else float("nan")


def icc_two_way(a, b):
    """Returns ICC(2,1) absolute agreement and ICC(3,1) consistency."""
    X = np.array([a, b], dtype=float).T  # n x k, k=2
    n, k = X.shape
    grand = X.mean()
    row = X.mean(1)
    col = X.mean(0)
    SSR = k * ((row - grand) ** 2).sum()
    SSC = n * ((col - grand) ** 2).sum()
    SST = ((X - grand) ** 2).sum()
    SSE = SST - SSR - SSC
    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    icc_abs = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)
    icc_con = (MSR - MSE) / (MSR + (k - 1) * MSE)
    return icc_abs, icc_con


# -------------------------------------------------------------------- main
def main():
    r1 = base.rater1_by_pair()
    r2 = base.rater2_by_pair()
    ids = sorted(set(r1) & set(r2))

    # complete pairs (rating)
    a = [r1[p]["rating"] for p in ids
         if r1[p]["rating"] is not None and r2[p]["rating"] is not None]
    b = [r2[p]["rating"] for p in ids
         if r1[p]["rating"] is not None and r2[p]["rating"] is not None]
    cm = confusion(a, b, Q)
    n = len(a)

    Wlin = agree_weights(Q, "linear")
    Wquad = agree_weights(Q, "quadratic")
    Wid = agree_weights(Q, "identity")

    po = np.trace(cm) / n
    within1 = np.mean([abs(x - y) <= 1 for x, y in zip(a, b)])

    print(f"RATING (0-4)  —  pairwise-complete n = {n}")
    print(f"  Observed exact agreement      : {po*100:5.1f}%")
    print(f"  Observed within-1 agreement   : {within1*100:5.1f}%")
    print("  ---- chance-corrected ----")
    print(f"  Cohen kappa (unweighted)      : {cohen(cm, Wid):+.3f}")
    print(f"  Cohen kappa (linear wt)       : {cohen(cm, Wlin):+.3f}")
    print(f"  Cohen kappa (quadratic wt)    : {cohen(cm, Wquad):+.3f}")
    print(f"  PABAK (nominal, q=5)          : {pabak(cm):+.3f}")
    print(f"  Gwet AC1 (unweighted)         : {gwet(cm, Wid):+.3f}")
    print(f"  Gwet AC2 (linear wt)          : {gwet(cm, Wlin):+.3f}")
    print(f"  Gwet AC2 (quadratic wt)       : {gwet(cm, Wquad):+.3f}")

    # Krippendorff on ALL pairs incl. the 2 R1-blanks (native missing handling)
    units = [(r1[p]["rating"], r2[p]["rating"]) for p in ids]
    print(f"  Krippendorff alpha (nominal)  : "
          f"{krippendorff(units, 'nominal', range(Q)):+.3f}")
    print(f"  Krippendorff alpha (ordinal)  : "
          f"{krippendorff(units, 'ordinal', range(Q)):+.3f}   "
          f"(uses all {sum(1 for u in units if u[0] is not None and u[1] is not None)}+"
          f"{sum(1 for u in units if u[0] is None or u[1] is None)} pairs)")
    icc_abs, icc_con = icc_two_way(a, b)
    print(f"  ICC(2,1) absolute agreement   : {icc_abs:+.3f}")
    print(f"  ICC(3,1) consistency          : {icc_con:+.3f}")

    # ---- methodology (3-cat) ----
    lab = ["no", "partial", "yes"]
    li = {v: i for i, v in enumerate(lab)}
    ma = [li[r1[p]["method"]] for p in ids
          if r1[p]["method"] is not None and r2[p]["method"] is not None]
    mb = [li[r2[p]["method"]] for p in ids
          if r1[p]["method"] is not None and r2[p]["method"] is not None]
    mcm = confusion(ma, mb, 3)
    Wid3 = agree_weights(3, "identity")
    Wlin3 = agree_weights(3, "linear")
    print(f"\nMETHODOLOGY (no/partial/yes)  —  n = {len(ma)}")
    print(f"  Observed exact agreement      : {np.trace(mcm)/len(ma)*100:5.1f}%")
    print(f"  Cohen kappa (unweighted)      : {cohen(mcm, Wid3):+.3f}")
    print(f"  Cohen kappa (linear/ordinal)  : {cohen(mcm, Wlin3):+.3f}")
    print(f"  PABAK (nominal, q=3)          : {pabak(mcm):+.3f}")
    print(f"  Gwet AC1 (unweighted)         : {gwet(mcm, Wid3):+.3f}")
    munits = [(r1[p]["method"], r2[p]["method"]) for p in ids]
    munits = [(None if x is None else li[x], None if y is None else li[y])
              for x, y in munits]
    print(f"  Krippendorff alpha (ordinal)  : "
          f"{krippendorff(munits, 'ordinal', range(3)):+.3f}")


if __name__ == "__main__":
    main()
