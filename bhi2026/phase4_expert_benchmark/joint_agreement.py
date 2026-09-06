"""How related are the rating and methodology columns, and what does combining
them do to inter-rater agreement?

Outputs:
  1. Within-rater coupling  (rating x methodology cross-tab, Spearman, how much
     methodology is determined by rating; where methodology adds info).
  2. Joint inter-rater agreement criteria (both-exact, rating-within-1 & method).
  3. Multivariate Krippendorff alpha on the (rating, methodology) vector, with a
     sweep over how much weight the method axis gets (lambda).
"""
import numpy as np
from collections import Counter
import compute_kappa_complete as base

M = {"no": 0, "partial": 1, "yes": 2}

r1 = base.rater1_by_pair()
r2 = base.rater2_by_pair()
ids = sorted(set(r1) & set(r2))


def vec(rec):
    if rec["rating"] is None or rec["method"] is None:
        return None
    return (rec["rating"], M[rec["method"]])


# ---------- 1. within-rater coupling ----------
print("=" * 64)
print("1. WITHIN-RATER COUPLING  (is methodology already inside rating?)")
print("=" * 64)
for name, rr in [("Rater 1", r1), ("Rater 2", r2)]:
    rows = [(rr[p]["rating"], rr[p]["method"]) for p in ids
            if rr[p]["rating"] is not None and rr[p]["method"] is not None]
    print(f"\n{name}  (n={len(rows)})  rating x methodology:")
    print("   rating |  no  partial  yes")
    for rt in range(5):
        c = Counter(m for r, m in rows if r == rt)
        if sum(c.values()) == 0:
            continue
        print(f"     {rt}    | {c.get('no',0):>3}   {c.get('partial',0):>4}  {c.get('yes',0):>4}")
    # Spearman rating vs method(0/1/2)
    rv = np.array([r for r, _ in rows], float)
    mv = np.array([M[m] for _, m in rows], float)
    from scipy.stats import spearmanr
    rho, p = spearmanr(rv, mv)
    print(f"   Spearman(rating, method) = {rho:+.3f}  (p={p:.1e})")
    # how determined is method by rating? conditional entropy H(method|rating)/H(method)
    def H(counts):
        tot = sum(counts);
        return -sum((c/tot)*np.log2(c/tot) for c in counts if c > 0) if tot else 0
    Hm = H(list(Counter(m for _, m in rows).values()))
    Hcond = 0.0
    for rt in range(5):
        sub = [m for r, m in rows if r == rt]
        if sub:
            Hcond += (len(sub)/len(rows)) * H(list(Counter(sub).values()))
    print(f"   H(method)={Hm:.3f} bits;  H(method|rating)={Hcond:.3f} bits;  "
          f"methodology is {100*(1-Hcond/Hm):.0f}% determined by rating")

# ---------- 2. joint inter-rater agreement ----------
print("\n" + "=" * 64)
print("2. JOINT INTER-RATER AGREEMENT  (do raters agree on BOTH axes?)")
print("=" * 64)
both = [(vec(r1[p]), vec(r2[p])) for p in ids
        if vec(r1[p]) is not None and vec(r2[p]) is not None]
n = len(both)
rating_exact = np.mean([a[0] == b[0] for a, b in both])
rating_w1 = np.mean([abs(a[0]-b[0]) <= 1 for a, b in both])
method_exact = np.mean([a[1] == b[1] for a, b in both])
joint_exact = np.mean([a == b for a, b in both])
joint_w1m = np.mean([abs(a[0]-b[0]) <= 1 and a[1] == b[1] for a, b in both])
print(f"  n = {n}")
print(f"  rating exact .................... {rating_exact*100:5.1f}%")
print(f"  methodology exact .............. {method_exact*100:5.1f}%")
print(f"  BOTH exact (rating & method) ... {joint_exact*100:5.1f}%")
print(f"  rating within-1 AND method exact {joint_w1m*100:5.1f}%")
# where rating disagrees, does method also disagree?
rd = [(a, b) for a, b in both if a[0] != b[0]]
md_given_rd = np.mean([a[1] != b[1] for a, b in rd]) if rd else float('nan')
ra = [(a, b) for a, b in both if a[0] == b[0]]
md_given_ra = np.mean([a[1] != b[1] for a, b in ra]) if ra else float('nan')
print(f"  P(method disagrees | rating disagrees) = {md_given_rd*100:.1f}%  (n={len(rd)})")
print(f"  P(method disagrees | rating agrees)    = {md_given_ra*100:.1f}%  (n={len(ra)})")

# ---------- 3. multivariate Krippendorff alpha ----------
print("\n" + "=" * 64)
print("3. MULTIVARIATE KRIPPENDORFF alpha on (rating, methodology)")
print("=" * 64)


def kripp_general(units, values, distfn):
    idx = {v: i for i, v in enumerate(values)}
    q = len(values)
    o = np.zeros((q, q))
    for u in units:
        present = [v for v in u if v is not None]
        m = len(present)
        if m < 2:
            continue
        for ix in range(m):
            for jx in range(m):
                if ix != jx:
                    o[idx[present[ix]], idx[present[jx]]] += 1.0/(m-1)
    nc = o.sum(1)
    N = nc.sum()
    D = np.array([[distfn(a, b) for b in values] for a in values])
    Do = (o * D).sum() / N
    De = (np.outer(nc, nc) * D).sum() / (N*(N-1))
    return 1 - Do/De if De > 0 else float('nan')


units = [(vec(r1[p]), vec(r2[p])) for p in ids]
values = sorted({v for u in units for v in u if v is not None})


def dist_lambda(lam):
    # normalized ordinal distance on each axis, squared, weighted
    def d(a, b):
        dr = abs(a[0]-b[0]) / 4.0
        dm = abs(a[1]-b[1]) / 2.0
        return (1-lam)*dr*dr + lam*dm*dm
    return d


print("  lambda  (0=rating only ... 1=method only)   alpha")
for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
    a = kripp_general(units, values, dist_lambda(lam))
    tag = "  <- rating-only (sanity vs 0.430)" if lam == 0 else (
          "  <- method-only" if lam == 1 else "")
    print(f"    {lam:>4.2f}                                    {a:+.3f}{tag}")
print("\n  Equal-weight combined (lambda=0.5) is the natural 'joint reliability'")
print("  of the full pair characterization.")
