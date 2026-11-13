"""
ML-Based Automatic Digital Filter Design  (low-pass FIR)
=========================================================
Pipeline
  1. Dataset  : random specs (wp, ws, Ap, As) -> reference designs
                 - Parks-McClellan (remez) minimum order   [Rabiner & Herrmann 1973]
                 - Kaiser window design (window method)    [Kaiser 1974]
  2. Neural regression
        Model A : specs -> minimum filter ORDER   (compare vs Kaiser formula)
        Model B : specs -> filter COEFFICIENTS    (compare vs reference, judged in
                  the FREQUENCY domain as in Jou 1990s / BiasNet 2021)
  3. Bayesian optimisation (GP + Expected Improvement)
        searches (order, Kaiser beta) to minimise computational cost subject to
        passband-ripple / stopband-attenuation constraints
  4. Pareto study : order vs ripple vs attenuation vs cost

Frequencies are normalised so that 1.0 = Nyquist.
Run:  python ml_filter_design.py
"""
import os, json
import numpy as np
from scipy import signal
from scipy.stats import norm
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
OUT = "results"; os.makedirs(OUT, exist_ok=True)
MAXTAPS = 101            # coefficient vector length for Model B (odd, symmetric)
HALF = MAXTAPS // 2 + 1  # we predict only h[0..50]; the rest follows by symmetry
GRID = 1024              # frequency grid for evaluating responses


# ------------------------------------------------------------------ helpers
def sample_specs(n):
    wp = rng.uniform(0.10, 0.60, n)
    dw = rng.uniform(0.06, 0.25, n)           # transition width
    Ap = rng.uniform(0.05, 1.00, n)           # peak passband ripple  (dB)
    As = rng.uniform(40.0, 80.0, n)           # stopband attenuation  (dB)
    return np.c_[wp, wp + dw, Ap, As]


def deltas(Ap, As):
    return 10 ** (Ap / 20) - 1, 10 ** (-As / 20)


def measure(h, wp, ws):
    """achieved passband ripple (dB, peak) and stopband attenuation (dB)"""
    w, H = signal.freqz(h, worN=GRID)
    w = w / np.pi
    mag = np.abs(H)
    pb = mag[w <= wp]; sb = mag[w >= ws]
    ripple = np.max(np.abs(20 * np.log10(np.maximum(pb, 1e-12))))
    atten = -20 * np.log10(np.max(sb) + 1e-12)
    return ripple, atten


def meets(h, spec):
    r, a = measure(h, spec[0], spec[1])
    return r <= spec[2] and a >= spec[3]


def remez_design(L, spec):
    wp, ws, Ap, As = spec
    dp, ds = deltas(Ap, As)
    return signal.remez(L, [0, wp / 2, ws / 2, 0.5], [1, 0], weight=[1 / dp, 1 / ds], fs=1.0)


def remez_min_order(spec, start):
    """smallest odd length (Type-I) that meets the spec -- the 'optimal' label"""
    L = max(5, start | 1)
    if meets(remez_design(L, spec), spec):       # go down while it still works
        while L - 2 >= 5 and meets(remez_design(L - 2, spec), spec):
            L -= 2
        return L - 1
    for _ in range(60):                          # otherwise go up
        L += 2
        if meets(remez_design(L, spec), spec):
            return L - 1
    return None


def kaiser_design(spec, L=None, beta=None):
    wp, ws, Ap, As = spec
    dp, ds = deltas(Ap, As)
    if L is None:
        L, beta = signal.kaiserord(-20 * np.log10(min(dp, ds)), ws - wp)  # Kaiser formula
        L |= 1
    return signal.firwin(L, (wp + ws) / 2, window=("kaiser", beta)), L


def pad_half(h):
    """centre-pad to MAXTAPS, return right half (symmetry => full filter)"""
    full = np.zeros(MAXTAPS)
    s = (MAXTAPS - len(h)) // 2
    full[s:s + len(h)] = h
    return full[MAXTAPS // 2:]


def unpad_full(half):
    return np.r_[half[:0:-1], half]


# ------------------------------------------------------------------ 1. dataset
BETAS = np.arange(0.5, 11.01, 0.5)


def kaiser_min_design(spec, L_start):
    """window method: smallest odd length L (and best beta) that meets the spec"""
    wp, ws, Ap, As = spec
    fc = (wp + ws) / 2
    for L in range(max(5, L_start | 1), MAXTAPS + 1, 2):
        best = None
        for b in BETAS:
            r, a = measure(signal.firwin(L, fc, window=("kaiser", b)), wp, ws)
            if r <= Ap and a >= As:
                margin = a - As
                if best is None or margin > best[1]:
                    best = (b, margin)
        if best is not None:
            return L, best[0]
    return None


def build_dataset(n=700):
    X, N_opt, N_kaiser, N_win, B_win, Hhalf = [], [], [], [], [], []
    for spec in sample_specs(n * 3):
        hk, Lk = kaiser_design(spec)
        if Lk > MAXTAPS:
            continue
        no = remez_min_order(spec, Lk)
        kw = kaiser_min_design(spec, no + 1) if no is not None else None
        if no is None or kw is None:
            continue
        X.append(spec); N_opt.append(no); N_kaiser.append(Lk - 1)
        N_win.append(kw[0] - 1); B_win.append(kw[1]); Hhalf.append(pad_half(hk))
        if len(X) == n:
            break
    return (np.array(X), np.array(N_opt), np.array(N_kaiser),
            np.array(N_win), np.array(B_win), np.array(Hhalf))


def features(X):
    wp, ws, Ap, As = X.T
    dw = ws - wp
    dp, ds = deltas(Ap, As)
    # physics-inspired features (Kaiser/Bellanger: N ~ -log(sqrt(dp*ds)) / dw)
    return np.c_[wp, ws, dw, 1 / dw, Ap, As, -np.log10(dp), -np.log10(ds),
                 -np.log10(np.sqrt(dp * ds)) / dw]


# ------------------------------------------------------------------ 2. neural regression
def train_order_model(X, N_opt, N_k):
    F = features(X)
    Ftr, Fte, ytr, yte, ktr, kte = train_test_split(F, N_opt, N_k, test_size=0.2, random_state=1)
    sc = StandardScaler().fit(Ftr)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000, early_stopping=True, random_state=0)
    mlp.fit(sc.transform(Ftr), ytr)
    pred = mlp.predict(sc.transform(Fte))
    res = dict(
        mlp_MAE=mean_absolute_error(yte, pred), mlp_R2=r2_score(yte, pred),
        kaiser_MAE=mean_absolute_error(yte, kte), kaiser_R2=r2_score(yte, kte),
        kaiser_mean_overdesign=float(np.mean(kte - yte)))
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for a, p, t in zip(ax, [pred, kte], ["MLP (neural regression)", "Kaiser formula"]):
        a.scatter(yte, p, s=8); a.plot([yte.min(), yte.max()], [yte.min(), yte.max()], "r--")
        a.set_xlabel("optimal order (Parks-McClellan)"); a.set_ylabel("predicted order"); a.set_title(t)
    plt.tight_layout(); plt.savefig(f"{OUT}/order_prediction.png", dpi=130); plt.close()
    return mlp, sc, res


def train_coeff_model(X, H):
    F = features(X)
    Ftr, Fte, Htr, Hte, Xtr, Xte = train_test_split(F, H, X, test_size=0.2, random_state=2)
    sc = StandardScaler().fit(Ftr); ys = StandardScaler().fit(Htr)
    mlp = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=6000, tol=1e-7, n_iter_no_change=60, random_state=0)
    mlp.fit(sc.transform(Ftr), ys.transform(Htr))
    Hp = ys.inverse_transform(mlp.predict(sc.transform(Fte)))
    ok, d_rip, d_att = 0, [], []
    for hp, ht, s in zip(Hp, Hte, Xte):
        rp, ap = measure(unpad_full(hp), s[0], s[1])
        rt, at = measure(unpad_full(ht), s[0], s[1])
        ok += (rp <= s[2]) and (ap >= s[3])
        d_rip.append(rp - rt); d_att.append(ap - at)
    res = dict(coeff_MSE=float(np.mean((Hp - Hte) ** 2)),
               spec_pass_rate=ok / len(Xte),
               mean_ripple_gap_dB=float(np.mean(d_rip)),
               mean_atten_gap_dB=float(np.mean(d_att)))
    # example plot
    i = int(np.argmax(Xte[:, 3] > 55))
    s = Xte[i]
    fig, ax = plt.subplots(figsize=(7, 4))
    for h, lab in [(unpad_full(Hte[i]), "reference (Kaiser)"), (unpad_full(Hp[i]), "ML predicted")]:
        w, H_ = signal.freqz(h, worN=GRID)
        ax.plot(w / np.pi, 20 * np.log10(np.abs(H_) + 1e-12), label=lab)
    ax.axvline(s[0], c="g", ls=":"); ax.axvline(s[1], c="g", ls=":")
    ax.axhline(-s[3], c="r", ls="--", label=f"required As={s[3]:.0f} dB")
    ax.set_ylim(-110, 5); ax.set_xlabel("normalised freq (x pi rad)"); ax.set_ylabel("|H| (dB)")
    ax.legend(); ax.set_title("Direct coefficient regression (baseline)")
    plt.tight_layout(); plt.savefig(f"{OUT}/coeff_model_response.png", dpi=130); plt.close()
    return res


def train_window_param_model(X, N_win, B_win, N_k, N_opt):
    """Model B2 (hybrid): neural regression predicts window-method parameters
    (length, Kaiser beta); the coefficients themselves come from the window formula,
    so the frequency-domain structure is guaranteed."""
    F = features(X); Y = np.c_[N_win, B_win]
    Ftr, Fte, Ytr, Yte, Xtr, Xte, ktr, kte, otr, ote = train_test_split(
        F, Y, X, N_k, N_opt, test_size=0.2, random_state=3)
    sc = StandardScaler().fit(Ftr); ys = StandardScaler().fit(Ytr)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000, early_stopping=True, random_state=0)
    mlp.fit(sc.transform(Ftr), ys.transform(Ytr))
    P = ys.inverse_transform(mlp.predict(sc.transform(Fte)))
    res = {}
    for margin in [0, 2, 4]:                           # safety margin in taps
        ok, orders = 0, []
        for (n, b), s in zip(P, Xte):
            L = (int(round(n)) + margin) | 1 if int(round(n)) % 2 == 0 else int(round(n)) + 1 + margin
            L = max(5, min(L + (L % 2 == 0), MAXTAPS))
            h = signal.firwin(L, (s[0] + s[1]) / 2, window=("kaiser", float(np.clip(b, 0.5, 11))))
            ok += meets(h, s); orders.append(L - 1)
        res[f"margin_{margin}"] = dict(spec_pass_rate=ok / len(Xte), mean_order=float(np.mean(orders)))
    res["kaiser_formula_mean_order"] = float(np.mean(kte))
    res["window_optimum_mean_order"] = float(np.mean(Yte[:, 0]))
    res["parks_mcclellan_mean_order"] = float(np.mean(ote))
    return res


# ------------------------------------------------------------------ 3. Bayesian optimisation
def bo_cost(params, spec):
    """cost = multiplications (symmetric FIR => ~ (L+1)/2) + penalty for violating the spec"""
    L = int(round(params[0])) | 1
    h, _ = kaiser_design(spec, L=L, beta=params[1])
    r, a = measure(h, spec[0], spec[1])
    pen = 50 * max(0, r - spec[2]) / spec[2] + 5 * max(0, spec[3] - a)
    return (L + 1) / 2 + pen, r, a, L


def bayes_opt(spec, L_range=(9, MAXTAPS), n_init=12, n_iter=40, seed=0):
    r_ = np.random.default_rng(seed)
    lo, hi = np.array([L_range[0], 1.0]), np.array([L_range[1], 10.0])
    P = lo + r_.random((n_init, 2)) * (hi - lo)
    y = np.log(np.array([bo_cost(p, spec)[0] for p in P]))   # log-cost tames the penalty cliff
    best_trace = [np.exp(y.min())]
    gp = GaussianProcessRegressor(Matern(nu=2.5, length_scale=[20, 3]) + WhiteKernel(1e-2),
                                  normalize_y=True, n_restarts_optimizer=2, random_state=0)
    for _ in range(n_iter):
        Pn = (P - lo) / (hi - lo)
        gp.fit(Pn, y)
        cand = r_.random((3000, 2))
        best_n = (P[np.argmin(y)] - lo) / (hi - lo)               # local search around incumbent
        cand = np.vstack([cand, np.clip(best_n + r_.normal(0, 0.05, (1000, 2)), 0, 1),
                          np.clip(best_n + r_.normal(0, 0.15, (1000, 2)), 0, 1)])
        mu, sd = gp.predict(cand, return_std=True)
        imp = y.min() - mu - 0.01                            # Expected Improvement (xi = 0.01)
        z = imp / (sd + 1e-9)
        ei = imp * norm.cdf(z) + sd * norm.pdf(z)
        j = np.argmax(ei) if ei.max() > 1e-10 else np.argmax(sd)   # explore if EI vanishes
        p_new = lo + cand[j] * (hi - lo)
        P = np.vstack([P, p_new]); y = np.append(y, np.log(bo_cost(p_new, spec)[0]))
        best_trace.append(np.exp(y.min()))
    c, r, a, L = bo_cost(P[np.argmin(y)], spec)
    return dict(order=L - 1, beta=float(P[np.argmin(y)][1]), ripple=r, atten=a, cost=c,
                feasible=bool(r <= spec[2] and a >= spec[3]), evals=len(y)), best_trace


def bo_study(order_model, scaler):
    """ML + BO hybrid: Model A predicts the Parks-McClellan order, which is a lower bound
    for the window method, so BO only searches [pred+1, pred+45] taps instead of everything."""
    specs = [np.array([0.30, 0.40, 0.2, 60.0]), np.array([0.20, 0.30, 0.5, 50.0]),
             np.array([0.45, 0.55, 0.1, 70.0])]
    rows = []
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, s in enumerate(specs):
        pred = float(order_model.predict(scaler.transform(features(s[None])))[0])
        L_lo = max(9, int(round(pred)) + 1); L_hi = min(MAXTAPS, L_lo + 45)
        res, tr = bayes_opt(s, L_range=(L_lo, L_hi))
        _, Lk = kaiser_design(s)
        no = remez_min_order(s, Lk)
        kw = kaiser_min_design(s, no + 1)
        grid_evals = ((L_hi - L_lo) // 2 + 1) * len(BETAS)
        rows.append(dict(spec=dict(wp=s[0], ws=s[1], Ap=s[2], As=s[3]), BO=res,
                         model_A_predicted_order=pred, kaiser_formula_order=Lk - 1,
                         window_grid_search_optimum=kw[0] - 1, parks_mcclellan_min_order=no,
                         grid_search_evals_in_same_box=grid_evals))
        ax.plot(tr, label=f"spec {k + 1}")
    ax.set_xlabel("BO iteration"); ax.set_ylabel("best cost so far"); ax.legend()
    ax.set_title("Bayesian optimisation convergence"); plt.tight_layout()
    plt.savefig(f"{OUT}/bo_convergence.png", dpi=130); plt.close()
    return rows


# ------------------------------------------------------------------ 4. Pareto study
def pareto_study():
    wp, ws = 0.3, 0.4
    orders = np.arange(20, 121, 4)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for Ap in [0.1, 0.5, 1.0]:
        att = []
        for N in orders:
            h = signal.remez(N + 1, [0, wp / 2, ws / 2, 0.5], [1, 0],
                             weight=[1 / (10 ** (Ap / 20) - 1), 1 / 10 ** (-60 / 20)], fs=1.0)
            att.append(measure(h, wp, ws)[1])
        ax[0].plot(orders, att, label=f"Ap={Ap} dB")
    ax[0].set_xlabel("filter order"); ax[0].set_ylabel("stopband attenuation (dB)")
    ax[0].set_title("Order vs attenuation (Parks-McClellan)"); ax[0].legend()
    # cost: multiplications per output sample (direct vs symmetric-folded)
    ax[1].plot(orders, orders + 1, label="direct form"); ax[1].plot(orders, orders // 2 + 1, label="symmetric (linear phase)")
    ax[1].set_xlabel("filter order"); ax[1].set_ylabel("multiplications / sample")
    ax[1].set_title("Computational cost"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/pareto.png", dpi=130); plt.close()


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    print("building dataset ...")
    X, N_opt, N_k, N_win, B_win, H = build_dataset(700)
    print(f"  {len(X)} designs; Kaiser formula overdesigns by {np.mean(N_k - N_opt):.1f} taps on average")
    print("training Model A (order) ...")
    omodel, oscaler, resA = train_order_model(X, N_opt, N_k)
    print("training Model B1 (direct coefficients, baseline) ...")
    resB1 = train_coeff_model(X, H)
    print("training Model B2 (window parameters) ...")
    resB2 = train_window_param_model(X, N_win, B_win, N_k, N_opt)
    print("Bayesian optimisation ...")
    resBO = bo_study(omodel, oscaler)
    pareto_study()
    out = dict(model_A_order=resA, model_B1_direct_coefficients=resB1,
               model_B2_window_parameters=resB2, bayesian_optimisation=resBO)
    json.dump(out, open(f"{OUT}/summary.json", "w"), indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))
