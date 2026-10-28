#!/usr/bin/env python3
"""
Digital wireless link over an AWGN channel: ASK / PSK / QPSK / FSK
Integrate-and-dump receiver with threshold detection.

Generates every figure and table used in the project report.
Run:  python wireless_link_simulation.py
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.special import erfc

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

 
# Project parameters 
 
SCHEMES = ["ASK", "PSK", "QPSK", "FSK"]          # (a) transmission scheme
VAMPS = [4, 6, 8, 10]                             # (b) carrier amplitude, volts
FSS = [12, 14, 18, 22]                            # (c) sampling frequency, Hz
BRS = [5, 6, 7, 8]                                # (d) bit rate, bits/s
BASE = dict(A=6.0, fs=18, BR=6)                   # baseline operating point
N0 = 1.0                                          # noise PSD for fixed-noise sweeps
PKT = 32                                          # bits per packet
NBITS = 640_000                                   # Monte Carlo bits (fixed noise)
NBITS_SNR = 320_000                               # Monte Carlo bits (SNR sweeps)
SNRS = np.arange(-6, 12, 2)                       # dB

COL = {"ASK": "#1f77b4", "PSK": "#d62728", "QPSK": "#2ca02c", "FSK": "#ff7f0e"}
MRK = {"ASK": "o", "PSK": "s", "QPSK": "^", "FSK": "D"}
PCOL = ["#440154", "#31688e", "#35b779", "#e0a800"]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold",
})

rng = np.random.default_rng(2026)


def Q(x):
    return 0.5 * erfc(np.asarray(x) / np.sqrt(2))


 
# Core link model (complex-baseband equivalent, one real sample per tick of fs)
 
def avg_power(scheme, A):
    """Average signal power in the baseband-equivalent model (a = A/sqrt2)."""
    return A ** 2 / 4 if scheme == "ASK" else A ** 2 / 2


def windows(nsym, fs, R):
    """Sample index where each symbol starts (handles non-integer fs/R)."""
    starts = np.ceil(np.arange(nsym + 1) * fs / R - 1e-9).astype(int)
    return starts, np.diff(starts)


def run(scheme, A, fs, BR, sigma2, nbits, g):
    """Transmit random bits, add AWGN (variance sigma2 per sample), detect."""
    a = A / np.sqrt(2)
    sd = np.sqrt(sigma2)
    bits = g.integers(0, 2, nbits)
    R, nsym = (BR / 2, nbits // 2) if scheme == "QPSK" else (BR, nbits)
    starts, cnt = windows(nsym, fs, R)
    N = starts[-1]
    sym = np.repeat(np.arange(nsym), cnt)
    noise = lambda: sd * g.standard_normal(N)
    integ = lambda x: np.add.reduceat(x, starts[:-1]) / cnt     # integrate and dump
    info = dict(starts=starts, R=R)
    if scheme == "ASK":                      # on-off keying
        s = a * bits[sym]
        x = s + noise(); st = integ(x)
        rx = (st > a / 2).astype(int)
        info.update(s=s, x=x, I=st, Q=integ(noise()), thr=a / 2)
    elif scheme == "PSK":                    # BPSK
        s = a * (2 * bits[sym] - 1)
        x = s + noise(); st = integ(x)
        rx = (st > 0).astype(int)
        info.update(s=s, x=x, I=st, Q=integ(noise()), thr=0.0)
    elif scheme == "QPSK":                   # Gray mapped, 2 bits per symbol
        bI, bQ = bits[0::2], bits[1::2]
        ai = a / np.sqrt(2) * (2 * bI - 1); aq = a / np.sqrt(2) * (2 * bQ - 1)
        sI = ai[sym]; xI = sI + noise(); xQ = aq[sym] + noise()
        stI, stQ = integ(xI), integ(xQ)
        rx = np.empty(nbits, int); rx[0::2] = stI > 0; rx[1::2] = stQ > 0
        info.update(s=sI, x=xI, I=stI, Q=stQ, thr=0.0)
    else:                                    # coherent orthogonal BFSK
        x0 = a * (1 - bits)[sym] + noise(); x1 = a * bits[sym] + noise()
        s0, s1 = integ(x0), integ(x1)
        rx = (s1 > s0).astype(int)
        info.update(s=a * (2 * bits[sym] - 1), x=x1 - x0, I=s0, Q=s1, thr=0.0)
    return bits, rx, info


def measure(scheme, A, fs, BR, sigma2, nbits, g):
    bits, rx, _ = run(scheme, A, fs, BR, sigma2, nbits, g)
    err = bits != rx
    return int(err.sum()), int(err.reshape(-1, PKT).any(axis=1).sum())


def ebn0(scheme, A, fs, BR, sigma2):
    return avg_power(scheme, A) * fs / (2 * BR * sigma2)


def ber_theory(scheme, e):
    return Q(np.sqrt(e)) if scheme in ("ASK", "FSK") else Q(np.sqrt(2 * e))


def ber_theory_quant(scheme, A, fs, BR, sigma2):
    """Theory that accounts for a whole number of samples in each window."""
    a2 = A ** 2 / 2
    R = BR / 2 if scheme == "QPSK" else BR
    _, n = windows(4000, fs, R)
    k = {"ASK": 4.0, "PSK": 1.0, "FSK": 2.0, "QPSK": 2.0}[scheme]
    return float(np.mean(Q(np.sqrt(a2 * n / (k * sigma2)))))


def snr_db(scheme, A, sigma2):
    return 10 * np.log10(avg_power(scheme, A) / sigma2)


rows = []


def record(tag, scheme, A, fs, BR, sigma2, nbits, g):
    be, pe = measure(scheme, A, fs, BR, sigma2, nbits, g)
    e = ebn0(scheme, A, fs, BR, sigma2)
    bt = float(ber_theory(scheme, e))
    ber = be / nbits
    npk = nbits // PKT
    r = dict(experiment=tag, scheme=scheme, Vamp_V=A, fs_Hz=fs, BR_bps=BR,
             SNR_dB=round(snr_db(scheme, A, sigma2), 3),
             EbN0_dB=round(10 * np.log10(e), 3), bits=nbits, bit_errors=be,
             packets=npk, packet_errors=pe, BER_sim=ber, BER_theory=bt,
             BER_theory_quantised=ber_theory_quant(scheme, A, fs, BR, sigma2),
             PER_sim=pe / npk, PER_theory=float(1 - (1 - bt) ** PKT))
    rows.append(r)
    return r


 
# Experiments
 
def sig2_fixed(fs):
    return N0 * fs / 2


res = {}
# (a) scheme comparison at baseline, fixed noise floor
res["a"] = {s: record("a_scheme", s, BASE["A"], BASE["fs"], BASE["BR"],
                      sig2_fixed(BASE["fs"]), NBITS, rng) for s in SCHEMES}
# (b) Vamp, (c) fs, (d) BR at fixed noise floor
res["b"] = {s: [record("b_vamp", s, A, BASE["fs"], BASE["BR"], sig2_fixed(BASE["fs"]),
                       NBITS, rng) for A in VAMPS] for s in SCHEMES}
res["c"] = {s: [record("c_fs", s, BASE["A"], f, BASE["BR"], sig2_fixed(f),
                       NBITS, rng) for f in FSS] for s in SCHEMES}
res["d"] = {s: [record("d_br", s, BASE["A"], BASE["fs"], b, sig2_fixed(BASE["fs"]),
                       NBITS, rng) for b in BRS] for s in SCHEMES}


# SNR sweeps (SNR = signal power / noise power per sample)
def snr_sweep(tag, s, A, fs, BR):
    out = []
    for d in SNRS:
        sg2 = avg_power(s, A) / 10 ** (d / 10)
        out.append(record(tag, s, A, fs, BR, sg2, NBITS_SNR, rng))
    return out


res["snr_a"] = {s: snr_sweep("snr_a_scheme", s, BASE["A"], BASE["fs"], BASE["BR"]) for s in SCHEMES}
res["snr_b"] = {s: [snr_sweep("snr_b_vamp", s, A, BASE["fs"], BASE["BR"]) for A in VAMPS] for s in SCHEMES}
res["snr_c"] = {s: [snr_sweep("snr_c_fs", s, BASE["A"], f, BASE["BR"]) for f in FSS] for s in SCHEMES}
res["snr_d"] = {s: [snr_sweep("snr_d_br", s, BASE["A"], BASE["fs"], b) for b in BRS] for s in SCHEMES}

with open(os.path.join(HERE, "results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)


 
# Figures
 
def box(ax, x, y, w, h, text, fc="#e8f1fb", ec="#2b5c8a", fs=10, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#12263a")


def arrow(ax, p, q, text=None, dy=0.18, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=1.6,
                                 color="#333333", **kw))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text, ha="center",
                fontsize=9.5, style="italic", color="#7a1f1f")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(14, 7.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.6); ax.axis("off")
    ax.text(7, 7.3, "System architecture: digital wireless link over an AWGN channel",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(0.3, 6.85, "TRANSMITTER", fontsize=10, color="#555", fontweight="bold")
    ax.text(7.3, 4.55, "CHANNEL", fontsize=10, color="#555", fontweight="bold")
    box(ax, 0.3, 4.9, 2.4, 1.3, "Message source\nm(t) + ADC\n(random bits)", fc="#fdf2d8", ec="#a87a00")
    box(ax, 3.9, 4.9, 2.8, 1.3, "Modulator (Tx)\nASK / PSK /\nQPSK / FSK", fc="#e1f3e1", ec="#2f7a2f")
    box(ax, 3.9, 6.45, 2.8, 0.55, "Carrier, Vamp = 4 to 10 V", fc="#f3e6f7", ec="#7a2f8a", fs=8.5)
    arrow(ax, (5.3, 6.45), (5.3, 6.2))
    arrow(ax, (2.7, 5.55), (3.9, 5.55), "bits", dy=0.15)
    ax.add_patch(plt.Circle((9.6, 5.55), 0.6, fc="#fde7e7", ec="#a83232", lw=1.6))
    ax.text(9.6, 5.55, "\u03a3", ha="center", va="center", fontsize=22, fontweight="bold")
    ax.text(9.6, 6.95, "AWGN  n(t)", ha="center", fontsize=10, color="#7a1f1f", fontweight="bold")
    arrow(ax, (9.6, 6.85), (9.6, 6.15))
    arrow(ax, (6.7, 5.55), (9.0, 5.55), "s(t)", dy=0.15)
    arrow(ax, (10.2, 5.55), (12.2, 5.55))
    ax.text(11.2, 5.72, "r(t) = s(t) + n(t)", ha="center", fontsize=9.5, style="italic", color="#7a1f1f")
    ax.text(0.3, 3.95, "RECEIVER", fontsize=10, color="#555", fontweight="bold")
    box(ax, 11.2, 2.5, 2.5, 1.3, "Processing circuits\n(filter, coherent\ndemodulator)", fc="#e8f1fb")
    arrow(ax, (12.45, 5.55), (12.45, 3.8))
    ax.text(12.55, 4.55, "fs = 12 to 22 Hz\n(sampled)", fontsize=8.5, color="#444")
    box(ax, 7.2, 2.5, 2.7, 1.3, "Integrate and dump\n(RC / sum over Tb)", fc="#e1f3e1", ec="#2f7a2f")
    box(ax, 3.7, 2.5, 2.2, 1.3, "Threshold\ndetector", fc="#fdf2d8", ec="#a87a00")
    box(ax, 0.3, 2.5, 2.4, 1.3, "Binary sequence\ndetector \u2192 m\u0302", fc="#f3e6f7", ec="#7a2f8a")
    arrow(ax, (11.2, 3.15), (9.9, 3.15), "r\u2080(t)")
    arrow(ax, (7.2, 3.15), (5.9, 3.15), "z")
    arrow(ax, (3.7, 3.15), (2.7, 3.15), "0 / 1")
    box(ax, 7.45, 0.9, 2.2, 0.8, "Bit clock, BR = 5 to 8 bps", fc="#f5f5f5", ec="#888", fs=8.5)
    arrow(ax, (8.55, 1.7), (8.55, 2.5))
    box(ax, 0.3, 0.3, 5.9, 1.6,
        "Performance analysis\nBit errors, packet errors, BER (simulated),\nBER (theory, Q-function), PER, SNR",
        fc="#fff3cd", ec="#a87a00", fs=9.5)
    arrow(ax, (1.5, 2.5), (1.5, 1.9))
    box(ax, 10.3, 0.3, 3.4, 1.6,
        "Report sweeps\n(i) BER vs a / b / c / d\n(ii) BER vs SNR for a / b / c / d\n(iii) analysis",
        fc="#eaf6f6", ec="#2a7f7f", fs=9)
    fig.savefig(os.path.join(FIG, "fig1_architecture.png"), bbox_inches="tight"); plt.close(fig)


def fig_modulated():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0]); BR = 6; fc = 18
    t = np.linspace(0, len(bits) / BR, 4000, endpoint=False)
    b = bits[(t * BR).astype(int)]
    A = 6
    ask = A * b * np.cos(2 * np.pi * fc * t)
    psk = A * np.cos(2 * np.pi * fc * t + np.pi * (1 - b))
    f0, f1 = 12, 24
    fsk = A * np.cos(2 * np.pi * np.where(b == 1, f1, f0) * t)
    sym = np.minimum((t * BR / 2).astype(int), 3)
    bI, bQ = bits[0::2][sym], bits[1::2][sym]
    qpsk = A * (np.cos(2 * np.pi * fc * t) * (2 * bI - 1) / np.sqrt(2)
                - np.sin(2 * np.pi * fc * t) * (2 * bQ - 1) / np.sqrt(2))
    fig, axs = plt.subplots(5, 1, figsize=(11, 9), sharex=True)
    axs[0].step(t, b, where="post", color="k")
    axs[0].set_ylabel("m(t)"); axs[0].set_title("Message bits 1 0 1 1 0 0 1 0 and the four modulated waveforms (Vamp = 6 V)")
    for ax, y, n in zip(axs[1:], [ask, psk, qpsk, fsk], SCHEMES):
        ax.plot(t, y, color=COL[n], lw=1.1); ax.set_ylabel(n); ax.set_ylim(-7.5, 7.5)
        for k in range(len(bits) + 1):
            ax.axvline(k / BR, color="gray", lw=0.5, ls=":")
    axs[-1].set_xlabel("time (s)")
    for k, v in enumerate(bits):
        axs[0].text((k + 0.5) / BR, 0.5, str(v), ha="center", va="center", color="#555")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_modulated_waveforms.png")); plt.close(fig)


def fig_pipeline():
    scheme, nb = "PSK", 30
    sg2 = avg_power(scheme, BASE["A"]) / 10 ** (-1 / 10)
    for seed in range(500):
        g = np.random.default_rng(seed)
        bits, rx, info = run(scheme, BASE["A"], BASE["fs"], BASE["BR"], sg2, nb, g)
        if 1 <= np.sum(bits != rx) <= 2:
            break
    fs, BR = BASE["fs"], BASE["BR"]
    k = np.arange(len(info["x"])); t = k / fs
    Tb = 1 / BR
    fig, axs = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    axs[0].step(np.arange(nb) * Tb, bits, where="post", color="k")
    axs[0].set_title("Receiver pipeline for PSK (Vamp = 6 V, fs = 18 Hz, BR = 6 bps, SNR = -1 dB)")
    axs[0].set_ylabel("m(t)  sent bits")
    axs[1].step(t, info["s"], where="post", color=COL["PSK"]); axs[1].set_ylabel("s(t)  clean")
    axs[2].plot(t, info["x"], ".-", color="#ff7f0e", lw=0.8, ms=4); axs[2].set_ylabel("r(t) = s + n")
    ends = (info["starts"][1:]) / fs
    err = bits != rx
    axs[3].vlines(ends - Tb / 2, 0, info["I"], color="#555", lw=1.2)
    axs[3].scatter(ends - Tb / 2, info["I"], c=np.where(err, "red", "#2ca02c"), s=36, zorder=3)
    axs[3].axhline(info["thr"], color="blue", ls="--", lw=1, label="threshold = 0")
    axs[3].set_ylabel("integrator output z"); axs[3].legend(loc="upper right")
    axs[4].step(np.arange(nb) * Tb, bits, where="post", color="k", lw=2.5, alpha=0.35, label="sent")
    axs[4].step(np.arange(nb) * Tb, rx, where="post", color="#1f77b4", ls="--", label="received")
    for i in np.where(err)[0]:
        axs[4].axvspan(i * Tb, (i + 1) * Tb, color="red", alpha=0.25)
    axs[4].set_ylabel("m\u0302  detected"); axs[4].legend(loc="upper right")
    axs[4].set_xlabel("time (s)  (red = bit error, green = correct decision)")
    for ax in axs:
        for i in range(nb + 1):
            ax.axvline(i * Tb, color="gray", lw=0.4, ls=":")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_receiver_pipeline.png")); plt.close(fig)


def fig_constellations():
    g = np.random.default_rng(7)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9))
    a = BASE["A"] / np.sqrt(2)
    for ax, s in zip(axs, SCHEMES):
        sg2 = avg_power(s, BASE["A"]) / 10 ** (2 / 10)
        bits, rx, info = run(s, BASE["A"], BASE["fs"], BASE["BR"], sg2, 3000, g)
        I, Qv = info["I"], info["Q"]
        if s == "ASK":
            lab = bits
        elif s == "PSK":
            lab = bits
        elif s == "QPSK":
            lab = bits[0::2] * 2 + bits[1::2]
        else:
            lab = bits
        ax.scatter(I, Qv, c=[PCOL[int(l) % 4] if s != "QPSK" else PCOL[int(l)] for l in lab],
                   s=7, alpha=0.5)
        ax.set_title(f"{s} (SNR = 2 dB)")
        ax.set_aspect("equal")
        if s == "ASK":
            ax.axvline(a / 2, color="blue", ls="--"); ax.set_xlabel("I statistic"); ax.set_ylabel("Q statistic (noise only)")
        elif s == "PSK":
            ax.axvline(0, color="blue", ls="--"); ax.set_xlabel("I statistic"); ax.set_ylabel("Q statistic (noise only)")
        elif s == "QPSK":
            ax.axvline(0, color="blue", ls="--"); ax.axhline(0, color="blue", ls="--")
            ax.set_xlabel("I statistic"); ax.set_ylabel("Q statistic")
        else:
            lim = [min(I.min(), Qv.min()), max(I.max(), Qv.max())]
            ax.plot(lim, lim, color="blue", ls="--"); ax.set_xlabel("tone-0 correlator"); ax.set_ylabel("tone-1 correlator")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_constellations.png")); plt.close(fig)


def fig_ber_snr_baseline():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    dense = np.linspace(-6, 10, 200)
    for s in SCHEMES:
        r = res["snr_a"][s]
        x = np.array([q["SNR_dB"] for q in r]); y = np.array([q["BER_sim"] for q in r])
        m = y > 0
        axs[0].semilogy(x[m], y[m], MRK[s], color=COL[s], label=f"{s} simulated")
        e = 10 ** (dense / 10) * BASE["fs"] / (2 * BASE["BR"])
        axs[0].semilogy(dense, ber_theory(s, e), "-", color=COL[s], lw=1.2, label=f"{s} theory")
        # Eb/N0 axis version
        xe = np.array([q["EbN0_dB"] for q in r])
        axs[1].semilogy(xe[m], y[m], MRK[s], color=COL[s], label=f"{s} simulated")
        ed = np.linspace(-4, 14, 200)
        axs[1].semilogy(ed, ber_theory(s, 10 ** (ed / 10)), "-", color=COL[s], lw=1.2)
    for ax, xl, tt in zip(axs, ["SNR (dB), signal power / noise power per sample", "Eb/N0 (dB)"],
                          ["BER vs SNR (project definition)", "Same data on the standard Eb/N0 axis"]):
        ax.set_xlabel(xl); ax.set_ylabel("Bit error rate"); ax.set_title(tt)
        ax.set_ylim(1e-5, 1)
    axs[0].legend(ncol=2, fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_ber_vs_snr_schemes.png")); plt.close(fig)


def fig_ber_vs_scheme():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(4); w = 0.38
    sim = [res["a"][s]["BER_sim"] for s in SCHEMES]; th = [res["a"][s]["BER_theory"] for s in SCHEMES]
    axs[0].bar(x - w / 2, sim, w, color=[COL[s] for s in SCHEMES], label="simulated")
    axs[0].bar(x + w / 2, th, w, color=[COL[s] for s in SCHEMES], alpha=0.4, hatch="//", label="theory")
    for i, v in enumerate(sim):
        axs[0].text(i - w / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    axs[0].set_xticks(x); axs[0].set_xticklabels(SCHEMES); axs[0].set_ylabel("BER")
    axs[0].set_title("(i-a) BER by modulation scheme\n(Vamp 6 V, fs 18 Hz, BR 6 bps, N0 = 1)"); axs[0].legend()
    per = [res["a"][s]["PER_sim"] for s in SCHEMES]; pt = [res["a"][s]["PER_theory"] for s in SCHEMES]
    axs[1].bar(x - w / 2, per, w, color=[COL[s] for s in SCHEMES])
    axs[1].bar(x + w / 2, pt, w, color=[COL[s] for s in SCHEMES], alpha=0.4, hatch="//")
    axs[1].set_xticks(x); axs[1].set_xticklabels(SCHEMES); axs[1].set_ylabel(f"PER ({PKT}-bit packets)")
    axs[1].set_title("Packet error rate by scheme")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_scheme_comparison.png")); plt.close(fig)


def fig_ber_vs_param():
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    # b: Vamp
    Ad = np.linspace(3.5, 10.5, 100)
    for s in SCHEMES:
        y = np.array([q["BER_sim"] for q in res["b"][s]]); m = y > 0
        axs[0].semilogy(np.array(VAMPS)[m], y[m], MRK[s] + "-", color=COL[s], label=f"{s} sim")
        e = np.array([ebn0(s, A, BASE["fs"], BASE["BR"], sig2_fixed(BASE["fs"])) for A in Ad])
        axs[0].semilogy(Ad, ber_theory(s, e), ":", color=COL[s])
    axs[0].set_xlabel("Carrier amplitude Vamp (V)"); axs[0].set_title("(i-b) BER vs Vamp (dotted = theory)")
    axs[0].set_xticks(VAMPS)
    # c: fs
    for s in SCHEMES:
        y = np.array([q["BER_sim"] for q in res["c"][s]]); m = y > 0
        axs[1].semilogy(np.array(FSS)[m], y[m], MRK[s] + "-", color=COL[s], label=s)
        yq = [q["BER_theory_quantised"] for q in res["c"][s]]
        axs[1].semilogy(FSS, yq, ":", color=COL[s])
        axs[1].semilogy(FSS, [q["BER_theory"] for q in res["c"][s]], "--", color=COL[s], lw=0.8, alpha=0.6)
    axs[1].set_xlabel("Sampling frequency fs (Hz)")
    axs[1].set_title("(i-c) BER vs fs\n(dashed = ideal theory, dotted = integer-sample theory)")
    axs[1].set_xticks(FSS)
    # d: BR
    for s in SCHEMES:
        y = np.array([q["BER_sim"] for q in res["d"][s]]); m = y > 0
        axs[2].semilogy(np.array(BRS)[m], y[m], MRK[s] + "-", color=COL[s], label=s)
        axs[2].semilogy(BRS, [q["BER_theory_quantised"] for q in res["d"][s]], ":", color=COL[s])
    axs[2].set_xlabel("Bit rate BR (bps)"); axs[2].set_title("(i-d) BER vs bit rate"); axs[2].set_xticks(BRS)
    for ax in axs:
        ax.set_ylabel("Bit error rate"); ax.set_ylim(1e-5, 1)
    axs[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_ber_vs_parameters.png")); plt.close(fig)


def fig_ber_snr_params():
    fig, axs = plt.subplots(3, 4, figsize=(17, 11), sharex=True, sharey=True)
    rowdefs = [("b", "Vamp (V)", VAMPS, "snr_b"), ("c", "fs (Hz)", FSS, "snr_c"), ("d", "BR (bps)", BRS, "snr_d")]
    dense = np.linspace(-6, 10, 200)
    for i, (key, lab, vals, rk) in enumerate(rowdefs):
        for j, s in enumerate(SCHEMES):
            ax = axs[i, j]
            for k, v in enumerate(vals):
                r = res[rk][s][k]
                x = np.array([q["SNR_dB"] for q in r]); y = np.array([q["BER_sim"] for q in r]); m = y > 0
                ax.semilogy(x[m], y[m], MRK[s] + "-", color=PCOL[k], ms=4, lw=1.1, label=f"{lab.split()[0]} = {v}")
                fsv = v if key == "c" else BASE["fs"]; brv = v if key == "d" else BASE["BR"]
                e = 10 ** (dense / 10) * fsv / (2 * brv)
                ax.semilogy(dense, ber_theory(s, e), ":", color=PCOL[k], lw=1)
            ax.set_ylim(1e-5, 1)
            if i == 0: ax.set_title(s)
            if j == 0: ax.set_ylabel(f"BER   [sweep of {lab}]")
            if i == 2: ax.set_xlabel("SNR (dB)")
            ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("(ii) BER vs SNR for each parameter value (markers = simulation, dotted = theory)",
                 fontweight="bold", y=0.995)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig8_ber_vs_snr_per_parameter.png")); plt.close(fig)


def fig_per():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    dense = np.linspace(-6, 10, 200)
    for s in SCHEMES:
        r = res["snr_a"][s]
        x = np.array([q["SNR_dB"] for q in r]); y = np.array([q["PER_sim"] for q in r])
        axs[0].plot(x, y, MRK[s], color=COL[s], label=f"{s} sim")
        e = 10 ** (dense / 10) * BASE["fs"] / (2 * BASE["BR"])
        axs[0].plot(dense, 1 - (1 - ber_theory(s, e)) ** PKT, "-", color=COL[s], lw=1)
    axs[0].set_xlabel("SNR (dB)"); axs[0].set_ylabel("PER"); axs[0].legend(fontsize=8)
    axs[0].set_title(f"Packet error rate vs SNR ({PKT}-bit packets)")
    for s in SCHEMES:
        y = np.array([q["PER_sim"] for q in res["b"][s]])
        axs[1].plot(VAMPS, y, MRK[s] + "-", color=COL[s], label=s)
    axs[1].set_xlabel("Vamp (V)"); axs[1].set_ylabel("PER"); axs[1].set_xticks(VAMPS)
    axs[1].set_title("PER vs Vamp (fixed noise floor)"); axs[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig9_packet_error_rate.png")); plt.close(fig)


fig_architecture(); fig_modulated(); fig_pipeline(); fig_constellations()
fig_ber_snr_baseline(); fig_ber_vs_scheme(); fig_ber_vs_param(); fig_ber_snr_params(); fig_per()

 
# Console summary
 
print("BASELINE (Vamp 6 V, fs 18 Hz, BR 6 bps, N0 = 1, %d bits, %d-bit packets)" % (NBITS, PKT))
print(f"{'scheme':6} {'SNR dB':>7} {'Eb/N0':>6} {'bit err':>8} {'pkt err':>8} {'BER sim':>10} {'BER th':>10} {'PER sim':>8} {'PER th':>8}")
for s in SCHEMES:
    r = res["a"][s]
    print(f"{s:6} {r['SNR_dB']:7.2f} {r['EbN0_dB']:6.2f} {r['bit_errors']:8d} {r['packet_errors']:8d} "
          f"{r['BER_sim']:10.2e} {r['BER_theory']:10.2e} {r['PER_sim']:8.3f} {r['PER_theory']:8.3f}")
for key, name in [("b", "Vamp"), ("c", "fs"), ("d", "BR")]:
    print(f"\nSweep {name}")
    for s in SCHEMES:
        print(s, [f"{q['BER_sim']:.2e}" for q in res[key][s]], "| th", [f"{q['BER_theory_quantised']:.2e}" for q in res[key][s]])
