import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 7, "mathtext.fontset": "stix"})
K = "#000000"


def box(ax, x, y, w, h, t, fs=6.0, fc="white", ls="-", lw=0.7, bold=False):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=K, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, linespacing=1.12,
            fontweight="bold" if bold else "normal")


def arr(ax, x1, y1, x2, y2, ls="-", t=None, dx=0, dy=0, fs=5.8, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=7, lw=0.7,
                                 color=K, ls=ls, shrinkA=0, shrinkB=0))
    if t:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, t, fontsize=fs, ha="center", style="italic")


def save(f, name):
    f.savefig(f"fig/{name}.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(f)


# Fig. 2 - assessment and aggregation workflow (single column)
f, ax = plt.subplots(figsize=(3.45, 2.45)); ax.set_xlim(0, 10); ax.set_ylim(-0.15, 8.0); ax.axis("off")
box(ax, 0.0, 7.0, 4.3, 0.9, "Request $x$: QR / VPA; optional\namount, purpose, message")
box(ax, 5.2, 7.0, 4.8, 0.9, "Read-only state before $t$: ledger,\naggregates, graph $\\mathcal{G}_{<t}$, profile")
arr(ax, 4.3, 7.45, 5.2, 7.45)
box(ax, 0.0, 5.6, 10.0, 0.85, "Ten evidence families, each returning $e_i=(a_i,\\,z_i,\\,\\Gamma_i,\\,\\Phi_i)$", fs=6.2)
arr(ax, 2.15, 7.0, 2.15, 6.45); arr(ax, 7.6, 7.0, 7.6, 6.45)
box(ax, 0.0, 3.85, 3.15, 1.2, "Active scores\n→ damped max $R_0$ (5)")
box(ax, 3.4, 3.85, 3.15, 1.2, "Facts of families\nwith $z_i\\geq20$\n→ $k$, $B(k)$ (6)")
box(ax, 6.8, 3.85, 3.2, 1.2, "Availability only\n→ coverage $C$ (7),\nlevel $L$")
for x in (1.57, 4.97, 8.4):
    arr(ax, x, 5.6, x, 5.05)
box(ax, 0.0, 2.2, 6.55, 1.1, "$R=\\min(99,R_0+B(k))$ → verdict $V$ (8):\nhard block, else bands 22 / 45 / 70")
arr(ax, 1.57, 3.85, 1.57, 3.3); arr(ax, 4.97, 3.85, 4.97, 3.3)
box(ax, 6.8, 2.2, 3.2, 1.1, "Reported apart from $R$\n(unavailable ≠ safe,\nunavailable ≠ fraud)", fs=6.0)
arr(ax, 8.4, 3.85, 8.4, 3.3)
box(ax, 0.0, 1.1, 10.0, 0.8, "Advisory response: $R$, $C$, $L$, $V$, findings, corroborating\nfacts and the families that could not be evaluated", fs=6.1)
arr(ax, 3.27, 2.2, 3.27, 1.9); arr(ax, 8.4, 2.2, 8.4, 1.9)
box(ax, 0.9, -0.1, 9.1, 0.85, "Observation path — only if the payer pays and confirms: ledger,\naggregates and graph updated in one transaction", fs=5.9, ls="--")
arr(ax, 5.45, 1.1, 5.45, 0.75, ls="--")
save(f, "fig2_workflow")

# Fig. 3 - chronological replay loop (single column)
f, ax = plt.subplots(figsize=(3.45, 2.15)); ax.set_xlim(0, 10); ax.set_ylim(0, 7.6); ax.axis("off")
steps = ["1. Set application clock to $t_k$",
         "2. Read pre-payment state: payer's rows $<t_k$;\nledger, aggregates and graph $<t_k$",
         "3. Assess $x_k$ with the production function\n→ $R$, $C$, $L$, $V$, evidence, findings",
         "4. Record evidence and V1 / V2 vectors built\nfrom the same state",
         "5. Shadow cross-check (tests 1–7, 10)",
         "6. Only now: record payment $x_k$\n(ledger, aggregates, graph)"]
hs = [0.62, 0.95, 0.95, 0.95, 0.62, 0.95]
y = 7.5
ys = []
for s, h in zip(steps, hs):
    y -= h
    box(ax, 0.0, y, 6.6, h, s, fs=5.6)
    ys.append((y, h))
    y -= 0.28
for (y1, h1), (y2, h2) in zip(ys[:-1], ys[1:]):
    arr(ax, 3.3, y1, 3.3, y2 + h2)
yl, hl = ys[-1]; y0, h0 = ys[0]
ax.plot([6.6, 7.05, 7.05], [yl + hl / 2, yl + hl / 2, y0 + h0 / 2], color=K, lw=0.7)
arr(ax, 7.05, y0 + h0 / 2, 6.6, y0 + h0 / 2)
ax.text(7.2, 3.9, "$k \\leftarrow k+1$", fontsize=6.3, rotation=90, va="center")
box(ax, 7.75, 0.3, 2.25, 3.6, "After the pass:\n\n8. independent\nrebuild from\nearlier rows\nonly\n\n9. earlier\nrecords\nunchanged", fs=5.3, ls="--")
save(f, "fig3_replay")
print("ok")
