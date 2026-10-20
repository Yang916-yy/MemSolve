"""Rebuild the current manuscript's vector figures from explicit local evidence.

Usage:
  python paper/figures.py --evidence /path/to/evidence.json --output paper \
      --audit-dir /path/outside/repository

Diagrams only (no experiment data needed):
  python paper/figures.py --schematics-only --output paper \
      --audit-dir /path/outside/repository

Evidence has a runs list with id="small", path (a directory containing
metrics.jsonl), metrics_sha256, metric_prefix="val", and rows containing
recorded epoch, val_acc1, val_acc5 and val_loss. A baselines list supplies
id="deit_original_4", path (the author's JSONL file), url and sha256.
The plot compares ordinary weights. Raw experiment records remain outside
Git. Missing epochs become gaps; there is no smoothing or uncertainty estimate.
Scientific-writing / scientific-visualization guidance: Kassis et al. (2026),
https://arxiv.org/abs/2609.00065. Figures are original vector artwork.
Diagram layout guidance: https://github.com/plbbl/academic-diagram-skill
VGGT Fig. 2 informed token alignment; MambaVision Fig. 3 informed branch routing.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

BLUE, GREEN, ORANGE, DARK = '#258BC4', '#329B73', '#D79924', '#202B35'
TEXT_INK = {BLUE: '#23668E', GREEN: '#246D50', ORANGE: '#855C16'}
STYLE = {
    'font.family': 'STIXGeneral', 'font.size': 9,
    'mathtext.fontset': 'stix', 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.labelcolor': DARK,
    'text.color': DARK, 'xtick.color': DARK, 'ytick.color': DARK,
    'axes.edgecolor': '#78828C', 'axes.linewidth': .65,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
}


# Diagram geometry uses local coordinates; font sizes stay in physical points.
# Pale matrix cells are schematic, not measured data.
DIAGRAM_FONT = {'title': 10, 'label': 9, 'detail': 9}
PALE_BLUE, PALE_GREEN, PALE_ORANGE = '#DDF3FF', '#E1F5DF', '#FFF0BC'
GRAY, PALE_GRAY, RULE = '#62717E', '#F4F6F8', '#D6DDE3'


def canvas(height, width=6.5, coordinate_width=468):
    fig = plt.figure(figsize=(width, height * width / coordinate_width))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, coordinate_width), ylim=(0, height), aspect='equal')
    ax.axis('off')
    return fig, ax


def label(ax, x, y, text, role='label', color=DARK, ha='center', **kwargs):
    return ax.text(x, y, text, fontsize=DIAGRAM_FONT[role],
                   color=TEXT_INK.get(color, color),
                   ha=ha, va='center', **kwargs)


def arrow(ax, xy, end, color=DARK):
    ax.add_patch(FancyArrowPatch(xy, end, arrowstyle='-|>', mutation_scale=8,
                                linewidth=.9, color=color, shrinkA=0, shrinkB=0,
                                zorder=2))


def route(ax, points, color=DARK, head=True):
    # Draw the shaft once; only its final segment carries an arrowhead.
    body = points[:-1] if head else points
    if len(body) > 1:
        ax.plot(*zip(*body), color=color, lw=.9, solid_capstyle='round', zorder=1)
    if head:
        arrow(ax, points[-2], points[-1], color)


def box(ax, x, y, w, h, text='', color=DARK, fill=PALE_GRAY):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                               boxstyle='round,pad=0,rounding_size=3',
                               linewidth=.85, edgecolor=color, facecolor=fill,
                               zorder=3))
    if text:
        label(ax, x+w/2, y+h/2, text, linespacing=1.4, zorder=4)


def matrix(ax, x, y, w, h, color=GREEN, nr=4, nc=3):
    from matplotlib.colors import to_rgb
    fill = 1 - (1 - np.array(to_rgb(color))) * .27
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor='none', zorder=3))
    for i in range(1, nr):
        ax.plot([x, x+w], [y+i*h/nr]*2, color='white', lw=.8, zorder=4)
    for j in range(1, nc):
        ax.plot([x+j*w/nc]*2, [y, y+h], color='white', lw=.8, zorder=4)
    ax.add_patch(Rectangle((x, y), w, h, facecolor='none', edgecolor=color,
                           linewidth=.85, zorder=4))


def pointop(ax, x, y, text, color=DARK):
    ax.add_patch(Circle((x, y), 5, facecolor='white', edgecolor=color,
                        linewidth=.85, zorder=5))
    label(ax, x, y, text, color=color, zorder=6)


def junction(ax, x, y, color=DARK):
    ax.add_patch(Circle((x, y), 1.5, facecolor=color, edgecolor='none', zorder=5))


def save(fig, stem, output, audit):
    for ext in ('pdf', 'svg'):
        fig.savefig(output/f'{stem}.{ext}', bbox_inches=None)
    svg = output/f'{stem}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(audit/f'{stem}.png', dpi=240, bbox_inches=None)
    plt.close(fig)


def graphical_abstract(output, audit):
    fig, ax = canvas(156, width=5.4)
    fig.set_size_inches(6.2, fig.get_figheight())
    ax.set_aspect('auto')
    label(ax, 10, 142, '(a) Sequential updates', 'title', ha='left', weight='bold')
    label(ax, 220, 142, '(b) Global equilibrium', 'title', ha='left', weight='bold')
    ax.plot([207, 207], [10, 130], color=RULE, lw=.7)

    for x, name in [(20, r'$U_0$'), (70, r'$U_1$'), (164, r'$U_N$')]:
        matrix(ax, x, 69, 24, 24, GRAY, nr=3, nc=3)
        label(ax, x+12, 57, name)
    arrow(ax, (44, 81), (70, 81), GRAY)
    arrow(ax, (94, 81), (119, 81), GRAY)
    label(ax, 133, 81, r'$\cdots$', color=GRAY)
    arrow(ax, (146, 81), (164, 81), GRAY)
    for x, name in [(62, r'$(k_1,c_1)$'), (156, r'$(k_N,c_N)$')]:
        box(ax, x, 108, 40, 20, name, BLUE, PALE_BLUE)
        arrow(ax, (x+20, 108), (x+20, 93), BLUE)
    label(ax, 104, 29, r'$U_t=U_{t-1}+\beta_t k_t e_t^{\mathsf{T}}$')
    label(ax, 104, 12, r'$e_t=c_t-U_{t-1}^{\mathsf{T}}k_t$', color=GRAY)

    # Converging associations and a shared readout contrast with the chain.
    for y in (50, 64, 78, 92):
        box(ax, 230, y, 20, 10, color=GRAY, fill=PALE_GRAY)
        route(ax, [(250, y+5), (284, 84)], GRAY, head=False)
        box(ax, 432, y, 20, 10, color=ORANGE, fill=PALE_ORANGE)
        arrow(ax, (408, 84), (432, y+5), ORANGE)
    arrow(ax, (284, 84), (308, 84), BLUE)
    route(ax, [(376, 84), (408, 84)], GREEN, head=False)
    box(ax, 308, 55, 68, 54, color=GREEN, fill=PALE_GREEN)
    label(ax, 342, 101, r'$U^\star$', color=GREEN)
    matrix(ax, 326, 72, 32, 24, GREEN, nr=3, nc=4)
    label(ax, 342, 62, r'$r\times h$', 'detail', color=GREEN)
    box(ax, 326, 114, 32, 16, r'$J$', GREEN, PALE_GREEN)
    arrow(ax, (342, 114), (342, 109), GREEN)
    label(ax, 279, 124, 'Write', 'detail', color=BLUE)
    label(ax, 279, 109, r'$P^{\mathsf{T}}$', color=BLUE)
    label(ax, 406, 124, 'Read', 'detail', color=GREEN)
    label(ax, 406, 109, r'$\mathcal{R}$', color=GREEN)
    label(ax, 240, 40, r'$X$')
    label(ax, 442, 40, r'$Y$')
    label(ax, 342, 15, r'$(I+J)U^\star=P^{\mathsf{T}}C$', color=GREEN)
    save(fig, 'graphical_abstract', output, audit)


def visual_backbone(output, audit):
    fig, ax = canvas(126, width=5.8)
    matrix(ax, 9, 86, 24, 24, BLUE, nr=4, nc=4)
    arrow(ax, (33, 98), (51, 98))
    box(ax, 51, 85, 65, 26, 'Patch embed', BLUE, PALE_BLUE)
    label(ax, 83.5, 118, r'$16\times16$', 'detail', color=BLUE)
    arrow(ax, (116, 98), (144, 98))
    for shift in (6, 3, 0):
        box(ax, 144+shift, 85+shift, 65, 26, color=GREEN, fill=PALE_GREEN)
    label(ax, 176.5, 98, 'Block', color=GREEN, zorder=5)
    label(ax, 229, 116, r'$\times12$', color=GREEN)
    arrow(ax, (215, 98), (242, 98))
    box(ax, 242, 85, 29, 26, 'LN')
    arrow(ax, (271, 98), (290, 98))
    box(ax, 290, 85, 58, 26, 'Mean pool')
    arrow(ax, (348, 98), (367, 98))
    box(ax, 367, 85, 48, 26, 'Linear')
    arrow(ax, (415, 98), (439, 98))
    label(ax, 451, 98, r'$\hat{y}$')

    # The dotted leaders expand one block; they are not data-flow edges.
    ax.plot([144, 18], [83, 76], color=RULE, lw=.8, ls='--')
    ax.plot([209, 450], [83, 76], color=RULE, lw=.8, ls='--')
    ax.add_patch(FancyBboxPatch((14, 8), 440, 68,
                               boxstyle='round,pad=0,rounding_size=5',
                               facecolor='#FAFBFC', edgecolor=RULE, lw=.8))
    label(ax, 24, 34, r'$X$')
    arrow(ax, (32, 34), (48, 34))
    box(ax, 48, 23, 61, 22, 'DWConv', BLUE, PALE_BLUE)
    arrow(ax, (109, 34), (119, 34))
    pointop(ax, 124, 34, '+', BLUE)
    label(ax, 78.5, 16, '3 × 3 CPE', 'detail', color=BLUE)
    arrow(ax, (129, 34), (148, 34))
    box(ax, 148, 23, 25, 22, 'LN')
    arrow(ax, (173, 34), (182, 34))
    box(ax, 182, 23, 62, 22, 'MemSolve', GREEN, PALE_GREEN)
    arrow(ax, (244, 34), (260, 34))
    pointop(ax, 265, 34, '+', GREEN)
    arrow(ax, (270, 34), (287, 34))
    box(ax, 287, 23, 25, 22, 'LN')
    arrow(ax, (312, 34), (321, 34))
    box(ax, 321, 23, 82, 22, 'GELU FFN', ORANGE, PALE_ORANGE)
    arrow(ax, (403, 34), (420, 34))
    pointop(ax, 425, 34, '+', ORANGE)
    arrow(ax, (430, 34), (441, 34))
    label(ax, 449, 34, r'$Y$')
    for start, end, color in [(39, 124, BLUE), (139, 265, GREEN), (278, 425, ORANGE)]:
        junction(ax, start, 34, color)
        route(ax, [(start, 34), (start, 61), (end, 61), (end, 39)], color)
    save(fig, 'visual_backbone', output, audit)


def baseline_records(run):
    raw = Path(run['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != run['sha256']:
        raise ValueError(f"Source log changed: {run['path']}")
    return [{'epoch': row['epoch'] + 1, 'val_acc1': row['test_acc1']}
            for row in map(json.loads, raw.decode().splitlines())]


def curves(evidence, output, audit):
    data = json.loads(evidence.read_text())
    runs = {r['id']: r for r in data['runs']}
    ids = ['small']
    provenance = []
    for key in ids:
        run = runs[key]
        raw = Path(run['path'])/'metrics.jsonl'
        if hashlib.sha256(raw.read_bytes()).hexdigest() != run['metrics_sha256']:
            raise ValueError(f'Source metrics changed: {raw}')
        original = [json.loads(line) for line in raw.read_text().splitlines()]
        prefix = run['metric_prefix']
        expected = [{'epoch': r['epoch'], **{
            f'val_{key}': r[f'{prefix}_{key}'] for key in ('acc1', 'acc5', 'loss')
        }} for r in original if f'{prefix}_acc1' in r and 'epoch' in r]
        if expected != run['rows']:
            raise ValueError(f'Evidence is inconsistent with source: {raw}')
        provenance.append({'id': key, 'path': str(raw), 'sha256': run['metrics_sha256'],
                           'metric_prefix': prefix})
    baselines = data['baselines']
    baseline_rows = {r['id']: baseline_records(r) for r in baselines}
    provenance.extend(baselines)
    fig, ax = plt.subplots(figsize=(2.9375, 2.1), layout='constrained')

    def plot(ax, records, name, color, style, width=1.4):
        last = max(r['epoch'] for r in records)
        by_epoch = {r['epoch']: r for r in records}
        if len(by_epoch) != len(records):
            raise ValueError(f'Duplicate epoch in {name}')
        epochs = np.arange(1, last + 1)
        values = [by_epoch.get(int(e), {}).get('val_acc1', np.nan) for e in epochs]
        ax.plot(epochs, values, color=color, ls=style, lw=width, label=name)

    plot(ax, runs['small']['rows'], 'MemSolve-S', GREEN, '-')
    plot(ax, baseline_rows['deit_original_4'], 'DeiT-S', BLUE, '--', 1.2)
    ax.grid(axis='y', color='#E4E8EC', lw=.65)
    ax.set(xlim=(0, 300), ylim=(0, 90), xlabel='Epoch',
           ylabel='Validation Top-1 (%)', xticks=[0, 100, 200, 300])
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    save(fig, 'training_curves', output, audit)
    (audit/'figure_provenance.json').write_text(json.dumps({
        'sources': provenance,
        'transformations': 'DeiT epoch 0..299 shifted to completed epochs 1..300; every epoch; no smoothing; missing epochs are gaps',
        'uncertainty': 'One run per model; no uncertainty interval estimated',
        'overview': 'Original schematic; main.tex equations frame, solve and readout',
        'review_status': 'Agent-generated draft; author scientific review pending',
        'matplotlib': matplotlib.__version__,
        'files': [f'{stem}.{ext}' for stem in (
            'graphical_abstract',
            'visual_backbone', 'training_curves') for ext in ('pdf', 'svg')],
        'alt_text': {
            'graphical_abstract': 'A generic ordered memory chain motivates a full-context alternative: all tokens contribute to one compact equilibrium state, and all outputs use that state. This is a conceptual comparison, not numerical equivalence to the chain.',
        'visual_backbone': 'Patch embedding, twelve constant-width blocks, token LayerNorm, mean pooling and a linear classifier. An expanded block shows three residual branches: CPE, pre-LayerNorm MemSolve and pre-LayerNorm GELU FFN.',
        'training_curves': 'Validation Top-1 over 300 epochs: MemSolve-S ordinary weights and the author-released four-GPU DeiT-S run. Protocols differ; this is not a wall-clock comparison.'}
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--schematics-only', action='store_true',
                        help='Rebuild the two diagrams without reading experiment logs.')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    parser.add_argument('--audit-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.schematics_only and args.evidence is None:
        parser.error('--evidence is required unless --schematics-only is used')
    args.output.mkdir(parents=True, exist_ok=True); args.audit_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        visual_backbone(args.output, args.audit_dir)
        graphical_abstract(args.output, args.audit_dir)
        if not args.schematics_only:
            curves(args.evidence, args.output, args.audit_dir)


if __name__ == '__main__':
    main()
