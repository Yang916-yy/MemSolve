"""Rebuild the current manuscript's vector figures from explicit local evidence.

Usage:
  python paper/figures.py --evidence /path/to/evidence.json --output paper \
      --audit-dir /path/outside/repository

Evidence has a runs list, with id, path, metrics_sha256, and rows containing
recorded epoch, val_acc1 and val_loss. Raw experiment records remain outside
Git. Missing epochs become gaps; there is no smoothing or uncertainty estimate.
Scientific-writing / scientific-visualization guidance: Kassis et al. (2026),
https://arxiv.org/abs/2609.00065. Figures are original vector artwork.
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

BLUE, GREEN, ORANGE, DARK = '#0072B2', '#00735A', '#B45100', '#202B35'
STYLE = {
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.labelcolor': DARK,
    'text.color': DARK, 'xtick.color': DARK, 'ytick.color': DARK,
    'axes.edgecolor': '#78828C', 'axes.linewidth': .65,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
}


def arrow(ax, xy, end, color=DARK, style='-', curve=0):
    ax.add_patch(FancyArrowPatch(xy, end, arrowstyle='-|>', mutation_scale=11,
                                linewidth=1.1, color=color, linestyle=style,
                                connectionstyle=f'arc3,rad={curve}'))


def save(fig, stem, output, audit):
    for ext in ('pdf', 'svg'):
        fig.savefig(output/f'{stem}.{ext}', bbox_inches=None)
    fig.savefig(audit/f'{stem}.png', dpi=180, bbox_inches=None)
    plt.close(fig)


def plate(ax, x, y, w, h, text='', color=BLUE, fill='#EFF6FA', fs=9):
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0,rounding_size=0.7',
                              linewidth=.9, edgecolor=color, facecolor=fill))
    if text:
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,linespacing=1.4)


def matrix(ax, x, y, w, h, color=GREEN, nr=4, nc=3):
    from matplotlib.colors import to_rgb
    base=np.array(to_rgb(color))
    for i in range(nr):
        for j in range(nc):
            strength=.13+.11*((i*3+j*2)%4)
            fill=1-(1-base)*strength
            ax.add_patch(Rectangle((x+j*w/nc,y+i*h/nr),w/nc,h/nr,
                                   edgecolor='white',linewidth=.65,facecolor=fill))
    ax.add_patch(Rectangle((x,y),w,h,edgecolor=color,linewidth=.9,facecolor='none'))


def pointop(ax,x,y,label,color=DARK):
    ax.add_patch(Circle((x,y),1.3,facecolor='white',edgecolor=color,linewidth=.85,zorder=5))
    ax.text(x,y,label,ha='center',va='center',fontsize=10,color=color,zorder=6)


def route(ax,points,color=DARK,ls='-',head=True,lw=1):
    x,y=zip(*points)
    ax.plot(x,y,color=color,lw=lw,ls=ls,zorder=1)
    if head:
        arrow(ax,points[-2],points[-1],color,ls)


def overview(output,audit):
    # Nested architectural views, following the overview/detail organization of
    # ViT3 Fig. 2; original layout, math and artwork for the MemSolve operator.
    fig=plt.figure(figsize=(6.5,4.05))
    ax=fig.add_axes([.018,.02,.964,.96]);ax.set(xlim=(0,100),ylim=(6,100));ax.axis('off')
    ax.text(1,97,r'(a) Block $\times 12$',fontsize=10,weight='bold')
    plate(ax,7,80,14,8,'3 × 3\nDWConv',BLUE,'#EFF6FA',8)
    ax.text(14,76,'CPE',fontsize=8,ha='center',color=BLUE)
    plate(ax,29,80,9,8,'LN',DARK,'#F6F7F8',8.5)
    plate(ax,42,80,19,8,'MemSolve',GREEN,'#EEF7F2',9)
    plate(ax,70,80,9,8,'LN',DARK,'#F6F7F8',8.5)
    plate(ax,83,80,11,8,'GELU\nFFN',ORANGE,'#FFF4E9',8)
    for x in (25,66,98):pointop(ax,x,84,'+')
    for start,end in [((1,84),(7,84)),((21,84),(23.6,84)),((26.4,84),(29,84)),
                      ((38,84),(42,84)),((61,84),(64.6,84)),((67.4,84),(70,84)),
                      ((79,84),(83,84)),((94,84),(96.6,84))]:arrow(ax,start,end)
    route(ax,[(4,84),(4,92),(25,92),(25,85.4)],BLUE)
    route(ax,[(27.5,84),(27.5,92),(66,92),(66,85.4)],GREEN)
    route(ax,[(68.5,84),(68.5,92),(98,92),(98,85.4)],ORANGE)
    ax.plot([1,99],[66,66],color='#CCD3D9',lw=.75)
    ax.text(1,62,'(b) MemSolve',fontsize=10,weight='bold')
    ax.text(3,37,r'$X$',fontsize=12,ha='center')
    plate(ax,10,32,12,10,r'$W_{BC}$',DARK,'#F6F7F8',10)
    arrow(ax,(5,37),(10,37))
    route(ax,[(22,37),(26,37),(26,51),(29,51)],BLUE)
    route(ax,[(22,37),(26,37),(26,28),(29,28)],ORANGE)
    matrix(ax,29,46,5,10,BLUE,nr=6,nc=3)
    ax.text(31.5,58,r'$B$',fontsize=10,ha='center',color=BLUE)
    plate(ax,40,46,24,10,r'$P=AR^{-1}$'+'\n'+r'$A=B/\sqrt{N}$',BLUE,'#EFF6FA',8)
    arrow(ax,(34,51),(40,51),BLUE)
    matrix(ax,70,46,5,10,BLUE,nr=6,nc=3)
    arrow(ax,(64,51),(70,51),BLUE)
    ax.text(72.5,58,r'$P\;[N\times r]$',fontsize=8.5,ha='center',color=BLUE)
    matrix(ax,29,22,6,12,ORANGE,nr=6,nc=4)
    ax.text(32,18,r'$C\;[N\times h]$',fontsize=8.5,ha='center',color=ORANGE)
    pointop(ax,44,28,'×',GREEN); arrow(ax,(35,28),(42.5,28),ORANGE)
    route(ax,[(72.5,46),(72.5,41),(44,41),(44,29.4)],BLUE)
    ax.text(47,35,r'$P^{\mathsf{T}}$',fontsize=9,color=BLUE)
    matrix(ax,52,25,6,6,GREEN,nr=3,nc=4)
    arrow(ax,(45.5,28),(52,28),GREEN)
    ax.text(55,34,r'$Z$',fontsize=10,ha='center',color=GREEN)
    ax.text(55,21,r'$r\times h$',fontsize=8,ha='center',color=GREEN)
    plate(ax,64,24,14,8,r'$F-\eta I$',GREEN,'#EEF7F2',10)
    arrow(ax,(58,28),(64,28),GREEN)
    ax.text(71,19,r'$r\times r$',fontsize=8,ha='center',color=GREEN)
    pointop(ax,85,28,'×',BLUE);arrow(ax,(78,28),(83.5,28),GREEN)
    route(ax,[(72.5,44),(85,44),(85,29.4)],BLUE)
    ax.text(87.5,35,r'$P$',fontsize=9,color=BLUE)
    pointop(ax,94,28,'+',ORANGE);arrow(ax,(86.5,28),(92.5,28),GREEN)
    arrow(ax,(95.5,28),(99,28),ORANGE)
    ax.text(98,33,r'$Y$',fontsize=11,ha='center',color=ORANGE)
    route(ax,[(32,22),(32,10),(94,10),(94,26.5)],ORANGE)
    ax.text(46,12,r'$\times\eta$',fontsize=10,color=ORANGE)
    save(fig,'operator_overview',output,audit)


def graphical_abstract(output,audit):
    fig=plt.figure(figsize=(6.5,2.5))
    ax=fig.add_axes([.018,.015,.964,.97]);ax.set(xlim=(0,100),ylim=(3,57));ax.axis('off')
    ax.text(1,54,'(a) Sequential updates',fontsize=9.5,weight='bold')
    ax.text(43,54,'(b) MemSolve',fontsize=9.5,weight='bold')
    ax.plot([39,39],[8,51],color='#CCD3D9',lw=.8)
    # Generic ordered memory chain: motivation, not an equivalence claim.
    for x,label in [(3,r'$U_0$'),(14,r'$U_1$'),(29,r'$U_t$')]:
        matrix(ax,x,27,6,7,color='#677481',nr=3,nc=3)
        ax.text(x+3,23,label,ha='center',fontsize=9)
    arrow(ax,(9,30.5),(14,30.5),'#677481')
    arrow(ax,(20,30.5),(25,30.5),'#677481')
    ax.text(27,30.5,'⋯',ha='center',va='center',fontsize=13,color='#677481')
    for x,label in [(14,r'$(k_1,c_1)$'),(29,r'$(k_t,c_t)$')]:
        plate(ax,x-1,41,8,5,'',BLUE,'#EFF6FA')
        ax.text(x+3,43.5,label,ha='center',va='center',fontsize=8)
        arrow(ax,(x+3,40.5),(x+3,35),BLUE)
    ax.text(19,12,r'$U_t=U_{t-1}+\beta_t k_t e_t^{\mathsf{T}}$'+'\n'+r'$e_t=c_t-U_{t-1}^{\mathsf{T}}k_t$',
            ha='center',va='center',fontsize=9,linespacing=1.7,color='#56616D')
    # All tokens participate in one compact state, followed by shared readout.
    colors=[BLUE]*5
    for i,y in enumerate([17,22,27,32,37]):
        plate(ax,44,y,5,3,'',colors[i],'#EFF6FA')
        arrow(ax,(49,y+1.5),(64,28),color='#5D8AA4')
        plate(ax,94,y,5,3,'',ORANGE,'#FFF4E9')
        arrow(ax,(81,28),(94,y+1.5),color='#B07D52')
    plate(ax,64,21,17,14,'',GREEN,'#F5FAF7')
    matrix(ax,66,24,7,8,GREEN,nr=3,nc=4)
    ax.text(77,28,r'$U^\star$',fontsize=12,ha='center',va='center',color=GREEN)
    ax.text(72.5,17,r'$r\times h$',fontsize=8,ha='center',color=GREEN)
    plate(ax,64,43,17,6,r'$J$',GREEN,'#EEF7F2',8)
    arrow(ax,(72.5,43),(72.5,35.5),GREEN)
    ax.text(55.5,40,r'$P^{\mathsf{T}}$',fontsize=10,ha='center',color=BLUE)
    ax.text(87.5,40,r'$\mathcal{R}$',fontsize=10,ha='center',color=ORANGE)
    ax.text(46.5,12,r'$X$',fontsize=10,ha='center')
    ax.text(95.5,12,r'$Y$',fontsize=10,ha='center')
    ax.text(72.5,7,r'$(I+J)U^\star=P^{\mathsf{T}}C$',fontsize=10,ha='center',color=GREEN)
    save(fig,'graphical_abstract',output,audit)


def curves(evidence, output, audit):
    data = json.loads(evidence.read_text())
    runs = {r['id']: r for r in data['runs']}
    ids = ['small1024', 'small2048ra', 'complement', 'equilibrium']
    provenance = []
    for key in ids:
        run = runs[key]
        raw = Path(run['path'])/'metrics.jsonl'
        if hashlib.sha256(raw.read_bytes()).hexdigest() != run['metrics_sha256']:
            raise ValueError(f'Source metrics changed: {raw}')
        original = [json.loads(line) for line in raw.read_text().splitlines()]
        expected = [{k: r[k] for k in ('epoch', 'val_acc1', 'val_acc5', 'val_loss')} for r in original]
        if expected != run['rows']:
            raise ValueError(f'Evidence is inconsistent with source: {raw}')
        provenance.append({'id': key, 'path': str(raw), 'sha256': run['metrics_sha256']})
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.25), layout='constrained')
    for row, specs in enumerate([
        [('small1024', 'Batch 1,024; no RA', BLUE, '-'),
         ('small2048ra', 'Batch 2,048; RA', ORANGE, '--')],
        [('complement', 'Complement', BLUE, '--'),
         ('equilibrium', 'Equilibrium', GREEN, '-')],
    ]):
        for key, label, color, style in specs:
            records = runs[key]['rows']; last = max(r['epoch'] for r in records)
            by_epoch = {r['epoch']: r for r in records}
            if len(by_epoch) != len(records):
                raise ValueError(f'Duplicate epoch in {key}')
            epochs = np.arange(1, last+1)
            for col, field in enumerate(['val_acc1', 'val_loss']):
                values = [by_epoch.get(int(e), {}).get(field, np.nan) for e in epochs]
                axs[row, col].plot(epochs, values, color=color, ls=style, lw=1.35, label=label)
        for col in range(2):
            ax = axs[row, col]
            ax.grid(axis='y', color='#E4E8EC', lw=.65)
            ax.set_xlabel('Epoch'); ax.set_xlim(0, 300 if row == 0 else 30)
            ax.set_ylabel('Validation Top-1 (%)' if col == 0 else 'Validation CE')
            ax.set_ylim(bottom=0)
            title = 'ImageNet-1K' if row == 0 else 'ImageNet-100'
            ax.set_title(f'({chr(97+row*2+col)}) {title}', loc='left', fontsize=10, weight='bold')
        axs[row, 0].legend(frameon=False, fontsize=8, loc='lower right')
    save(fig, 'training_curves', output, audit)
    (audit/'figure_provenance.json').write_text(json.dumps({
        'sources': provenance, 'transformations': 'None; every epoch; no smoothing; missing epochs are gaps',
        'uncertainty': 'One training seed per configuration; no uncertainty interval estimated',
        'overview': 'Original schematic; main.tex equations frame, solve and readout',
        'review_status': 'Agent-generated draft; author scientific review pending',
        'matplotlib': matplotlib.__version__,
        'files': ['graphical_abstract.pdf', 'graphical_abstract.svg', 'operator_overview.pdf', 'operator_overview.svg', 'training_curves.pdf', 'training_curves.svg'],
        'alt_text': {
            'graphical_abstract': 'A generic ordered memory chain motivates a full-context alternative: all tokens contribute to one compact equilibrium state, and all outputs use that state. This is a conceptual comparison, not numerical equivalence to the chain.',
            'operator_overview': 'The visual block has residual CPE, a pre-LayerNorm MemSolve branch and a pre-LayerNorm GELU FFN branch. One MemSolve head forms P, writes Z=P transpose C, transforms Z by F minus eta I, and reads through P before adding eta C.',
            'training_curves': 'Four panels plot validation Top-1 and cross-entropy across all 300 ImageNet-1K epochs and all 30 ImageNet-100 epochs. Both Small runs improve through training; the learned-core variant finishes above the complement variant in the short comparison.'}
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    parser.add_argument('--audit-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True); args.audit_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        overview(args.output, args.audit_dir)
        graphical_abstract(args.output, args.audit_dir)
        curves(args.evidence, args.output, args.audit_dir)


if __name__ == '__main__':
    main()
