"""Build the six-figure manuscript assets and separate supplementary figures."""
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / 'paper'
TABLES = ['fast_policy_levels', 'e1_policy_contrasts', 'e1_resource_configurations',
          'spatial_consequences', 'e2_action_contrasts', 'e2_fast_objective']


def latex(text):
    """Convert the caption's limited Markdown and mathematical notation."""
    protected = []
    def keep(value):
        protected.append(value)
        return f'ZZLATEX{len(protected)-1}ZZ'
    text = re.sub(r'\$[^$]+\$', lambda m: keep(m.group()), text)
    for token, value in [('10⁻¹⁰', r'$10^{-10}$'), ('D_F', r'$D_F$'),
                         ('M_F', r'$M_F$'), ('M_O', r'$M_O$')]:
        text = text.replace(token, keep(value))
    text = re.sub(r'\bw\s*=\s*([01])\b', lambda m: keep('$w=' + m[1] + '$'), text)
    text = text.replace('−', keep(r'$-$')).replace('→', keep(r'$\rightarrow$'))
    text = text.replace('–', '--').replace('—', '---').replace('’', "'").replace('‘', "'")
    text = text.replace('“', '``').replace('”', "''")
    text = re.sub(r'[\\{}%&#_$]', lambda m: {'\\': r'\textbackslash{}', '{': r'\{', '}': r'\}',
                  '%': r'\%', '&': r'\&', '#': r'\#', '_': r'\_', '$': r'\$'}[m.group()], text)
    text = text.replace('~', r'\textasciitilde{}').replace('^', r'\textasciicircum{}')
    for i, value in enumerate(protected):
        text = text.replace(f'ZZLATEX{i}ZZ', value)
    return text


def read_captions(path):
    text = path.read_text()
    matches = re.finditer(r'\*\*Figure\s+(S?\d+)\. (.*?)\*\*\s*(.*?)(?=\n\*\*Figure\s+|\Z)', text, re.S)
    return {m[1]: (m[2], ' '.join(m[3].split())) for m in matches}


def preamble(title, supplementary=False):
    text = r'''\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=0.6in]{geometry}
\usepackage{graphicx,booktabs,amsmath}
\usepackage{caption}
\captionsetup{font=normalsize,labelfont=bf,justification=raggedright,singlelinecheck=false,skip=10pt}
\setlength{\parindent}{0pt}
\setlength{\textfloatsep}{18pt}
\setlength{\floatsep}{18pt}
\renewcommand{\arraystretch}{1.2}
\makeatletter
\setlength{\@fptop}{0pt}
\makeatother
NUMBERING
\begin{document}
\begin{center}
{\Large\bfseries TITLE\par}
\vspace{0.4em}
{\normalsize SUBTITLE}
\end{center}
\vspace{0.5em}
'''
    return text.replace('SUBTITLE', 'Supplementary figures' if supplementary else 'Figures and tables for independent review').replace(
        'TITLE', latex(title)
    ).replace('NUMBERING', r'\renewcommand{\thefigure}{S\arabic{figure}}' if supplementary else '')


def compile_pdf(name):
    pdflatex = shutil.which('pdflatex')
    if not pdflatex:
        raise RuntimeError('pdflatex is required; TinyTeX provides it in this workspace.')
    for _ in range(2):
        result = subprocess.run([pdflatex, '-interaction=nonstopmode', '-halt-on-error', name + '.tex'],
                                cwd=PAPER, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout[-6000:] + result.stderr)
    log = (PAPER / (name + '.log')).read_text(errors='replace')
    problems = re.findall(r'(?:Overfull|Underfull) \\[hv]box[^\n]*|LaTeX Warning: Float too large[^\n]*', log)
    if problems:
        raise RuntimeError(name + ':\n' + '\n'.join(problems))
    print(f'{name}: clean layout; {PAPER / (name + ".pdf")}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-compile', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'figure_manifest.json').read_text())
    records = manifest['figures']
    assert len([r for r in records if r['role'] == 'main']) == manifest['main_count'] == 6
    assert len([r for r in records if r['role'] == 'supplement']) == manifest['supplementary_count'] == 12
    captions = read_captions(ROOT / 'figure_captions.md')
    captions.update(read_captions(ROOT / 'supplementary_figure_captions.md'))
    framework = (ROOT / 'figures/framework_caption.md').read_text().split('\n\nArchitecture source:', 1)[0]
    m = re.search(r'\*\*Figure 2\. (.*?)\*\*\s*(.*)', framework, re.S)
    if not m:
        raise ValueError('Framework caption is missing its Figure 2 heading.')
    captions['2'] = (m[1], ' '.join(m[2].split()))
    title = re.search(r'英文工作标题：\*(.*?)\*', (ROOT / 'task.md').read_text())[1]
    sections = {'main': [], 'supplement': []}
    index = ['# Figure index', '', '| Figure | Label | Source PDF | LaTeX block |', '| --- | --- | --- | --- |']
    for record in sorted(records, key=lambda r: (r['role'], r['order'])):
        if not (ROOT / record['filename']).exists():
            raise FileNotFoundError(record['filename'])
        title_text, body = captions[record['caption_number']]
        if record['role'] == 'main':
            number = re.match(r'fig(\d+)', record['block'])[1]
            block = Path('figure_blocks') / f'fig{number}.tex'
        else:
            block = Path('supplementary_figure_blocks') / f's{record["order"]:02}.tex'
        (PAPER / block).parent.mkdir(parents=True, exist_ok=True)
        text = ('% Generated by build_paper_assets.py; edit the Markdown caption source.\n'
                '\\begin{figure}[!htbp]\n\\centering\n'
                f'\\makebox[\\linewidth][c]{{\\includegraphics[width=7in]{{../{record["filename"]}}}}}\n'
                f'\\caption[{latex(title_text)}]{{\\textbf{{{latex(title_text)}}} {latex(body)}}}\n'
                f'\\label{{{record["label"]}}}\n\\end{{figure}}\n')
        (PAPER / block).write_text(text)
        sections[record['role']].append(f'\\input{{{block.as_posix()}}}')
        index.append(f'| {record["caption_number"]} | `{record["label"]}` | `{record["filename"]}` | `{block}` |')
    # These generated blocks moved to the supplement; their historical copies are archived.
    for stale in ('fig03.tex', 'fig04.tex'):
        (PAPER / 'figure_blocks' / stale).unlink(missing_ok=True)
    for name in TABLES:
        if not (PAPER / 'tables' / (name + '.tex')).exists():
            raise FileNotFoundError(name)
        sections['main'].append(f'\\input{{tables/{name}.tex}}')
    for role, name in [('main', 'figures_and_tables'), ('supplement', 'supplementary_figures')]:
        content = preamble(title, role == 'supplement') + '\n\\clearpage\n'.join(sections[role]) + '\n\\end{document}\n'
        (PAPER / (name + '.tex')).write_text(content)
        if not args.no_compile:
            compile_pdf(name)
    (PAPER / 'figure_index.md').write_text('\n'.join(index) + '\n')


if __name__ == '__main__':
    assert latex('D_F, w = 1, CO_29, 95%, 10⁻¹⁰') == r'$D_F$, $w=1$, CO\_29, 95\%, $10^{-10}$'
    main()
