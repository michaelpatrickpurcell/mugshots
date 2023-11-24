import tikz
import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from itertools import product

from pylatex import Document, TikZ
from pylatex import Package, Command
from pylatex.utils import italic, bold, NoEscape
from pylatex.basic import NewLine

block_indices = [
    [0,1,3,7,15,31,36,54,63],
    [0,2,6,14,30,35,53,62,72],
    [0,4,12,28,33,51,60,70,71],
    [0,5,23,32,42,43,45,49,57],
    [0,8,24,29,47,56,66,67,69],
    [0,9,19,20,22,26,34,50,55],
    [0,10,11,13,17,25,41,46,64],
    [0,16,21,39,48,58,59,61,65],
    [0,18,27,37,38,40,44,52,68],
    [1,2,4,8,16,32,37,55,64],
    [1,5,13,29,34,52,61,71,72],
    [1,6,24,33,43,44,46,50,58],
    [1,9,25,30,48,57,67,68,70],
    [1,10,20,21,23,27,35,51,56],
    [1,11,12,14,18,26,42,47,65],
    [1,17,22,40,49,59,60,62,66],
    [1,19,28,38,39,41,45,53,69],
    [2,3,5,9,17,33,38,56,65],
    [2,7,25,34,44,45,47,51,59],
    [2,10,26,31,49,58,68,69,71],
    [2,11,21,22,24,28,36,52,57],
    [2,12,13,15,19,27,43,48,66],
    [2,18,23,41,50,60,61,63,67],
    [2,20,29,39,40,42,46,54,70],
    [3,4,6,10,18,34,39,57,66],
    [3,8,26,35,45,46,48,52,60],
    [3,11,27,32,50,59,69,70,72],
    [3,12,22,23,25,29,37,53,58],
    [3,13,14,16,20,28,44,49,67],
    [3,19,24,42,51,61,62,64,68],
    [3,21,30,40,41,43,47,55,71],
    [4,5,7,11,19,35,40,58,67],
    [4,9,27,36,46,47,49,53,61],
    [4,13,23,24,26,30,38,54,59],
    [4,14,15,17,21,29,45,50,68],
    [4,20,25,43,52,62,63,65,69],
    [4,22,31,41,42,44,48,56,72],
    [5,6,8,12,20,36,41,59,68],
    [5,10,28,37,47,48,50,54,62],
    [5,14,24,25,27,31,39,55,60],
    [5,15,16,18,22,30,46,51,69],
    [5,21,26,44,53,63,64,66,70],
    [6,7,9,13,21,37,42,60,69],
    [6,11,29,38,48,49,51,55,63],
    [6,15,25,26,28,32,40,56,61],
    [6,16,17,19,23,31,47,52,70],
    [6,22,27,45,54,64,65,67,71],
    [7,8,10,14,22,38,43,61,70],
    [7,12,30,39,49,50,52,56,64],
    [7,16,26,27,29,33,41,57,62],
    [7,17,18,20,24,32,48,53,71],
    [7,23,28,46,55,65,66,68,72],
    [8,9,11,15,23,39,44,62,71],
    [8,13,31,40,50,51,53,57,65],
    [8,17,27,28,30,34,42,58,63],
    [8,18,19,21,25,33,49,54,72],
    [9,10,12,16,24,40,45,63,72],
    [9,14,32,41,51,52,54,58,66],
    [9,18,28,29,31,35,43,59,64],
    [10,15,33,42,52,53,55,59,67],
    [10,19,29,30,32,36,44,60,65],
    [11,16,34,43,53,54,56,60,68],
    [11,20,30,31,33,37,45,61,66],
    [12,17,35,44,54,55,57,61,69],
    [12,21,31,32,34,38,46,62,67],
    [13,18,36,45,55,56,58,62,70],
    [13,22,32,33,35,39,47,63,68],
    [14,19,37,46,56,57,59,63,71],
    [14,23,33,34,36,40,48,64,69],
    [15,20,38,47,57,58,60,64,72],
    [15,24,34,35,37,41,49,65,70],
    [16,25,35,36,38,42,50,66,71],
    [17,26,36,37,39,43,51,67,72],
]

blacklist = [2,4,5,7,9,13,17,19,20,25,32,33,37,38,51,57,61,63,66,67,74,83,87,93,94,95,96]
replace_list = [2,25,83,96]

def generate_locations(seed=None):
    n_rows = 7
    n_cols = 11

    locations = [0.66 * (np.array([i,j]) - np.array((n_cols/2, n_rows/2))) for i in range(n_cols) for j in range(n_rows)]

    if seed is None:
        seed = np.random.randint(2**31)
    print('Initial seed = %i' % seed)
    np.random.seed(seed)

    np.random.shuffle(locations)

    return locations, seed


def generate_faces_grid(face_indices, n_rows, n_cols):

    locations = [0.66 * (np.array([i,j]) - np.array((n_cols/2, n_rows/2))) for i in range(n_cols) for j in range(n_rows)]
    np.random.shuffle(locations)

    pic = tikz.Picture()
    # pic.usetikzlibrary('shapes.geometric')

    for i,location in enumerate(locations):
        point = '(%fin, %fin)' % (location[0], location[1])
        pic.node(r'\includegraphics[height = 0.5in]{Images/bw_face_icons_individuals/face_%i.png}' % face_indices[i], at=point)

    tikzpicture = pic.code()
    return tikzpicture

def generate_latex_doc(tikzpicture, scale, seqno):
    geometry_options = {'top': '1.25cm', 'bottom': '1.0cm', 'left': '7cm', 'right':'7cm', 'marginparwidth': '6.0cm', 'marginparsep': '0pt'}
    doc = Document(documentclass = 'scrartcl',
                document_options = ["paper=a4","parskip=half", "landscape"],
                fontenc=None,
                inputenc=None,
                lmodern=False,
                textcomp=False,
                page_numbers=False,
                geometry_options=geometry_options)

    doc.packages.append(Package('tikz'))
    doc.packages.append(Package('fontspec'))
    doc.packages.append(Package('enumitem'))
    doc.packages.append(Package('booktabs'))
    doc.packages.append(Package('marginnote'))

    doc.preamble.append(Command('setkomafont', NoEscape(r'section}{\setmainfont{Century Gothic}\LARGE\bfseries')))
    doc.preamble.append(Command('RedeclareSectionCommand', 'section', ([r'runin=false', NoEscape(r'afterskip=0.0\baselineskip'), NoEscape(r'beforeskip=1.0\baselineskip')])))
    doc.change_length("\columnsep", "10mm")

    doc.append(NoEscape(r'\begin{center}\setmainfont[Scale=2.5]{Century Gothic}\Huge \textbf{Mugshots}\end{center}'))

    doc.append(Command(NoEscape(r'setmainfont{TeX Gyre Schola}')))
    doc.append(Command(NoEscape(r'raggedright')))

    doc.append(Command(r'vspace{0.0cm}'))
    doc.append(Command(r'vspace{-0.5cm}'))

    doc.append(Command(NoEscape(r'begin{center}')))
    doc.append(NoEscape(r'\scalebox{%f}{' % scale))
    doc.append(NoEscape(tikzpicture))
    doc.append(NoEscape(r'}'))

    f = open('rules_text.tex')
    rules_text = f.read()
    f.close()
    doc.append(NoEscape(rules_text))

    doc.append(Command(NoEscape(r'vfill')))
    doc.append(NoEscape(r'{\LARGE Designed by Michael Purcell}'))
    doc.append(NoEscape(r'\normalmarginpar\marginnote{\raggedright\textbf{Sequence Number}: %i / 73}' % (seqno)))
    doc.append(NoEscape(r'\reversemarginpar\marginnote{\raggedright\textbf{Contact}: ttkttkt@gmail.com}'))
    doc.append(Command(NoEscape(r'end{center}')))

    return doc


def generate_gallery_doc(tikzpicture, scale):
    geometry_options = {'top': '1.25cm', 'bottom': '1.0cm', 'left': '1cm', 'right':'1cm'}
    doc = Document(documentclass = 'scrartcl',
                document_options = ["paper=a4","parskip=half", "landscape"],
                fontenc=None,
                inputenc=None,
                lmodern=False,
                textcomp=False,
                page_numbers=False,
                geometry_options=geometry_options)

    doc.packages.append(Package('tikz'))
    doc.packages.append(Package('fontspec'))
    doc.packages.append(Package('enumitem'))
    doc.packages.append(Package('booktabs'))
    doc.packages.append(Package('marginnote'))

    doc.preamble.append(Command('setkomafont', NoEscape(r'section}{\setmainfont{Century Gothic}\LARGE\bfseries')))
    doc.preamble.append(Command('RedeclareSectionCommand', 'section', ([r'runin=false', NoEscape(r'afterskip=0.0\baselineskip'), NoEscape(r'beforeskip=1.0\baselineskip')])))
    doc.change_length("\columnsep", "10mm")

    doc.append(NoEscape(r'\begin{center}\setmainfont[Scale=2.5]{Century Gothic}\Huge \textbf{Mugshots Gallery}\end{center}'))

    doc.append(Command(NoEscape(r'setmainfont{TeX Gyre Schola}')))
    doc.append(Command(NoEscape(r'raggedright')))

    doc.append(Command(r'vspace{0.0cm}'))
    doc.append(Command(r'vspace{-0.5cm}'))

    doc.append(Command(NoEscape(r'begin{center}')))
    doc.append(NoEscape(r'\scalebox{%f}{' % scale))
    doc.append(NoEscape(tikzpicture))
    doc.append(NoEscape(r'}'))

    doc.append(Command(NoEscape(r'end{center}')))

    return doc


def save_tikzpicture(tikzpicture, filename):
    f = open(filename, 'w')
    # f.write(r'\documentclass{standalone}')
    # f.write(r'\usepackage{tikz}')
    # f.write(r'\begin{document}')
    f.write(tikzpicture)
    # f.write(r'\end{document}')   
    f.close()

def save_latex_doc(latex_doc, filename):
    latex_doc.generate_tex(filename)#, compiler='xelatex')


if __name__ == '__main__':
    seed = None
    # seed = 983291822
    # locations, seed = generate_locations(seed=seed)
    # print(seed)

    np.random.shuffle(block_indices)

    face_indices = np.arange(1,101)
    filtered_face_indices = [i for i in face_indices if i not in blacklist]
    filtered_face_indices += replace_list
    filtered_face_indices = np.array(filtered_face_indices)

    tikzpicture_7x11 = generate_faces_grid(filtered_face_indices[:77], n_rows=7, n_cols=11)
    # save_tikzpicture(tikzpicture=tikzpicture_7x11, filename='playtest/faces_grid_7x11.tex')
    gallery = generate_gallery_doc(tikzpicture_7x11, scale=1.4125)
    save_latex_doc(gallery, 'PnPFiles/mugshots_gallery')

    tikzpictures_3x3 = []
    for i,block in enumerate(block_indices):
        brady_bunch = generate_faces_grid(filtered_face_indices[block], n_rows=3, n_cols=3)
        tikzpictures_3x3.append(brady_bunch)
        # save_tikzpicture(tikzpicture=brady_bunch, filename='playtest/faces_grid_%i.tex' %i)

    # tikzpicture_3x3b = generate_faces_grid(face_indices[8:17], n_rows=3, n_cols=3)
    # save_tikzpicture(tikzpicture=tikzpicture_3x3b, filename='faces_grid_3x3b.tex')

    latex_docs = []
    for i,tikzpicture in enumerate(tikzpictures_3x3):
        latex_docs.append(generate_latex_doc(tikzpicture, scale=2.88, seqno=i+1))

    for i,latex_doc in enumerate(latex_docs):
        save_latex_doc(latex_doc, 'PnPFiles/mugshots_sheet_%i' % (i+1))


    subprocess.call(["xelatex", "PnPFiles/mugshots_gallery.tex", "--output-directory=PnPFiles"])
    subprocess.call(["xelatex", "PnPFiles/mugshots_gallery.tex", "--output-directory=PnPFiles"])
    os.remove("PnPFiles/mugshots_gallery.tex")
    os.remove("PnPFiles/mugshots_gallery.aux")
    os.remove("PnPFiles/mugshots_gallery.log")

    for i in range(73):
        subprocess.call(["xelatex", "PnPFiles/mugshots_sheet_%i.tex" % (i+1), "--output-directory=PnPFiles"])    
        subprocess.call(["xelatex", "PnPFiles/mugshots_sheet_%i.tex" % (i+1), "--output-directory=PnPFiles"])    
        os.remove("PnPFiles/mugshots_sheet_%i.tex" % (i+1))
        os.remove("PnPFiles/mugshots_sheet_%i.aux" % (i+1))
        os.remove("PnPFiles/mugshots_sheet_%i.log" % (i+1))