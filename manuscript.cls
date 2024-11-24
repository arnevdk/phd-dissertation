\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{manuscript}[2024/10/02 Dissertation manuscript]
\LoadClass[a4paper,oneside]{report}

\RequirePackage{dissertation}
\RequirePackage[pass]{geometry}
\RequirePackage[modulo]{lineno}
\RequirePackage{floatpag}

% Full page figure
\setlength{\fullpagefigwidth}{\textwidth}
\newcommand{\fullpagefig}[3][bottomleft]{%
  \begin{figure}[p]
    \thisfloatpagestyle{empty}
    #2
    #3
  \end{figure}
}

\renewcommand{\epigraph}[2]{}
