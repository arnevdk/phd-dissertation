\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{manuscript}[2024/10/02 Dissertation manuscript]
\LoadClass[a4paper,oneside]{report}

\RequirePackage{dissertation}
\RequirePackage[pass]{geometry}
\RequirePackage{lineno}

% Line numbers


% Full page figure
\setlength{\fullpagefigwidth}{\textwidth}
\newcommand{\fullpagefig}[5][bottomleft]{%
  \begin{figure}[p]
    #2
    \caption[#3]{#4}%
    \label{#5}
  \end{figure}
}
