\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{booklet}[2024/10/02 Dissertation booklet]
\LoadClass{book}

\RequirePackage{dissertation}

\RequirePackage{afterpage}
\RequirePackage{float}
\RequirePackage{ragged2e}
\RequirePackage[T1]{fontenc}
\RequirePackage[cmintegrals,cmbraces]{newtxmath}
\RequirePackage{fancyhdr}
\RequirePackage{tocloft}
\RequirePackage[font={footnotesize, sf},labelfont=bf]{caption}
\RequirePackage{opensans}
\RequirePackage{titlesec}
\RequirePackage{ebgaramond}
\RequirePackage{multicol}

% Set page size ================================================================
\RequirePackage[paperheight=240mm,paperwidth=160mm,bottom=30mm]{geometry}


% Full page figure =============================================================



% Custom command to position the caption in the bottom-right

\setlength{\fullpagefigwidth}{\paperwidth}
\newcommand{\fullpagefig}[3][bottomleft]{%
    \afterpage{%
       \clearpage
       \pagecolor{lightgray} % Set background color
       \pagestyle{empty}    % Remove headers/footers
       % Begin the figure
       \begin{figure}[H]
            \centering
            \begin{tikzpicture}[remember picture, overlay]  % Overlay environment
                % Place the figure content (image, TikZ, PGFPlot, etc.)
                \node[anchor=center,inner sep=0] (image) at (current page.center) {#2};
                % Overlay caption
                \ifthenelse{\equal{#1}{bottomleft}}{%
                  \node[
                      anchor=south west,
                      text width=0.5\paperwidth,
                      inner sep=1cm,
                  ] at (current page.south west) {%
                    \justifying#3
                  };
                }{%
                  \node[
                      anchor=north west,
                      text width=0.5\paperwidth,
                      inner sep=1cm,
                  ] at (current page.north west) {%
                    \justifying#3
                  };
                }
            \end{tikzpicture}
       \end{figure}
       \clearpage
       \pagecolor{white}  % Reset background color to white for next pages
       \pagestyle{plain}  % Reset to default page style
       \clearpage
    }
}


% Setup font ===================================================================

% Serif font: Garamond
%\usepackage{ebgaramond-maths}

% Sans-serif family: Open-Sans
\renewcommand{\sffamily}{\opensans}

% Setup titles
\titleformat{\part}[display]{\pagecolor{lightgray}\vskip10em\bfseries}{\sffamily\Large\partname\
	\thepart}{10pt}{\Huge}[\afterpage{\nopagecolor}]
\titleformat{\chapter}[display]{\bfseries}{\sffamily\Large\chaptertitlename\ \thechapter}{10pt}{\Huge}
\titleformat{\section}{\sffamily\Large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\sffamily\large\bfseries}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\sffamily\bfseries}{\thesubsubsection}{1em}{}
%\titlespacing*{\subsubsection}{0pt}{1em}{0pt}

% Setup captions
\captionsetup{justification=justified,singlelinecheck=off}

% Set bibliography font size
\renewcommand*{\bibfont}{\footnotesize}


%% Redefine the figure environment
%\renewenvironment{figure}{%
%  \begin{figure}\sffamily\sansmath%
%}{%
%  \end{figure}
%}
%
%% Redefine the table environment
%\renewenvironment{table}{%
%  \begin{table}\sffamily\sansmath%
%}{%
%  \end{table}
%}
%
%% Redefine the algorithm environment
%\renewenvironment{algorithm}{%
%  \begin{algorithm}\sffamily\sansmath%
%}{%
%  \end{algorithm}
%}

% Setup pagestyle ==============================================================

% Capture part and chapter title
\newcommand*\parttitle{}
\let\origpart\part%
\renewcommand*{\part}[2][]{%
	\ifx\\#1\\% optional argument not present?
	\origpart{#2}%
		\renewcommand*\parttitle{#2}%
	\else
	\origpart[#1]{#2}%
		\renewcommand*\parttitle{#1}%
	\fi
}

\renewcommand{\chaptermark}[1]{\markboth{\thechapter\hskip 1em #1}{}}
\newcommand{\PartmarkHead}{
	\ifx\parttitle\empty\else\partname\ \thepart:\ \parttitle\fi
}
% Setup header
\fancypagestyle{front}{%
	\fancyhf{} % clear all header and footer fields
	\fancyhead[RO]{\sffamily\thepage\hfill}%
	\fancyhead[LE]{\sffamily\hfill\thepage}%
}
\fancypagestyle{main}{%
	\fancyhf{} % clear all header and footer fields
	\fancyhead[RO]{\sffamily\thepage\hfill\PartmarkHead}%
	\fancyhead[LE]{\sffamily\leftmark\hfill\thepage}%
}
\fancypagestyle{plain}{%
	\fancyhf{} % clear all header and footer fields
}
\renewcommand{\headrulewidth}{0pt}

\let\oldfrontmatter\frontmatter
\renewcommand{\frontmatter}{\oldfrontmatter\pagestyle{front}}
\let\oldbackmatter\backmatter
\renewcommand{\backmatter}{\oldbackmatter\pagestyle{front}}
\let\oldmainmatter\mainmatter
\renewcommand{\mainmatter}{\oldmainmatter\pagestyle{main}}

% Setup tables==================================================================

% Setup ToC
\renewcommand\cftpartfont{\sffamily\bfseries}
\renewcommand\cftpartpagefont{\sffamily\bfseries}
\renewcommand\cftchapfont{\sffamily}
\renewcommand\cftchappagefont{\sffamily}
\renewcommand\cftsecfont{\sffamily\small}
\renewcommand\cftsecpagefont{\sffamily}
\renewcommand\cftsubsecfont{\sffamily\footnotesize}
\renewcommand\cftsubsecpagefont{\sffamily}
\makeatletter
\renewcommand\@cftmaketoctitle{\chapter{\contentsname}}
\makeatother
%\setcounter{tocdepth}{1}

% Setup LoF
\renewcommand\cftfigfont{\sffamily\footnotesize}
\renewcommand\cftfigpagefont{\sffamily\footnotesize}
\makeatletter
\renewcommand\@cftmakeloftitle{\chapter{\listfigurename}}

% Setup LoT
\renewcommand\cfttabfont{\sffamily\footnotesize}
\renewcommand\cfttabpagefont{\sffamily\footnotesize}
\makeatletter
\renewcommand\@cftmakelottitle{\chapter{\listtablename}}
\makeatother
\makeatother
