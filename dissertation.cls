\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{dissertation}[2024/10/02 Dissertation booklet or manuscript]

\RequirePackage{xkeyval}
\RequirePackage{xcolor}
\RequirePackage{pgfplots}
\RequirePackage{amsmath}
\RequirePackage{sansmath}

% Dynamically extend documentclass ============================================

\DeclareOptionX{manuscript}{%
    \gdef\extendclasstype{article}
    \gdef\myclasstype{manuscript}
}
\DeclareOptionX{booklet}{%
    \gdef\extendclasstype{report}
    \gdef\myclasstype{booklet}
}
\ProcessOptionsX
%:pass unrecognised options off to \myclasstype
\DeclareOptionX*{%
    \PassOptionsToClass{\CurrentOption}{report}}
%:process options
\ProcessOptionsX
%:load \myclasstype
\LoadClass{\extendclasstype}% add options if desired


% Colors ======================================================================

\definecolor{accent1}{HTML}{F29018}
\definecolor{accent2}{HTML}{F63D1E}
\definecolor{accent3}{HTML}{78479d}
\definecolor{lightgray}{HTML}{EAEAEA}
\definecolor{darkgray}{HTML}{8D807A}
\definecolor{muteblack}{HTML}{262321}
% Extra accent brownish?



% Figures =====================================================================

% Setup tikz figures
\tikzset{%
  every node/.append style={font=\sffamily}
}

% Setup PGF plots
\RequirePackage[T1]{fontenc}
\RequirePackage{textcomp}
\RequirePackage{pgfplots}
\usepgfplotslibrary{groupplots,dateplot}
\usetikzlibrary{patterns,shapes.arrows}
\pgfplotsset{%
  compat=newest,
  every axis/.append style={%
    scale only axis,
    enlarge y limits=false,
    enlarge x limits=false,
    every tick label/.append style={color=darkgray, font=\sffamily\sansmath\footnotesize},
    every axis label/.append style={color=darkgray, font=\sffamily\sansmath\footnotesize},
    every title/.append style={color=darkgray, font=\sffamily\sansmath\footnotesize},
    axis line style={color=darkgray}, % Set the axis line color to darkgray
    cycle list={%
        {accent1, thick},    % First color (red)
        {accent2, thick},   % Second color (blue)
        {accent3, thick},  % Third color (green)
    },
  }
}

\newlength{\fullpagefigwidth}

% Input option specific =======================================================
\input{cls/\myclasstype.tex}
