from env.global_bill_adaptative_optimiser import GlobalBillAdaptativeOptimiser
from envs import create_env
import os
import click
import numpy as np
from envs import create_env_fcts
from utils.utils import unique_consecutives_values
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import time
import plotly.io as pio
from itertools import product
pio.kaleido.scope.mathjax = None
from time import time
from utils.utils import epsilonify

TABLE_TEMPLATE = r"""
\begin{table}[h]
    \begin{subtable}[h]{1.0\textwidth}
        \centering
        \begin{tabular}{l c c c c}
        \toprule
        Member & \makecell{Buying\\price} & \makecell{Selling\\price} & \makecell{Net consumption(+)\\ Net production(--) \\ (1)} & \makecell{Net consumption(+)\\ Net production(--) \\ (2)} \\
        \midrule
        Unit & $Eur/kWh$ & $Eur/kWh$ & $kWh$ & $kWh$\\
        \midrule 
        %INPUT_LINES%
        \bottomrule
       \end{tabular}
       \label{tab:%PREFIX_INPUT_TABLE%rec_%N_MEMBERS%_example_inputs}
       \caption{}
    \end{subtable}
    \newline
\vspace*{0.05 cm}
\newline
    
    \begin{subtable}[h]{1.0\textwidth}
        \centering
        \begin{tabular}{l c}
        \toprule
        Parameter & Value\\
        \midrule
        $\Delta_C$ & $DELTAC$\\
        $\Delta_M$ & $DELTAM$\\
        $\Delta_B$ & $DELTAB$\\
        $\Lambda^+$ & $LAMBDAPLUS$ $Eur/kWh$\\
        $\Lambda^-$ & $LAMBDAMINUS$ $Eur/kWh$\\
        $P^+$ & $INJECTION_PEAK_COST$ $Eur/kW$\\
        $P^-$ & $OFFTAKE_PEAK_COST$ $Eur/kW$\\
        \midrule
        Bill & Value\\
        \midrule
        Commodity Bill & %COMMODITY_BILL%\\
        Peak Bill & %PEAK_BILL%\\
        \textbf{Total} & \textbf{%TOTAL_BILL%}\\
        \midrule
        \textcolor{cyan}{REC Bill} & \textcolor{cyan}{Value}\\
        \midrule
        \textcolor{cyan}{Commodity Bill} & \textcolor{cyan}{%REC_COMMODITY_BILL%}\\
        \textcolor{cyan}{Peak Bill} & \textcolor{cyan}{%REC_PEAK_BILL%}\\
        \textbf{\textcolor{cyan}{Global REC Bill}} & \textbf{\textcolor{cyan}{%REC_GLOBAL_BILL%}}\\
        \bottomrule
        \end{tabular}
        \label{tab:%PREFIX_INPUT_TABLE%rec_%N_MEMBERS%_example_bills}
        \caption{}
     \end{subtable}
    \newline
\vspace*{0.05 cm}
\newline
    
    \begin{subtable}[h]{1.0\textwidth}
        \centering
        \begin{tabular}{l c c c c c c}
        \toprule
        Member & \makecell{Net cons.(+) \\ Net prod.(--) \\ (Retail)(1)}& \makecell{\textcolor{cyan}{Net cons.(+)} \\ \textcolor{cyan}{Net prod.(--)} \\ \textcolor{cyan}{(REC)(1)}} & \makecell{Net cons.(+) \\ Net prod.(--) \\ (Retail)(2)}& \makecell{\textcolor{cyan}{Net cons.(+)} \\ \textcolor{cyan}{Net prod.(--)} \\ \textcolor{cyan}{(REC)(2)}} & \makecell{Offtake\\ peak} & \makecell{Injection\\ peak}\\
        \midrule
        Unit & $kWh$ & $kWh$ & $kWh$ & $kWh$ & $kW$ & $kW$\\
        \midrule 
        %REALLOCATION_SOLUTION%
        \bottomrule
        \end{tabular}
        \label{tab:%PREFIX_INPUT_TABLE%rec_%N_MEMBERS%_example_realloc_scheme}
        \caption{}
     \end{subtable}
     \caption{Optimal solution of the reallocation scheme problem for a REC of %N_MEMBERS% members with its composition specified in (a). Bills before and after the reallocation is reported in (b). The details about the reallocation scheme solution%PLURAL% is provided in (c).}
     \label{tab:%PREFIX_INPUT_TABLE%rec_%N_MEMBERS%_example_results}
\end{table}
"""

TABLE_TEMPLATE_WITH_COMMOODITY = r"""
\begin{table}[h]
    \begin{subtable}[h]{1.0\textwidth}
        \centering
        \begin{tabular}{l c c c c}
        \toprule
        Parameter & Value\\
        \midrule
        $\Delta_C$ & $DELTAC$\\
        $\Delta_M$ & $DELTAM$\\
        $\Delta_B$ & $DELTAB$\\
        $\Lambda^+$ & $LAMBDAPLUS$ $Eur/kWh$\\
        $\Lambda^-$ & $LAMBDAMINUS$ $Eur/kWh$\\
        $P^+$ & $INJECTION_PEAK_COST$ $Eur/kW$\\
        $P^-$ & $OFFTAKE_PEAK_COST$ $Eur/kW$\\
        \midrule
        Member & \makecell{Buying\\price} & \makecell{Selling\\price} & \makecell{Net consumption(+)\\ Net production(--) \\ (1)} & \makecell{Net consumption(+)\\ Net production(--) \\ (2)} \\
        \midrule
        Unit & $Eur/kWh$ & $Eur/kWh$ & $kWh$ & $kWh$\\
        \midrule 
        %INPUT_LINES%
        \bottomrule
       \end{tabular}
       \label{tab:%PREFIX_INPUT_TABLE%rec_%N_MEMBERS%_example_inputs}
       \caption{}
    \end{subtable}
    \newline
\vspace*{0.05 cm}
\newline
    
    \begin{subtable}[h]{1.0\textwidth}
        \centering
        \begin{tabular}{l c}
        \toprule
        Bill & Value\\
        \midrule
        Commodity Bill & %COMMODITY_BILL%\\
        Peak Bill & %PEAK_BILL%\\
        \textbf{Total} & \textbf{%TOTAL_BILL%}\\
        \midrule
        \textcolor{blue}{REC Bill} & \textcolor{blue}{Value}\\
        \midrule
        \textcolor{blue}{Commodity Bill} & \textcolor{blue}{%REC_COMM_COMMODITY_BILL%}\\
        \textcolor{blue}{Peak Bill} & \textcolor{blue}{%REC_COMM_PEAK_BILL%}\\
        \textbf{\textcolor{blue}{Global REC Bill}} & \textbf{\textcolor{blue}{%REC_COMM_GLOBAL_BILL%}}\\
        \midrule
        \textcolor{cyan}{REC Bill} & \textcolor{cyan}{Value}\\
        \midrule
        \textcolor{cyan}{Commodity Bill} & \textcolor{cyan}{%REC_COMMODITY_BILL%}\\
        \textcolor{cyan}{Peak Bill} & \textcolor{cyan}{%REC_PEAK_BILL%}\\
        \textbf{\textcolor{cyan}{Global REC Bill}} & \textbf{\textcolor{cyan}{%REC_GLOBAL_BILL%}}\\
        \bottomrule
        \end{tabular}
        \label{tab:%PREFIX_INPUT_TABLE%rec_%N_MEMBERS%_example_bills}
        \caption{}
     \end{subtable}
    \newline
\vspace*{0.05 cm}
\newline
    
    \begin{subtable}[h]{1.0\textwidth}
        \centering
        \begin{tabular}{l c c c c c c}
        \toprule
        Member & \makecell{Net cons.(+) \\ Net prod.(--) \\ (Retail)(1)}& \makecell{\textcolor{black}{Net cons.(+)} \\ \textcolor{black}{Net prod.(--)} \\ \textcolor{black}{(REC)(1)}} & \makecell{Net cons.(+) \\ Net prod.(--) \\ (Retail)(2)}& \makecell{\textcolor{black}{Net cons.(+)} \\ \textcolor{black}{Net prod.(--)} \\ \textcolor{black}{(REC)(2)}} & \makecell{Offtake\\ peak} & \makecell{Injection\\ peak}\\
        \midrule
        Unit & $kWh$ & $kWh$ & $kWh$ & $kWh$ & $kW$ & $kW$\\
        \midrule
        %COMM_REALLOCATION_SOLUTION%
        \midrule
        %REALLOCATION_SOLUTION%
        \midrule
        \bottomrule
        \end{tabular}
        \label{tab:%PREFIX_INPUT_TABLE%rec_%N_MEMBERS%_example_realloc_scheme}
        \caption{}
     \end{subtable}
     \caption{Optimal solution of the reallocation scheme problem for a REC of %N_MEMBERS% members with its composition specified in (a). Bills before and after the reallocation is reported in (b). The details about the reallocation scheme solution%PLURAL% is provided in (c). }
     \label{tab:%PREFIX_INPUT_TABLE%rec_%N_MEMBERS%_example_results}
\end{table}
"""


TABLE_WITH_THREE_MEMBERS=r"""
\begin{tabular}{lcr lcr lcr}
\toprule
    \multicolumn{9}{c}{Market period 1}\\
    \cmidrule(lr){0-8}
\multicolumn{3}{c}{\shortstack{Net consumption \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{Retail \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{REC \\ $kWh$}} \\
    \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9}
        $M_1$  & $M_2$ & $M_3$ & $M_1$  & $M_2$ & $M_3$ & $M_1$  & $M_2$ & $M_3$     \\ 
        \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9}
        $\mathbf{[m1r1nc]}$ & $\mathbf{[m2r1nc]}$ & $\mathbf{[m3r1nc]}$ & $\mathbf{[m1r1rt]}$ & $\mathbf{[m2r1rt]}$ & $\mathbf{[m3r1rc]}$ & $\mathbf{[m1r1rc]}$ & $\mathbf{[m2r1rc]}$ & $\mathbf{[m3r1rc]}$ \\
        \midrule
        \multicolumn{9}{c}{Market period 2}\\
        \cmidrule(lr){0-8}
        \multicolumn{3}{c}{\shortstack{Net consumption \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{Retail \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{REC \\ $kWh$}} \\
       \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9}
        $M_1$  & $M_2$ & $M_3$ & $M_1$  & $M_2$ & $M_3$ & $M_1$  & $M_2$ & $M_3$\\
        \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9}
        $\mathbf{[m1r2nc]}$ & $\mathbf{[m2r2nc]}$ & $\mathbf{[m3r2nc]}$ & $\mathbf{[m1r2rt]}$ & $\mathbf{[m2r2rt]}$ & $\mathbf{[m3r2rc]}$ & $\mathbf{[m1r2rc]}$ & $\mathbf{[m2r2rc]}$ & $\mathbf{[m3r2rc]}$ \\
        \midrule
        \multicolumn{9}{c}{Billing period}\\
        \cmidrule(lr){0-8}
        \multicolumn{3}{c}{\shortstack{Offtake peak  \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{Injection peak \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{Global REC Bill \\ $Eur$}} \\
        \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9}  
        $M_1$  & $M_2$ & $M_3$  & $M_1$  & $M_2$ & $M_3$  & NO-REC  & \multicolumn{2}{r}{REC}      \\ 
        \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9} 
        $\mathbf{[m1op]}$  & $\mathbf{[m2op]}$   & $\mathbf{[m3op]}$   & $\mathbf{[m1ip]}$ & $\mathbf{[m2ip]}$ & $\mathbf{[m3ip]}$  & \color{red}{$\mathbf{[sumbill]}$}  & \multicolumn{2}{r}{\color{blue}{$\mathbf{[recbill]}$}}      \\ 
    \bottomrule
\end{tabular}
"""

TABLE_WITH_THREE_MEMBERS_THREE_MARKET_PERIODS=r"""
\begin{tabular}{lcr lcr lcr}
\toprule
    \multicolumn{9}{c}{Market period 1}\\
    \cmidrule(lr){0-8}
\multicolumn{3}{c}{\shortstack{Net consumption \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{Retail \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{REC \\ $kWh$}} \\
    \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9}
        $M_1$  & $M_2$ & $M_3$ & $M_1$  & $M_2$ & $M_3$ & $M_1$  & $M_2$ & $M_3$     \\ 
        \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9}
        $\mathbf{[m1r1nc]}$ & $\mathbf{[m2r1nc]}$ & $\mathbf{[m3r1nc]}$ & $\mathbf{[m1r1rt]}$ & $\mathbf{[m2r1rt]}$ & $\mathbf{[m3r1rc]}$ & $\mathbf{[m1r1rc]}$ & $\mathbf{[m2r1rc]}$ & $\mathbf{[m3r1rc]}$ \\
        \midrule
        \multicolumn{9}{c}{Market period 2}\\
        \cmidrule(lr){0-8}
        \multicolumn{3}{c}{\shortstack{Net consumption \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{Retail \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{REC \\ $kWh$}} \\
       \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9}
        $M_1$  & $M_2$ & $M_3$ & $M_1$  & $M_2$ & $M_3$ & $M_1$  & $M_2$ & $M_3$\\
        \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9}
        $\mathbf{[m1r2nc]}$ & $\mathbf{[m2r2nc]}$ & $\mathbf{[m3r2nc]}$ & $\mathbf{[m1r2rt]}$ & $\mathbf{[m2r2rt]}$ & $\mathbf{[m3r2rc]}$ & $\mathbf{[m1r2rc]}$ & $\mathbf{[m2r2rc]}$ & $\mathbf{[m3r2rc]}$ \\
        \midrule
        \multicolumn{9}{c}{Market period 3}\\
        \cmidrule(lr){0-8}
        \multicolumn{3}{c}{\shortstack{Net consumption \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{Retail \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{REC \\ $kWh$}} \\
       \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9}
        $M_1$  & $M_2$ & $M_3$ & $M_1$  & $M_2$ & $M_3$ & $M_1$  & $M_2$ & $M_3$\\
        \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9}
        $\mathbf{[m1r3nc]}$ & $\mathbf{[m2r3nc]}$ & $\mathbf{[m3r3nc]}$ & $\mathbf{[m1r3rt]}$ & $\mathbf{[m2r3rt]}$ & $\mathbf{[m3r3rc]}$ & $\mathbf{[m1r3rc]}$ & $\mathbf{[m2r3rc]}$ & $\mathbf{[m3r3rc]}$ \\
        \midrule
        \multicolumn{9}{c}{Billing period}\\
        \cmidrule(lr){0-8}
        \multicolumn{3}{c}{\shortstack{Offtake peak  \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{Injection peak \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{Global REC Bill \\ $Eur$}} \\
        \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9}  
        $M_1$  & $M_2$ & $M_3$  & $M_1$  & $M_2$ & $M_3$  & NO-REC  & \multicolumn{2}{r}{REC}      \\ 
        \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9} 
        $\mathbf{[m1op]}$  & $\mathbf{[m2op]}$   & $\mathbf{[m3op]}$   & $\mathbf{[m1ip]}$ & $\mathbf{[m2ip]}$ & $\mathbf{[m3ip]}$  & \color{red}{$\mathbf{[sumbill]}$}  & \multicolumn{2}{r}{\color{blue}{$\mathbf{[recbill]}$}}      \\ 
    \bottomrule
\end{tabular}
"""

TABLE_WITH_THREE_MEMBERS_FOUR_MARKET_PERIODS=r"""
\begin{tabular}{lcr lcr lcr}
\toprule
    \multicolumn{9}{c}{Market period 1}\\
    \cmidrule(lr){0-8}
\multicolumn{3}{c}{\shortstack{Net consumption \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{Retail \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{REC \\ $kWh$}} \\
    \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9}
        $M_1$  & $M_2$ & $M_3$ & $M_1$  & $M_2$ & $M_3$ & $M_1$  & $M_2$ & $M_3$     \\ 
        \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9}
        $\mathbf{[m1r1nc]}$ & $\mathbf{[m2r1nc]}$ & $\mathbf{[m3r1nc]}$ & $\mathbf{[m1r1rt]}$ & $\mathbf{[m2r1rt]}$ & $\mathbf{[m3r1rc]}$ & $\mathbf{[m1r1rc]}$ & $\mathbf{[m2r1rc]}$ & $\mathbf{[m3r1rc]}$ \\
        \midrule
        \multicolumn{9}{c}{Market period 2}\\
        \cmidrule(lr){0-8}
        \multicolumn{3}{c}{\shortstack{Net consumption \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{Retail \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{REC \\ $kWh$}} \\
       \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9}
        $M_1$  & $M_2$ & $M_3$ & $M_1$  & $M_2$ & $M_3$ & $M_1$  & $M_2$ & $M_3$\\
        \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9}
        $\mathbf{[m1r2nc]}$ & $\mathbf{[m2r2nc]}$ & $\mathbf{[m3r2nc]}$ & $\mathbf{[m1r2rt]}$ & $\mathbf{[m2r2rt]}$ & $\mathbf{[m3r2rc]}$ & $\mathbf{[m1r2rc]}$ & $\mathbf{[m2r2rc]}$ & $\mathbf{[m3r2rc]}$ \\
        \midrule
        \multicolumn{9}{c}{Market period 3}\\
        \cmidrule(lr){0-8}
        \multicolumn{3}{c}{\shortstack{Net consumption \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{Retail \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{REC \\ $kWh$}} \\
       \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9}
        $M_1$  & $M_2$ & $M_3$ & $M_1$  & $M_2$ & $M_3$ & $M_1$  & $M_2$ & $M_3$\\
        \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9}
        $\mathbf{[m1r3nc]}$ & $\mathbf{[m2r3nc]}$ & $\mathbf{[m3r3nc]}$ & $\mathbf{[m1r3rt]}$ & $\mathbf{[m2r3rt]}$ & $\mathbf{[m3r3rc]}$ & $\mathbf{[m1r3rc]}$ & $\mathbf{[m2r3rc]}$ & $\mathbf{[m3r3rc]}$ \\
        \midrule
        \multicolumn{9}{c}{Market period 4}\\
        \cmidrule(lr){0-8}
        \multicolumn{3}{c}{\shortstack{Net consumption \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{Retail \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{REC \\ $kWh$}} \\
       \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9}
        $M_1$  & $M_2$ & $M_3$ & $M_1$  & $M_2$ & $M_3$ & $M_1$  & $M_2$ & $M_3$\\
        \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9}
        $\mathbf{[m1r4nc]}$ & $\mathbf{[m2r4nc]}$ & $\mathbf{[m3r4nc]}$ & $\mathbf{[m1r4rt]}$ & $\mathbf{[m2r4rt]}$ & $\mathbf{[m3r4rc]}$ & $\mathbf{[m1r4rc]}$ & $\mathbf{[m2r4rc]}$ & $\mathbf{[m3r4rc]}$ \\
        \midrule
        \multicolumn{9}{c}{Billing period}\\
        \cmidrule(lr){0-8}
        \multicolumn{3}{c}{\shortstack{Offtake peak  \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{Injection peak \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{Global REC Bill \\ $Eur$}} \\
        \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9}  
        $M_1$  & $M_2$ & $M_3$  & $M_1$  & $M_2$ & $M_3$  & NO-REC  & \multicolumn{2}{r}{REC}      \\ 
        \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9} 
        $\mathbf{[m1op]}$  & $\mathbf{[m2op]}$   & $\mathbf{[m3op]}$   & $\mathbf{[m1ip]}$ & $\mathbf{[m2ip]}$ & $\mathbf{[m3ip]}$  & \color{red}{$\mathbf{[sumbill]}$}  & \multicolumn{2}{r}{\color{blue}{$\mathbf{[recbill]}$}}      \\ 
    \bottomrule
\end{tabular}
"""


TABLE_WITH_THREE_MEMBERS_MANY_MARKET_PERIODS=r"""
\begin{tabular}{ccc lcr lcr lcr}
\toprule
    
    %MARKET_PERIODS
        \midrule
        \multicolumn{3}{c}{} & \multicolumn{3}{c}{\shortstack{Offtake peak  \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{Injection peak \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{Global REC Bill \\ $Eur$}} \\
        \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12} 
        \multicolumn{3}{c}{Billing period} & $M_1$  & $M_2$ & $M_3$  & $M_1$  & $M_2$ & $M_3$  & NO-REC  & \multicolumn{2}{r}{REC}      \\ 
        \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12}
        \multicolumn{3}{c}{1} & $\mathbf{[m1op]}$  & $\mathbf{[m2op]}$   & $\mathbf{[m3op]}$   & $\mathbf{[m1ip]}$ & $\mathbf{[m2ip]}$ & $\mathbf{[m3ip]}$  & \color{red}{$\mathbf{[sumbill]}$}  & \multicolumn{2}{r}{\color{blue}{$\mathbf{[recbill]}$}}      \\ 
    \bottomrule
\end{tabular}
"""

MARKET_PERIOD_TEMPLATE_THREE_MEMBERS_HEADER=r"""
\multicolumn{3}{c}{} & \multicolumn{3}{c}{\shortstack{Net consumption \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{Retail \\ $kWh$}} & \multicolumn{3}{c}{\shortstack{REC \\ $kWh$}} \\
    \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12}
        \multicolumn{3}{c}{Market period} & $M_1$  & $M_2$ & $M_3$ & $M_1$  & $M_2$ & $M_3$ & $M_1$  & $M_2$ & $M_3$     \\ 
"""

MARKET_PERIOD_TEMPLATE_THREE_MEMBERS=r"""
        \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-9} \cmidrule(lr){10-12}
        \multicolumn{3}{c}{[R]} & $\mathbf{[m1r[R]nc]}$ & $\mathbf{[m2r[R]nc]}$ & $\mathbf{[m3r[R]nc]}$ & $\mathbf{[m1r[R]rt]}$ & $\mathbf{[m2r[R]rt]}$ & $\mathbf{[m3r[R]rt]}$ & $\mathbf{[m1r[R]rc]}$ & $\mathbf{[m2r[R]rc]}$ & $\mathbf{[m3r[R]rc]}$ \\
"""

TABLE_WITH_TWO_MEMBERS=r"""
\begin{tabular}{cc cc cc}
    \toprule
    \multicolumn{6}{c}{Market period 1}\\
    \cmidrule(lr){0-5}
\multicolumn{2}{c}{\shortstack{Net consumption \\ $kWh$}} & \multicolumn{2}{c}{\shortstack{Retail \\ $kWh$}} & \multicolumn{2}{c}{\shortstack{REC \\ $kWh$}} \\
    \cmidrule(lr){0-1} \cmidrule(lr){3-4} \cmidrule(lr){5-6}
        $M_1$  & $M_2$ & $M_1$  & $M_2$ & $M_1$  & $M_2$      \\ 
        \cmidrule(lr){0-1} \cmidrule(lr){3-4} \cmidrule(lr){5-6}
        $\mathbf{[m1r1nc]}$ & $\mathbf{[m2r1nc]}$ & $\mathbf{[m1r1rt]}$ & $\mathbf{[m2r1rt]}$ & $\mathbf{[m1r1rc]}$ & $\mathbf{[m2r1rc]}$ \\
        \midrule
        \multicolumn{6}{c}{Market period 2}\\
        \cmidrule(lr){0-5}
        \multicolumn{2}{c}{\shortstack{Net consumption \\ $kWh$}} & \multicolumn{2}{c}{\shortstack{Retail \\ $kWh$}} & \multicolumn{2}{c}{\shortstack{REC \\ $kWh$}} \\
        \cmidrule(lr){0-1} \cmidrule(lr){3-4} \cmidrule(lr){4-6}
        $M_1$  & $M_2$  & $M_1$  & $M_2$ & $M_1$  & $M_2$\\
        \cmidrule(lr){0-1} \cmidrule(lr){3-4} \cmidrule(lr){5-6}
        $\mathbf{[m1r2nc]}$ & $\mathbf{[m2r2nc]}$ & $\mathbf{[m1r2rt]}$ & $\mathbf{[m2r2rt]}$ & $\mathbf{[m1r2rc]}$ & $\mathbf{[m2r2rc]}$ \\
        \midrule
        \multicolumn{6}{c}{Billing period}\\
        \cmidrule(lr){0-5}
        \multicolumn{2}{c}{\shortstack{Offtake peak  \\ $kWh$}} & \multicolumn{2}{c}{\shortstack{Injection peak \\ $kWh$}} & \multicolumn{2}{c}{\shortstack{Global REC Bill \\ $Eur$}} \\
        \cmidrule(lr){0-1} \cmidrule(lr){3-4} \cmidrule(lr){5-6}  
        $M_1$  & $M_2$  & $M_1$  & $M_2$  & NO-REC  & REC      \\ 
        \cmidrule(lr){0-1} \cmidrule(lr){3-4} \cmidrule(lr){5-6} 
         $\mathbf{[m1op]}$  & $\mathbf{[m2op]}$   & $\mathbf{[m1ip]}$ & $\mathbf{[m2ip]}$  & \color{red}{$\mathbf{[sumbill]}$}  & \color{blue}{$\mathbf{[recbill]}$}    \\ 
    \bottomrule
\end{tabular}
"""

TABLE_CONFIG_TWO_MEMBERS=r"""
\begin{tabular}{cc cc cccc}
    \toprule
\multicolumn{2}{c}{\shortstack{Buying price \\ Eur/$kWh$}} & \multicolumn{2}{c}{\shortstack{Selling price \\ Eur/$kWh$}} &  \multicolumn{4}{c}{\shortstack{Network costs \\ $Eur/kWh$}}   \\
    \cmidrule(lr){0-1} \cmidrule(lr){3-4} \cmidrule(lr){5-8} 
        $M_1$  & $M_2$  & $M_1$  & $M_2$ & $P^-$ & $P^+$ & $\Lambda^-$ & $\Lambda^+$     \\ 
    \cmidrule(lr){0-1} \cmidrule(lr){3-4} \cmidrule(lr){5-8} 
        $\mathbf{[m1bp]}$ & $\mathbf{[m2bp]}$    & $\mathbf{[m1sp]}$ & $\mathbf{[m2sp]}$ & $\mathbf{[opc]}$ & $\mathbf{[ipc]}$ & $\mathbf{[nfrs]}$ & $\mathbf{[nfrb]}$  \\
    \bottomrule
\end{tabular}
"""

TABLE_CONFIG_THREE_MEMBERS=r"""
\begin{tabular}{ccc ccc cccc}
    \toprule
\multicolumn{3}{c}{\shortstack{Buying price \\ Eur/$kWh$}} & \multicolumn{3}{c}{\shortstack{Selling price \\ Eur/$kWh$}} &  \multicolumn{4}{c}{\shortstack{Network costs \\ $Eur/kWh$}}   \\
    \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-10} 
        $M_1$  & $M_2$ & $M_3$  & $M_1$  & $M_2$ & $M_3$ & $P^-$ & $P^+$ & $\Lambda^-$ & $\Lambda^+$     \\ 
    \cmidrule(lr){0-2} \cmidrule(lr){4-6} \cmidrule(lr){7-10} 
        $\mathbf{[m1bp]}$ & $\mathbf{[m2bp]}$ & $\mathbf{[m3bp]}$  & $\mathbf{[m1sp]}$ & $\mathbf{[m2sp]}$ & $\mathbf{[m3sp]}$ & $\mathbf{[opc]}$ & $\mathbf{[ipc]}$ & $\mathbf{[nfrs]}$ & $\mathbf{[nfrb]}$  \\
    \bottomrule
\end{tabular}
"""


def build_template_table(n_members: int, n_market_periods: int):
    if n_members == 2:
        return TABLE_WITH_TWO_MEMBERS
    elif n_members == 3:
        if n_market_periods == 2:
            return TABLE_WITH_THREE_MEMBERS
        elif n_market_periods == 3:
            return TABLE_WITH_THREE_MEMBERS_THREE_MARKET_PERIODS
        else:
            market_period_str = MARKET_PERIOD_TEMPLATE_THREE_MEMBERS_HEADER
            for r in range(1, n_market_periods+1):
                market_period_str += MARKET_PERIOD_TEMPLATE_THREE_MEMBERS.replace("[R]", str(r)) + "\n"
            return TABLE_WITH_THREE_MEMBERS_MANY_MARKET_PERIODS.replace("%MARKET_PERIODS", market_period_str)

def number_as_meter(r:float):
    r_rounded = np.round(r, 2)
    r_separated = str(r_rounded).split(".")
    r_integer = r_separated[0]
    nb_zeros_to_pad = 5 - len(r_integer)
    if nb_zeros_to_pad > 0:
        r_integer = "0"*nb_zeros_to_pad + r_integer
    r_decimal = r_separated[1][:2]
    nb_zeros_to_pad = 2 - len(r_decimal)
    if nb_zeros_to_pad > 0:
        r_decimal = "0"*nb_zeros_to_pad + r_decimal
    return list(r_integer), list(r_decimal)

def replace_in_template(template:str, pattern:str, r_integer:list[str], r_decimal:list[str]):
    decimal_place_7 = pattern.replace("#", "7")
    decimal_place_6 = pattern.replace("#", "6")
    integer_place_5 = pattern.replace("#", "5")
    integer_place_4 = pattern.replace("#", "4")
    integer_place_3 = pattern.replace("#", "3")
    integer_place_2 = pattern.replace("#", "2")
    integer_place_1 = pattern.replace("#", "1")
    template_res = template.replace(decimal_place_7, r_decimal[-1])
    template_res = template_res.replace(decimal_place_6, r_decimal[-2])
    template_res = template_res.replace(integer_place_5, r_integer[-1])
    template_res = template_res.replace(integer_place_4, r_integer[-2])
    template_res = template_res.replace(integer_place_3, r_integer[-3])
    template_res = template_res.replace(integer_place_2, r_integer[-4])
    template_res = template_res.replace(integer_place_1, r_integer[-5])
    return template_res

    


@click.command()
@click.option('--n-members', "n_members", type=int, help='Number of members', default=2)
@click.option('--seed', type=int, default=None, help="random seed for generating meters and prices")
@click.option('--billing-period', "billing_period", type=int, default=2, help="billing period size")
@click.option('--force-peak-costs', "force_peak_costs", is_flag=True, help="whether to force peak costs when peak coeffs are disabled for optim")
@click.option('--disable-peak-coeffs', "disable_peak_coeffs", is_flag=True, help="Disable peak coefficients")
@click.option('--output-time', "output_time", is_flag=True, help="Only output time computation")
@click.option('--peak-cost', "peak_cost", type=int, help='Peak cost', default=1.0)
@click.option('--optimised-rec-fees', "optimised_rec_fees", is_flag=True, help="Optimised rec fees calculation (monoprice case)")
@click.option('--type-solve', "type_solve", type=click.Choice(["cvxpy", "mosek"]), help="Solver", default="mosek")
def run_examples(n_members, seed, billing_period, force_peak_costs, disable_peak_coeffs, output_time, peak_cost, optimised_rec_fees, type_solve):
    if seed is None:
        seed = np.random.randint(1, 1000000)
        print(seed)
    if not output_time and n_members not in (2, 3):
        print("Only 2 or 3 members are supported")
        exit()
    random_gen = np.random.RandomState(seed=seed)
    offtake_peak_cost = peak_cost * (not disable_peak_coeffs) + 1e-12 * (disable_peak_coeffs)
    injection_peak_cost = peak_cost * (not disable_peak_coeffs) + 1e-12 * (disable_peak_coeffs)
    Delta_M = 1
    Delta_P = billing_period
    Delta_C = 1
    members = list([i for i in range(n_members)])
    buying_prices = {
        (member, "buying_price"):np.round(np.asarray([0.2+i*0.02]*billing_period), 2) for i, member in enumerate(members) 
    }
    selling_prices = {
        (member, "selling_price"): np.round(np.asarray([0.04+i*0.01]*billing_period), 2) for i, member in enumerate(members) 
    }
    
    if optimised_rec_fees:
        rec_import_fees = sorted(list([b[0] for b in buying_prices.values()]))
        rec_import_fees = round(rec_import_fees[-1] - rec_import_fees[0] + 0.01, 2)
        rec_export_fees = sorted(list([s[0] for s in selling_prices.values()]))
        rec_export_fees = round(rec_export_fees[-1] - rec_export_fees[0] + 0.01, 2)
    else:
        rec_import_fees = round(max(
            abs(p[0] - p[1]) for p in product(*list(buying_prices.values()))
        ) + 0.01, 2)
        rec_export_fees = round(max(
            abs(p[0] - p[1]) for p in product(*list(selling_prices.values()))
        ) + 0.01, 2)
    optimal_reallocation_schemer = GlobalBillAdaptativeOptimiser(
        members,
        current_offtake_peak_cost=offtake_peak_cost,
        current_injection_peak_cost=injection_peak_cost,
        Delta_M=Delta_M,
        Delta_C=Delta_C,
        Delta_P=Delta_P,
        rec_import_fees=rec_import_fees,
        rec_export_fees=rec_export_fees,
        type_solve=type_solve,
        dpp_compile=False

    )
    
    optimal_reallocation_schemer.reset()
    condition=False
    t_max = float("-inf")

    if output_time:
        consumption_meters = {
            (member, "consumption_meters"):np.round(random_gen.uniform(low=0, high=10000, size=billing_period), billing_period) for member in members
        }
        production_meters = {
            (member, "production_meters"):np.round(random_gen.uniform(low=0, high=10000, size=billing_period), billing_period) for member in members
        }
        state = {
            **consumption_meters,
            **production_meters
        }
        state["metering_period_counter"] = 1
        state["peak_period_counter"] = billing_period
        
        exogenous_prices = {
            **buying_prices,
            **selling_prices
        }
        t = time()
        print("Solving...")
        metering_period_expr, peak_period_expr, offtake_peaks, injection_peaks = optimal_reallocation_schemer.optimise_global_bill(state, exogenous_prices, detailed_solution=False)
        print(time() - t)
    else:
        while not condition:
            consumption_meters = {
                (member, "consumption_meters"):np.round(random_gen.uniform(low=0, high=1000, size=billing_period), billing_period) for member in members
            }
            production_meters = {
                (member, "production_meters"):np.round(random_gen.uniform(low=0, high=1000, size=billing_period), billing_period) for member in members
            }
            state = {
                **consumption_meters,
                **production_meters
            }
            state["metering_period_counter"] = 1
            state["peak_period_counter"] = billing_period
            
            exogenous_prices = {
                **buying_prices,
                **selling_prices
            }
            t = time()
            metering_period_expr, peak_period_expr, offtake_peaks, injection_peaks, rec_imports, rec_exports, grid_imports, grid_exports = optimal_reallocation_schemer.optimise_global_bill(state, exogenous_prices, detailed_solution=True)
            t = time() - t
            t_max = max(t_max, t)
            if type(grid_imports) == dict:
                rec_imports_matrix = np.asarray(list(rec_imports.values()))
                rec_exports_matrix = np.asarray(list(rec_exports.values()))
                rec_exchange_matrix = rec_imports_matrix + rec_exports_matrix
            else:
                rec_exchange_matrix = rec_imports + rec_exports
            agg_rec_exchange_matrix = np.sum(rec_exchange_matrix, axis=0)

            if type(grid_imports) != dict:
                grid_imports = {
                    member:grid_imports[i] for i, member in enumerate(members)
                }
                grid_exports = {
                    member:grid_exports[i] for i, member in enumerate(members)
                }
                rec_imports = {
                    member:rec_imports[i] for i, member in enumerate(members)
                }
                rec_exports = {
                    member:rec_exports[i] for i, member in enumerate(members)
                }
                if offtake_peaks is not None:
                    offtake_peaks = {
                        member:offtake_peaks[i] for i, member in enumerate(members)
                    }
                if injection_peaks is not None:
                    injection_peaks = {
                        member:injection_peaks[i] for i, member in enumerate(members)
                    }
            if force_peak_costs:
                offtake_peaks = {
                    k:epsilonify(max(v), 10e-8) for k,v in grid_imports.items()
                }
                injection_peaks = {
                    k:epsilonify(max(v), 10e-8) for k,v in grid_exports.items()
                }
            peak_period_expr = 1.0 * sum(
                list(offtake_peaks.values()) + list(injection_peaks.values())
            )
            
            
            condition = np.all(agg_rec_exchange_matrix > 0)
            if not disable_peak_coeffs or force_peak_costs:
                if n_members == 2:
                    condition = condition and (
                        (offtake_peaks[0] > 0 and injection_peaks[1] > 0)
                        or
                        (offtake_peaks[1] > 0 and injection_peaks[0] > 0)
                    )
                else:
                    condition = condition and (
                        (offtake_peaks[0] > 0 and (injection_peaks[1] > 0 or injection_peaks[2] > 0))
                        or
                        (offtake_peaks[1] > 0 and (injection_peaks[0] > 0 or injection_peaks[2] > 0))
                        or
                        (offtake_peaks[2] > 0 and (injection_peaks[0] > 0 or injection_peaks[1] > 0))
                    )
                if disable_peak_coeffs and force_peak_costs:
                    true_optimal_reallocation_schemer = GlobalBillAdaptativeOptimiser(
                        members,
                        current_offtake_peak_cost=1.0,
                        current_injection_peak_cost=1.0,
                        Delta_M=Delta_M,
                        Delta_C=Delta_C,
                        Delta_P=Delta_P,
                        rec_import_fees=rec_import_fees,
                        rec_export_fees=rec_export_fees

                    )
                    true_optimal_reallocation_schemer.reset()
                    true_metering_period_expr, true_peak_period_expr, true_offtake_peaks, true_injection_peaks, true_rec_imports, true_rec_exports, true_grid_imports, true_grid_exports = true_optimal_reallocation_schemer.optimise_global_bill(state, exogenous_prices, detailed_solution=True)
                    if type(true_grid_imports) == dict:
                        true_rec_imports_matrix = np.asarray(list(true_rec_imports.values()))
                        true_rec_exports_matrix = np.asarray(list(true_rec_exports.values()))
                        true_rec_exchange_matrix = true_rec_imports_matrix + true_rec_exports_matrix
                    else:
                        true_rec_exchange_matrix = true_rec_imports + true_rec_exports
                    
                    if type(true_grid_imports) != dict:
                        true_grid_imports = {
                            member:true_grid_imports[i] for i, member in enumerate(members)
                        }
                        true_grid_exports = {
                            member:true_grid_exports[i] for i, member in enumerate(members)
                        }
                        true_rec_imports = {
                            member:true_rec_imports[i] for i, member in enumerate(members)
                        }
                        true_rec_exports = {
                            member:true_rec_exports[i] for i, member in enumerate(members)
                        }
                        true_offtake_peaks = {
                            member:true_offtake_peaks[i] for i, member in enumerate(members)
                        }
                        true_injection_peaks = {
                            member:true_injection_peaks[i] for i, member in enumerate(members)
                        }
                    true_offtake_peaks = {
                        k:epsilonify(max(v), 10e-8) for k,v in true_grid_imports.items()
                    }
                    true_injection_peaks = {
                        k:epsilonify(max(v), 10e-8) for k,v in true_grid_exports.items()
                    }
                    true_peak_period_expr = 1.0 * sum(
                        list(true_offtake_peaks.values()) + list(true_injection_peaks.values())
                    )
                    true_bill = true_metering_period_expr + true_peak_period_expr
                    nopeak_bill = metering_period_expr + peak_period_expr
                    from pprint import pprint
                    true_agg_rec_exchange_matrix = np.sum(true_rec_exchange_matrix, axis=0)
                    condition2 = np.all(true_agg_rec_exchange_matrix > 0)
                    condition_3 = abs(sum(list(true_offtake_peaks.values())) - sum(list(offtake_peaks.values()))) > 1
                    condition_4 = abs(sum(list(true_injection_peaks.values())) - sum(list(injection_peaks.values()))) > 1
                    
                    condition = condition and condition_4 and condition_3 and (np.all(agg_rec_exchange_matrix > 0) or condition2) and (abs(true_bill/nopeak_bill) <= 0.66)
        
        
        raw_net_initial_consumption_meters = {
            member: consumption_meters[(member, "consumption_meters")] - production_meters[(member, "production_meters")] for member in members
        }
        raw_net_initial_production_meters = {
            member: production_meters[(member, "production_meters")] - consumption_meters[(member, "consumption_meters")] for member in members
        }
        net_initial_consumption_meters = {
            member: np.maximum(consumption_meters[(member, "consumption_meters")] - production_meters[(member, "production_meters")], 0.0) for member in members
        }
        net_initial_production_meters = {
            member: np.maximum(production_meters[(member, "production_meters")] - consumption_meters[(member, "consumption_meters")], 0.0) for member in members
        }
        metering_period_expr_before = sum([
            (np.sum(net_initial_consumption_meters[member]) * buying_prices[(member, "buying_price")][0]
            - np.sum(net_initial_production_meters[member]) * selling_prices[(member, "selling_price")][0]) for member in members
        ])
        if force_peak_costs:
            peak_period_expr_before = sum([
                (np.max(net_initial_consumption_meters[member]) * 1.0
                + np.max(net_initial_production_meters[member]) * 1.0) for member in members
            ])
        else:
            peak_period_expr_before = sum([
                (np.max(net_initial_consumption_meters[member]) * offtake_peak_cost
                + np.max(net_initial_production_meters[member]) * injection_peak_cost) for member in members
            ])

        total_bill = np.round(metering_period_expr_before + peak_period_expr_before, 2)
        metering_period_expr_before = np.round(metering_period_expr_before, 2)
        peak_period_expr_before = np.round(peak_period_expr_before, 2)
                
        total_rec_bill = np.round(metering_period_expr + peak_period_expr, 2)
        metering_period_expr = np.round(metering_period_expr, 2)
        peak_period_expr = np.round(peak_period_expr, 2)

        table_config = (TABLE_CONFIG_TWO_MEMBERS if n_members == 2 else TABLE_CONFIG_THREE_MEMBERS)
        table_config = table_config.replace("[opc]", str(int(np.round(epsilonify(offtake_peak_cost), 0))))
        table_config = table_config.replace("[ipc]", str(int(np.round(epsilonify(injection_peak_cost), 0))))
        table_config = table_config.replace("[nfrs]", str(rec_export_fees))
        table_config = table_config.replace("[nfrb]", str(rec_import_fees))
        #$\mathbf{[opc]}$ & $\mathbf{ipc}$ & $\mathbf{nfrs}$ & $\mathbf{nfrb}$
        for i in range(1,n_members+1):
            member_buying_price = buying_prices[(i-1, "buying_price")][-1]
            member_selling_price = selling_prices[(i-1, "selling_price")][-1]
            table_config = table_config.replace(f"[m{i}bp]", str(float(member_buying_price)))
            table_config = table_config.replace(f"[m{i}sp]", str(float(member_selling_price)))

        table_result = build_template_table(n_members, billing_period)
        table_result = table_result.replace("[sumbill]", str(float(total_bill)))
        table_result = table_result.replace("[recbill]", str(float(total_rec_bill)))
        if disable_peak_coeffs and not force_peak_costs:
            table_result = table_result.replace(r"$\mathbf{[m1ip]}$", "[m1ip]")
            table_result = table_result.replace(r"$\mathbf{[m2ip]}$", "[m2ip]")
            table_result = table_result.replace(r"$\mathbf{[m3ip]}$", "[m3ip]")
            table_result = table_result.replace(r"$\mathbf{[m1op]}$", "[m1op]")
            table_result = table_result.replace(r"$\mathbf{[m2op]}$", "[m2op]")
            table_result = table_result.replace(r"$\mathbf{[m3op]}$", "[m3op]")
        for i in range(1,n_members+1):
            if disable_peak_coeffs and not force_peak_costs:
                offtake_peak = "/"
                injection_peak = "/"
            else:
                offtake_peak = str(float(np.round(offtake_peaks[i-1], 2)))
                injection_peak = str(float(np.round(injection_peaks[i-1], 2)))
            table_result = table_result.replace(f"[m{i}ip]", injection_peak)
            table_result = table_result.replace(f"[m{i}op]", offtake_peak)
            for r in range(1,billing_period+1):
                member_net_consumption = np.round(raw_net_initial_consumption_meters[i-1][r-1], 2)
                net_grid_import = np.round(grid_imports[i-1][r-1] - grid_exports[i-1][r-1], 2)
                net_rec_import = np.round(rec_imports[i-1][r-1] - rec_exports[i-1][r-1], 2)
                table_result = table_result.replace(f"[m{i}r{r}rt]", str(float(net_grid_import)))
                table_result = table_result.replace(f"[m{i}r{r}rc]", str(float(net_rec_import)))
                table_result = table_result.replace(f"[m{i}r{r}nc]", str(float(member_net_consumption)))
        
            
        #print(table_result)
        
        if disable_peak_coeffs and force_peak_costs:
            true_total_rec_bill = np.round(true_metering_period_expr + true_peak_period_expr, 2)
            true_metering_period_expr = np.round(true_metering_period_expr, 2)
            true_peak_period_expr = np.round(true_peak_period_expr, 2)
            table_result_true = build_template_table(n_members, billing_period)
            table_result_true = table_result_true.replace("[sumbill]", str(float(total_bill)))
            table_result_true = table_result_true.replace("[recbill]", str(float(true_total_rec_bill)))
            for i in range(1,n_members+1):
                offtake_peak = str(float(np.round(true_offtake_peaks[i-1], 2)))
                injection_peak = str(float(np.round(true_injection_peaks[i-1], 2)))
                table_result_true = table_result_true.replace(f"[m{i}ip]", injection_peak)
                table_result_true = table_result_true.replace(f"[m{i}op]", offtake_peak)
                for r in range(1,billing_period+1):
                    member_net_consumption = np.round(raw_net_initial_consumption_meters[i-1][r-1], 2)
                    net_grid_import = np.round(true_grid_imports[i-1][r-1] - true_grid_exports[i-1][r-1], 2)
                    net_rec_import = np.round(true_rec_imports[i-1][r-1] - true_rec_exports[i-1][r-1], 2)
                    #print(i, r, net_grid_import)
                    table_result_true = table_result_true.replace(f"[m{i}r{r}rt]", str(float(net_grid_import)))
                    table_result_true = table_result_true.replace(f"[m{i}r{r}rc]", str(float(net_rec_import)))
                    table_result_true = table_result_true.replace(f"[m{i}r{r}nc]", str(float(member_net_consumption)))
        #print(table_config)
        print("TABLE RESULT")
        print(table_result)
        if disable_peak_coeffs and force_peak_costs:
            print("TABLE RESULT TRUE REC BILL")
            print(table_result_true)
        print(offtake_peaks, injection_peaks)
        if disable_peak_coeffs and force_peak_costs:
            print(true_offtake_peaks, true_injection_peaks)
        print(seed)
    

if __name__ == "__main__":
    run_examples()