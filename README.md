----------------------------------------------------------------------------------------------------------------------------------
Warm-start Quantum Approximate Optimization Algorithm implementation based on the paper "Warm-starting quantum optimization" by
Daniel J. Egger, Jakub Marecek, Stefan Woerner. https://arxiv.org/abs/2009.10095
----------------------------------------------------------------------------------------------------------------------------------


The repository is divided in three files, the first one, "ContinuousStart" gathers the implementation of the following three simulations carried out in the section "Simulations with Continuous-Valued
Warm-start" from the paper. Each of them has an .sh file to run in QMIO cluster.

_______________________________________________________________________________________________________________________________________________________________________________________
ContinuousStart ------
- Portfolio Optmization for different QAOA depths, n = 6, with budget constraint added via penalty.  (~4min StatevectorEstimator, ~30min AerSimulator)
        -->  run 1run_cont1.sh, then 1run_cont1_plot.sh to plot results
   
- 250 Portfolio instances for depth-one QAOA, n =  6.                                                (~4min StatevectorEstimator)   
        -->  run 2run_cont2.sh, then 2run_cont2_plot.sh to plot results      

- Simulated Quantum Annealing Portfolio Optimization, n = 6.                                         (~8min StatevectorEstimator)
        -->  run 3run_cont3.sh, then 3run_cont3_plot.sh to plot results

On each of them a comparison between the warm and cold start cases for QAOA is carried out.
_______________________________________________________________________________________________________________________________________________________________________________________


The next two folders contain the files for the next section of the paper: "Simulations with Rounded Warm-Start", each implemented with a different package.

_______________________________________________________________________________________________________________________________________________________________________________________
RoundStart ------
This folder contains the three simulation carried out in the second section of the paper. In general much slower than Qulacs unless multi-threading or GPU acceleration is included.

- Regularization parameter sweep for 10 fully-connected graphs. Initial parameter search, five best cuts.
        -->  run 1run_round1_cuts.sh, then run 1run_round1_master.sh and finally run 1run_cont1_plot.sh to plot results

- Recursive warm-start QAOA, recursive QAOA, recursive GW, GW. 
        -->  run 2run_round2_master.sh substituting the type of recursive procedure to use, then run () to plot results                (CS ~70min Aer 5threads)

- n=6 fully connected graph, depth sweep.
        -->  run 3run_round3, then run 3run_round3_master.sh to plot results
_______________________________________________________________________________________________________________________________________________________________________________
_______________________________________________________________________________________________________________________________________________________________________________
RoundQulacs ------
This folder covers the first simulation, a sweep of different regularization parameter values. It is implemented in the a64 partition of QMIO with Qulacs to improve computation times. Additionally, recursive method has been added as a Python class. 

- Regularization parameter sweep for 10 fully-connected graphs. Initial parameter search, five best cuts.                               (run on a64, specific Qulacs module for hpc)
        -->  run 1run_round1_cuts.sh, then run 1run_round1_master.sh and finally run 1run_cont1_plot.sh to plot results

- Recursive warm-start QAOA, recursive QAOA, recursive GW, GW. 
        -->  run run_recursive.sh indicating the type of recursive procedure to use, (then run 2run_round2_plot.sh to plot results)     (run with Qulacs on Qmio)

        (20qubits WS-RQAOA ~3min, )
_______________________________________________________________________________________________________________________________________________________________________________________



!! The code was run in a miniconda environment, the necessary packages are gathered in another file for reproductibility.
!! The .sh files were made for the CESGA HPC architecture, ran on the QMIO cluster. 

----------------------------------------------
Julio Souto Garnelo
QuantumSpain
Galician Supercomputing Center (CESGA) 
----------------------------------------------
