----------------------------------------------------------------------------------------------------------------------------------
Warm-start Quantum Approximate Optimization Algorithm implementation based on the paper "Warm-starting quantum optimization" by
Daniel J. Egger, Jakub Marecek, Stefan Woerner. https://arxiv.org/abs/2009.10095
----------------------------------------------------------------------------------------------------------------------------------


The repository is divided in three files, the first one, "ContinuousStart" gathers the implementation of the following three simulations carried out in the section "Simulations with Continuous-Valued
Warm-start" from the paper. Each of them has an .sh file to run in QMIO cluster.

ContinuousStart ------
- Portfolio Optmization for different QAOA depths, n = 6, with budget constraint added via penalty. (~4min StatevectorEstimator, ~30min AerSimulator)
        -->  run 1run_cont1.sh, then 1run_cont1_plot.sh to plot results
   
- 250 Portfolio instances for depth-one QAOA, n = 6.                                                (~4min StatevectorEstimator)   
        -->  run 2run_cont2.sh, then 2run_cont2_plot.sh to plot results      

- Simulated Quantum Annealing Portfolio Optimization, n = 6.                                        (~8min StatevectorEstimator)
        -->  run 3run_cont3.sh, then 3run_cont3_plot.sh to plot results

On each of them a comparison between the warm and cold start cases for QAOA is carried out.

The next two folders contain the files for the next section of the paper: "Simulations with Rounded Warm-
Start".

RoundQulacs ------
This folder only covers the first simulation, a sweep of different regularization parameter values. It is implemented in the a64 partition of QMIO with Qulacs to improve computation times.




RoundStart ------





  


The code was run in a miniconda environment, the necessary packages are gathered in another file for reproductibility.

----------------------------------------------
Julio Souto Garnelo
QuantumSpain
Galician Supercomputing Center (CESGA) 
----------------------------------------------
