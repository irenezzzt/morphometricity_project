# Manual for Morphometricity Project Code
    1. Content list:
        a. Core functions: morphometricity.py (matlab ver: morphometricity.m)
        b. Functions for running simulation: simulation_function.py
        c. Test cases for validating the functions: test_cases.py
        d. Simulation series: simulation.py
        e. UK Biobank data analysis: data_processing.py + ukb_analysis.py
    2. Process to run:
        a. To obtain figure 1 and 2 (estimated morphometricity under correctly and wrongly specified model, AIC chosen kernel): simulation.py
        b. To obtain figure 3 (estimated morphometricity for chosen traits in UKB): 
            i. Run data_processing.py and save the data for 2nd visit: ses2.csv
            ii. Run ukb_analysis.py
