
## Exathlon Benchmark

Taken from official implementation: https://github.com/exathlonbenchmark/exathlon

We keep only code for ED evaluation of original implementation, please refer to the original repository for details of usage.

Furthermore, for easy of reproducibility we have uploaded the pre-processed data used in our analysis: [here](https://drive.google.com/file/d/1eVSprZa4O1D-aPmvQ2T8LP6gUOK5vtjs/view?usp=sharing)  (need to be placed inside /EXATHLON/preprossedData/)

### Reproduce:


```
cd EXATHLON
python main.py --method "LIME" --gt --limek 5 --coverage center
```
Note that depending on the method this may take a while. 

When choosing to run with context (TEMPC), a database with calculated contexts for all scores is created.
This database is used by default (if exist) for future runs (to not calculate context again.)

The context calculated for Exathlon data in paper can be found: [here](https://drive.google.com/file/d/1rGw41ijoy2JeUQuKxydcAtOMskAV7-UT/view?usp=drive_link) (place it under \EXATHLON folder)

Note: default evaluation uses "center" policy (other choices "all")

### Show results of Exathlon:

Paper's results:
```
python ./show_Exathlon_results.py --filename "paper_results/ExathlonResult.csv" --method 'TEMPC'

Type TEST_BURSTY_INPUT_: 1.39 & 1.00 & 1.80
Type TEST_BURSTY_INPUT_CRASH_: 1.43 & 0.86 & 2.50
Type TEST_STALLED_INPUT_: 1.09 & 1.00 & 1.75
Type TEST_CPU_CONTENTION_: 1.60 & 0.90 & 2.17
Type TEST_DRIVER_FAILURE_: 3.00 & 0.14 & 1.00
Type TEST_EXECUTOR_FAILURE_: 1.22 & 0.38 & 1.00
Type TEST_UNKNOWN_: 1.61 & 0.60 & 1.50
```

After re-running (default path):
```
python ./show_Exathlon_results.py --filename "EXATHLON/myADoutput/model_dependent_1.0_cov_center_explanation_comparison.csv" --method 'TEMPC_gt'

Type TEST_BURSTY_INPUT_: 1.39 & 1.00 & 1.80
Type TEST_BURSTY_INPUT_CRASH_: 1.43 & 0.86 & 2.50
Type TEST_STALLED_INPUT_: 1.09 & 1.00 & 1.75
Type TEST_CPU_CONTENTION_: 1.60 & 0.90 & 2.17
Type TEST_DRIVER_FAILURE_: 3.00 & 0.14 & 1.00
Type TEST_EXECUTOR_FAILURE_: 1.22 & 0.38 & 1.00
Type TEST_UNKNOWN_: 1.61 & 0.60 & 1.50

```
