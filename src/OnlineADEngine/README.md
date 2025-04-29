This framework is used as engine for applying anomaly detection techniques, 
providing rich range-based evaluation and logging using mlflow, taken from here: link

This framework allows for easy usage of semi-supervised and unsupervised anomaly detectors, 
handling the creation of reference data to fit detectors. 

There are three choices for fitting detectors in potentially normal data:
- Online: where an initial fragment of data is used to fit anomaly detectors **(we use that)**
- Sliding: where periodically detectors are fitted in recent historical data.
- Historical: where detectors are fitted in given dataset before predict.

For more information refer to the original repository.


# Running experiments in datasets
To run experiment we have to implement two scripts, one for loading dataset as a dictionary (look in loadDataset.py).
The dictionary should include all meta-data required by the framework (look existing examples and refer to original repository examples).

The second step is to instantiate an experiment as done in scripts for

# This framework include mango framework:
