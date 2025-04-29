import pandas as pd
import mlflow

from experiment import PdMExperiment
from evaluation.evaluation import myeval as pdm_evaluate 


class StreamingSemiSupervisedPdMExperiment(PdMExperiment):
    def execute(self) -> None:
        super()._register_experiment()

        with mlflow.start_run(experiment_id=self.experiment_id) as parent_run:
            for current_row_index, current_row in self.target_data.iterrows():
                print(current_row_index)

            super()._finish_run(parent_run=parent_run)