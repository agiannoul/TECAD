import numpy as np
from .utils.metrics import metricor
from .analysis.robustness_eval import generate_curve
        
def get_metrics(score, labels, best_threshold_examined, metric='all', version='opt_mem', slidingWindow=None, thre=250):
	metrics = {}
	if metric == 'vus':
		grader = metricor()
		_, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels, score, slidingWindow, version, thre)

		metrics['VUS_ROC'] = VUS_ROC
		metrics['VUS_PR'] = VUS_PR

		return metrics

	elif metric == 'range_auc':
		grader = metricor()
		R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
		
		metrics['R_AUC_ROC'] = R_AUC_ROC
		metrics['R_AUC_PR'] = R_AUC_PR

		return metrics

	elif metric == 'auc':
		
		grader = metricor()
		AUC_ROC = grader.metric_new_auc(labels, score, plot_ROC=False)
		_, _, AUC_PR = grader.metric_PR(labels, score)

		metrics['AUC_ROC'] = AUC_ROC
		metrics['AUC_PR'] = AUC_PR

		return metrics
	
	else :
		from .basic_metrics import basic_metricor
        
		grader = metricor()
		_, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels, score, slidingWindow, version, thre)
		R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
		grader = basic_metricor()
		AUC_ROC, Precision, Recall, F, Rrecall, ExistenceReward, OverlapReward, Rprecision, RF, Precision_at_k = grader.metric_new(labels, score, best_threshold_examined=best_threshold_examined, plot_ROC=False)
		_, _, AUC_PR = grader.metric_PR(labels, score)

		from .affiliation.generics import convert_vector_to_events
		from .affiliation.metrics import pr_from_events

		discrete_score = np.array(score > 0.5, dtype=np.float32)
		events_pred = convert_vector_to_events(discrete_score)
		events_gt = convert_vector_to_events(labels)
		Trange = (0, len(discrete_score))
		affiliation_metrics = pr_from_events(events_pred, events_gt, Trange)
		metrics['AUC_ROC'] = AUC_ROC
		metrics['AUC_PR'] = AUC_PR
		metrics['Precision'] = Precision
		metrics['Recall'] = Recall
		metrics['F'] = F
		metrics['Precision_at_k'] = Precision_at_k
		metrics['Rprecision'] = Rprecision
		metrics['Rrecall'] = Rrecall
		metrics['RF'] = RF
		metrics['R_AUC_ROC'] = R_AUC_ROC
		metrics['R_AUC_PR'] = R_AUC_PR
		metrics['VUS_ROC'] = VUS_ROC
		metrics['VUS_PR'] = VUS_PR
		metrics['Affiliation_Precision'] = affiliation_metrics['Affiliation_Precision']
		metrics['Affiliation_Recall'] = affiliation_metrics['Affiliation_Recall']
		
		if metric == 'all':
			return metrics
		else:
			return metrics[metric]