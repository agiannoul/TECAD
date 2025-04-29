For each method XAI method (SHAP,LIME and Context) we gather the importance features from each sample and stored them in pickles files.

evaluate_XAI_results_f1_MRR.py: code for calculating f1, and MRR score from paper along with an example of explanations for particular type of anomalies.

### Running BATADAL XAI evaluation
```
cd EvalXAI/XAI_BATALAD/
```
Generate importance features
```
python .\runXAI.py  --method 'LIME' --filename "limetest.pickle" --limek 5
python .\runXAI.py  --method 'SHAP' --filename "shap.pickle" --shapthreshold 0.1
python .\runXAI.py  --method 'TEMPC' --filename "context.pickle"         
```

Evaluate importance features

```
 python .\ED_evaluation.py --filename "limetest.pickle"
```

Which will result in output the metrics for each type of anomaly:

```
Type 1
        consinces : 5.0
        norm_ED2_concistancy : 5.632687890463552
        prop_ex : 1.0
        concistancy : 4.81575162835808
Type 3
        consinces : 5.0
        norm_ED2_concistancy : 5.5358989044102405
        prop_ex : 1.0
        concistancy : 4.790745691865299
Type 5
        consinces : 5.0
        norm_ED2_concistancy : 5.565799803588472
        prop_ex : 1.0
        concistancy : 4.798517112151137
Type 7
        consinces : 5.0
        norm_ED2_concistancy : 5.53660879213352
        prop_ex : 1.0
        concistancy : 4.7909306818472315
Type 8
        consinces : 5.0
        norm_ED2_concistancy : 5.383988276621209
        prop_ex : 1.0
        concistancy : 4.750603363439073
Type 9
        consinces : 5.0
        norm_ED2_concistancy : 5.482747132577549
        prop_ex : 1.0
        concistancy : 4.776827032138912
Type 10
        consinces : 5.0
        norm_ED2_concistancy : 5.278061512870996
        prop_ex : 1.0
        concistancy : 4.7219362594577206
Type 12
        consinces : 5.0
        norm_ED2_concistancy : 5.580094569583952
        prop_ex : 1.0
        concistancy : 4.802217667463985
Type 13
        consinces : 5.0
        norm_ED2_concistancy : 5.486878270404348
        prop_ex : 1.0
        concistancy : 4.7779136640189925
Type 14
        consinces : 5.0
        norm_ED2_concistancy : 5.136641594997203
        prop_ex : 1.0
        concistancy : 4.682753509207083
Type global
        consinces : 5.0
        norm_ED2_concistancy : 5.652298652013947
        prop_ex : 1.0
        concistancy : 4.820765790636433
```


