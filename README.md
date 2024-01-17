# **Training Attention and Validation Attention Consistency (TAVAC)**

**Authors**: Yue Zhao, Dylan Agyemang, Matt Mahoney, Sheng Li


  **Contact**: sheng.li@jax.org


## Description 
Overfitting affects model interpretation when predictions are made out of random noise. To address this issue, we introduce a novel metric – Training Attention and Validation Attention Consistency (TAVAC) – for evaluating ViT model degree of overfitting on imaging datasets and quantify the reproducibility of interpretation.

<img width="946" alt="image" src="https://github.com/LabShengLi/TAVAC/assets/8755378/fe2aa47a-be5b-4796-bc5c-cd12cac0a967">

## Sample run

While in TAVAC directory:

```sh
$ python experiments/VTransformerCrossValidationTumorPred_hugface.py --patient_id 0
$ python experiments/VTransformerCrossValidationTumorPred_hugface.py --patient_id 1
```
Then we can run all cells in experiments/Vit2Stage_attention_consistency-st-net_tumor_classification.ipynb

Please run all cells for tst_patient = 'Stage1' and tst_patient = 'Stage2' in cell [3] respectively

Then we can run all cells in experiments/MetricsCalculation-stnet-tumorPred.ipynb


