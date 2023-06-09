# XGBCircM6A
A tool for predicting the m6A site of circRNA based on XGBoost. DeepCircm6a is a tool for circRNA-m6A predicting. Here we build a model based on XGBoost algorithm to predict the m6A site on circRNA. The model has good prediction ability after training, the prediction accuracy has reached 0.976, and the prediction performance of the test set is also excellent. You can download this tool directly to predict your data, or use your data to train it before use.  
# Dependencies  
Python v3.8.8;  
numpy v1.19.4;  
pandas v1.2.4;  
pytorch v1.6.0;  
biopython v1.77;  
sklearn v0.24.2;  
argparse v1.1;  
matplotlib v3.4.3;  
tqdm v4.55.1;  
xgboost v1.6.2;  
re v2.2.1;  
joblib v1.0.1
# Usage  
## 1.Pretreatment
  
Before inputting the data into our prediction tool, you need to extract your sequence into a 65bp sequence centered on base A.  
  
## 2.Prediction
  
In this part, you can directly use the model we have built to predict the m6A site. The command is as follows:  
  
```python predict.py input_fa (51bp) -model_path (path of checkpoint) -outfile (filename of output)```  
  
In addition, you can use ```model_ 5_ folds_ CV_ 2. py ```to select different data to train our model, and use```test_ 2. py ```to test the prediction performance of the model. Here we provide ```test_ Data. tar. gz ```for your use.  
