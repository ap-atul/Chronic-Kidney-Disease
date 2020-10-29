# Chronic-Kidney-Disease
Prediction on Chronic Kidney Disease using Sklearn and Custom ML.

## Instructions to run
(requires python 3+)
1. Install all libraries using(requirements file)
```
$ pip3 install -r requirements.txt
```
2. Run the ui.py file
```
$ python3 ui.py
```

## Screen

![screen](https://github.com/AP-Atul/Chronic-Kidney-Disease/blob/master/charts/window.png)

## Few Notes
The UCI Machine Learning Repository data set includes: 
Link :: https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease
We use 24 + class = 25 ( 11 numeric ,14 nominal) 

Training : 127 records
Testing : 33 records

* ge - age
* bp - blood pressure
* sg - specific gravity
* al - albumin
* su - sugar
* rbc - red blood cells
* pc - pus cell
* pcc - pus cell clumps
* ba - bacteria
* bgr - blood glucose random
* bu - blood urea
* sc - serum creatinine
* sod - sodium
* pot - potassium
* hemo - hemoglobin
* pcv - packed cell volume
* wc - white blood cell count
* rc - red blood cell count
* htn - hypertension
* dm - diabetes mellitus
* cad - coronary artery disease
* appet - appetite
* pe - pedal edema
* ane - anemia
* class - class

## Classification Algo
Classes 
1. Chronic (ckd)
2. Not Chronic (notckd)

 * Logistic Regression
 * Naive Bayes
 * KNN

## Accuracies

(These are the saved models accuracies)
1. KNN accuracy: 
    * Custom : 90.62
    * SKLearn : 90.62

2. NB accuracy:
    * Custom : 100
    * SKLearn : 100

3. LR accuracy:
    * Custom : 68.75
    * SKLearn : 100

## Directory details
1. dataset : processed csv file
2. charts : plots to visualize data
3. lib : custom implementations of all the algos
4. model : saved pre-trained model (both custom and inbuilt)
5. custom/ inbuilt : runner files to to prediction (training also)