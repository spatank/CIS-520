clc; close all; clear;

TP_1 = 280;
FN_1 = 20;
FP_1 = 100;
TN_1 = 19600;
TPR_1 = TP_1/(TP_1+FN_1);
TNR_1 = TN_1/(TN_1+FP_1);
Precision_1 = TP_1/(TP_1+FP_1);
GM_1 = sqrt(TPR_1 * TNR_1);
F1_1 = (2*Precision_1*TPR_1)/(Precision_1+TPR_1);


TP_2 = 270;
FN_2 = 30;
FP_2 = 60;
TN_2 = 12140;
TPR_2 = TP_2/(TP_2+FN_2);
TNR_2 = TN_2/(TN_2+FP_2);
Precision_2 = TP_2/(TP_2+FP_2);
GM_2 = sqrt(TPR_2*TNR_2);
F1_2 = (2*Precision_2*TPR_2)/(Precision_2+TPR_2);
