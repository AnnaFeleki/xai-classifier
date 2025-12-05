# DeepFCM-PSO
This is the official implementation of
[https://www.mdpi.com/2076-3417/13/21/11953](https://www.mdpi.com/2076-3417/13/21/11953)

This repository provides the implementation of DeepFCM, an explainable AI framework for diagnosing Coronary Artery Disease using Myocardial Perfusion Imaging (MPI) and clinical data, as presented in “Explainable Deep Fuzzy Cognitive Map Diagnosis of Coronary Artery Disease: Integrating Myocardial Perfusion Imaging, Clinical Data, and Natural Language Insights” (Applied Sciences, 2023). DeepFCM combines a lightweight RGB-CNN for processing polar-map images with an FCM-based classifier that incorporates selected clinical features. The system produces not only accurate predictions but also clear visual and textual explanations using Grad-CAM and a natural-language interpretation module. This repository includes the complete pipeline—image preprocessing, CNN training, FCM modelling, PSO-based optimization, evaluation, and explainability tools—allowing users to reproduce the results reported in the study and adapt the method to their own datasets.

# Paper abstract:
Myocardial Perfusion Imaging (MPI) has played a central role in the non-invasive identifi
cation of patients with Coronary Artery Disease (CAD). Clinical factors, such as recurrent diseases,
 predisposing factors, and diagnostic tests, also play a vital role. However, none of these factors offer
 a straightforward and reliable indication, making the diagnosis of CAD a non-trivial task for nuclear
 medicine experts. While Machine Learning (ML) and Deep Learning (DL) techniques have shown
 promise in this domain, their “black-box” nature remains a significant barrier to clinical adoption,
 a challenge that the existing literature has not yet fully addressed. This study introduces the Deep
 
