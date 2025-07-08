# Graph-based Brain Connectivity Analysis for Seizure On-set Prediction

This project aims to develop a robust, real-time seizure prediction
system using Brain Network Transformers (BNTs) to analyze brain connectivity from electroencephalogram (EEG)
data. Unlike traditional methods that overlook the dynamic connectivity
between brain regions, our graph-based approach models these intricate re-
lationships, offering superior accuracy and explainability critical for clinical
applications.
The proposed system processes raw EEG signals into graph representa-
tions, where nodes correspond to brain regions and edges reflect functional
connectivity. These graphs are analyzed using GNNs and BNTs, leveraging
their ability to capture both local and global patterns in brain connectivity.
The model is optimized for deployment on GPU-enabled Nvidia Jetson de-
vices, ensuring real-time inference on live EEG data. Additionally, a mobile
application, developed using React Native Expo, will visualize brain activ-
ity graphs and provide timely notifications for pre-ictal states, enhancing
accessibility and usability for caregivers and medical professionals.
This project employs open-source EEG datasets for training and val-
idation, emphasizing portability, cost-effectiveness, and scalability. By in-
tegrating real-time data acquisition, graph-based analysis, and user-friendly
mobile interfaces, the system has the potential to transform epilepsy man-
agement, particularly in resource-constrained settings. Through its focus
on brain connectivity and cutting-edge graph-based techniques, this solution
represents a significant advancement in seizure prediction technology.
<br>

This repo contains code that is to be deployed on the nvidia jetson. For other components of this project prefer to the following repos: <br>
- <a href="https://github.com/M-Ali-Haider/FYPBackend/tree/main">Backend for mobile and web application</a>
- <a href="https://github.com/M-Ali-Haider/fyp-web"> Frontend for web app </a>
- <a href="https://github.com/M-Ali-Haider/fyp-app"> Frontend for mobile app </a>
- <a href="https://github.com/Humzafazal72/BrainNetworkTransformer-with-Lazy-Data-Loaders"> Model Training </a>
- <a href="https://github.com/M-Ali-Haider/Brain-Connectivity-Analysis"> Data preprocessing </a>
