# three-step-PCE
This code is associated with the publication 'Global Sensitivity Analysis of Stochastic Re-entry Trajectory using Explainable Surrogate Models.'

In that publication, we utilized a three-step Polynomial Chaos Expansion (PCE) surrogate modelling approach to approximate the relationship between ground-reaching velocity, falling range, and falling time as functions of seven random input variables. The discontinuity in the response surface is an inherent characteristic of the problem, necessitating the use of the three-step strategy. The data was then subjected to explainability modules to extract key insights. The original dataset was produced using JAXA's trajectory analysis module and is not publicly accessible. However, the specific trajectory analysis dataset used in this paper can be provided upon reasonable request.

This code example illustrates the three-step-PCE approach using a two-variable airfoil uncertainty quantification problem, sourced from:

Kawai, S., & Shimoyama, K. (2014). Kriging-model-based uncertainty quantification in computational fluid dynamics. In 32nd AIAA Applied Aerodynamics Conference (p. 2737)

The input variables are the Mach number, modelled with a normal distribution (mean = 0.729, standard deviation = 0.005), and the angle of attack, also with a normal distribution (mean = 2.31 degrees, standard deviation = 0.2 degrees). The output of interest is the pressure coefficient  at the upper surface of the airfoil. 

In this demonstration, we illustrate the deployment of the three-step PCE, beginning with clustering, then classification, followed by local model building. We also show how to apply explainability modules (such as SHAP, PDP, ICE) and Sobol indices to derive key insights and visualize the inner workings of the input-output relationship.

The code was run using MATLAB R2023a. To fully run the code, you need the following modules:
- MATLAB Statistics and Machine Learning toolbox for clustering, neural net-based classification, and kernel density estimates.
- SHAPMODE to calculate Shapley values (https://github.com/optimuspram/SHAPMODE)
- UQlab to build the PCE model (https://www.uqlab.com/)

If you have questions, you can direct your questions to: pramsp@itb.ac.id

Pramudita Satria Palar
Assistant Professor 
Faculty of Mechanical and Aerospace Engineering 
Bandung Institute of Technology Indonesia
