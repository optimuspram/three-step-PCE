function ypreds = PCE_ensemble_soft_mixture_demo(X)
% Load the classification model
load classification_and_local_models net myPCE_KS

[idl,clas] = net.predict(X); % Predict label

yp1 =  uq_evalModel(myPCE_KS{1},X);
yp2 =  uq_evalModel(myPCE_KS{2},X);
yp3 =  uq_evalModel(myPCE_KS{3},X);
ypreds = yp1.*clas(:,1) + yp2.*clas(:,2) + yp3.*clas(:,3);

