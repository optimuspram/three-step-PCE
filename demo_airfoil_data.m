clc;
clear all;
close all;

uqlab % Initialize UQlab

% The dataset for this demo is taken from:
% Kawai, S., & Shimoyama, K. (2014). Kriging-model-based uncertainty
% quantification in computational fluid dynamics.
% In 32nd AIAA Applied Aerodynamics Conference (p. 2737).

% The first input variable is Mach number [-], with distribution normal(0.729, 0.005)
% The second input variable is angle of attack [deg], with distribution normal(2.31, 0.2)

load RAE_2822_DATA X_RAE % Load data set

ms = [0.729, 2.31]; % Mean of random inputs
ss = [0.005, 0.2]; % Standard deviation of random inputs
nsamp = 200; % Number of training points
nvar = 2; % Size of input variables

X_all = X_RAE(:,1:2); % All points
Y_all = X_RAE(:,3:5); % Responses at all points;

X_all_norm = (X_all-ms)./ss; % Normalized inputs
Y_all_norm = (Y_all-mean(Y_all))./std(Y_all); % Normalized output

X_train = X_all_norm(1:nsamp,1:2); % Training points
Y_train = Y_all_norm(1:nsamp,3); % Responses at training points;

X_test = X_all_norm(nsamp+1:end,1:2); % Test points
Y_test = Y_all_norm(nsamp+1:end,3); % Responses at test points


%% Using conventional PCE

% Set the PCE metamodel
PCEOpts.Type = 'Metamodel';
PCEOpts.MetaType = 'PCE';
PCEOpts.TruncOptions.qNorm = 1;
PCEOpts.Degree = [1:3];
varnames = {'M','AoA'};
for im=1:nvar
    InputOptsN.Marginals(im).Type = 'Gaussian'; % Kernel density
    InputOptsN.Marginals(im).Parameters =  [0,1]; % Samples for density estimation
    InputOptsN.Marginals(im).Name = varnames{im};
end

% Generate the PCE metamodels
myInputN = uq_createInput(InputOptsN);
PCEOpts.ExpDesign.X = X_train;
PCEOpts.ExpDesign.Y = Y_train;
myPCEI = uq_createModel(PCEOpts); % Save the PCE model
Y_pred_PCE = uq_evalModel(myPCEI,X_test);


%% Three-step strategy
% Clustering
XCOMB = [X_train Y_train]; % Combined training set
nclust = 3; % Number of cluster
GMModel = fitgmdist(XCOMB,nclust); % GMM with three clusters
P = posterior(GMModel, XCOMB);
[~,Y_lab] = max(P,[],2);

Y_train_lab = Y_lab(1:nsamp,1); % Labeling based on clustering (training set)
Y_test_lab = Y_lab(nsamp+1:end,1);  % Labeling based on clustering (test set)

% Classification using deep learning
net = fitcnet(X_train, Y_train_lab,"LayerSizes",[40 40 40],'Activations','tanh');

[idl,clas] = net.predict(X_test); % Predict label at test set

% Build local models
for LOOP = 1:3
    [IN] = find(Y_train_lab==LOOP); % Find solutions that belong to the cluster
    for im=1:nvar
        InputNew.Marginals(im).Type = 'KS'; % Kernel density estimation
        InputNew.Marginals(im).Parameters =  X_train(IN,im); % Samples for density estimation
    end

    myInputN = uq_createInput(InputNew);
    PCEOpts.ExpDesign.X = X_train(IN,:);
    PCEOpts.ExpDesign.Y = Y_train(IN,1);
    myPCE_KS{LOOP} = uq_createModel(PCEOpts); % Save the PCE model
end

% Save the neural net and PCE into a file
save classification_and_local_models myPCE_KS net

% Predictions
Y_pred_soft = PCE_ensemble_soft_mixture_demo(X_test); % Soft mixture
Y_pred_hard = PCE_ensemble_hard_mixture_demo(X_test); % Hard mixture

% Calculate NMAE and RMSE
NMAE_PCE= mean(abs(Y_test(:,1)-Y_pred_PCE(:,1)))./iqr(Y_all_norm(:,1));
NMAE_soft = mean(abs(Y_test(:,1)-Y_pred_soft(:,1)))./iqr(Y_all_norm(:,1));
NMAE_hard = mean(abs(Y_test(:,1)-Y_pred_hard(:,1)))./iqr(Y_all_norm(:,1));
RMSE_PCE = sqrt(mean(abs(Y_test(:,1)-Y_pred_PCE(:,1)).^2))./iqr(Y_all_norm(:,1));
RMSE_soft = sqrt(mean(abs(Y_test(:,1)-Y_pred_soft(:,1)).^2))./iqr(Y_all_norm(:,1));
RMSE_hard = sqrt(mean(abs(Y_test(:,1)-Y_pred_hard(:,1)).^2))./iqr(Y_all_norm(:,1));

figure()
scatter3(X_test(:,1),X_test(:,2),Y_pred_PCE,'rx'); hold on
scatter3(X_test(:,1),X_test(:,2),Y_pred_soft,'md');
scatter3(X_test(:,1),X_test(:,2),Y_pred_hard,'g+');
scatter3(X_test(:,1),X_test(:,2),Y_test,'bo');
legend({'PCE','Soft','Hard','Test'});
xlabel('M (normalized)');
ylabel('AoA (normalized)');

%% Explainability and Sobol indices

% SOBOL INDICES
ModelOpts.mFile = 'PCE_ensemble_soft_mixture_demo';
myInputN = uq_createInput(InputOptsN);
myModel = uq_createModel(ModelOpts);
SobolOpts.Type = 'Sensitivity';
SobolOpts.Method = 'Sobol';
SobolOpts.Sobol.Order = 3;
SobolOpts.Sobol.SampleSize = 1e5;
mySobolAnalysisMC = uq_createAnalysis(SobolOpts);

% Plot total sobol indices
figure()
bar(mySobolAnalysisMC.Results.Total);
set(gca,'xtick',[1 2])
set(gca,'xticklabels',varnames)
ylabel('Total Sobol indices');

% Plot first order Sobol indices
figure()
bar(mySobolAnalysisMC.Results.FirstOrder)
set(gca,'xtick',[1 2])
set(gca,'xticklabels',varnames)
ylabel('First order Sobol indices');
%% REGRESSION SHAP
func = @(x) PCE_ensemble_soft_mixture_demo(x);
[SHAP,y_c] = KERNEL_SHAP(func, X_test(1:200,:),[0,0]);

% Plot mean averaged SHAP
figure()
bar(mean(abs(SHAP)));
set(gca,'xtick',[1 2])
set(gca,'xticklabels',varnames)
ylabel('Averaged SHAP');

% Plot SHAP dependence plots
figure()
scatter(X_test(1:200,1),SHAP(:,1)); hold on
scatter(X_test(1:200,2),SHAP(:,2));
legend({'M','AoA'});
xlabel('Normalized input');
ylabel('SHAP');
axis([-2,2,-2,1.5])

%% PDP and ICE
xz = sort(randn(200,1)'); % For plotting
nk = 200; % Number of ICE lines to plot
nz = length(xz);
for jj = 1:nvar
    for ii = 1:nk
        xp = X_test(ii,:); XD = repmat(xp,nz,1);
        XD(:,jj) = xz';
        predd = PCE_ensemble_soft_mixture_demo(XD);
        y_ice_PCE_soft{jj}(ii,:) = predd';
        ii
    end
end

% Calculate PD feature importance for the first and second variable
for ii = 1:2
    y_pdp = mean(y_ice_PCE_soft{ii})
    PDfi(ii,1) = sqrt((1/(nk-1))*sum((y_pdp-mean(y_pdp)).^2));
end

% Plot PDP feature importance
figure()
bar(PDfi);
set(gca,'xtick',[1 2])
set(gca,'xticklabels',varnames)
ylabel('PDP feature importance');

% Plot ICE and PDP for the first variable
figure()
y_ice = y_ice_PCE_soft{1};
plot(xz,mean(y_ice),'r-','LineWidth',1);
hold on
K = length(xz);
jm = 1;
for ii = 1:nk
    plot1 = plot(xz,y_ice (ii,:),'b-','LineWidth',0.1);
    plot1.Color(4) = 0.2;
    jm = jm+K;
    hold on
end
legend({'f-PD','f-ICE'})
xlabel('Normalized input')
ylabel('f-PD,f-ICE')

% Plot ICE and PDP for the second variable
figure()
y_ice = y_ice_PCE_soft{2};
plot(xz,mean(y_ice),'r-','LineWidth',1);
hold on
K = length(xz);
jm = 1;
for ii = 1:nk
    plot1 = plot(xz,y_ice (ii,:),'b-','LineWidth',0.1);
    plot1.Color(4) = 0.2;
    jm = jm+K;
    hold on
end
legend({'f-PD','f-ICE'})
xlabel('Normalized input')
ylabel('f-PD,f-ICE')
