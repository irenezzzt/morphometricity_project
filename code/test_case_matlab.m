pkg load statistics

% image and asm
Z =  [1 2 3; 2 1 2; 3 5 1; 1 0 -3; 0 5 2]
ASM = corrcoef(Z')

% covariates
X = [0 1 20 ; 1 1 30; 1 0 40 ; 0 1 35; 0 0 25]
[N, L] = size(X)

% fixed effect
beta = normrnd(0, 10, [1, L])

% random effect 
mu = repelem(0, N)
beta0i = mvnrnd(mu,100*ASM)

% error
eps = normrnd(0, 1.5, [1,N])

% phenotype
y= 1 + beta0i + beta*X' + eps

% normalize X,y
y_norm = (y - mean(y))/std(y)
x_norm = (X-mean(X))./std(X)


[flag, m2, SE, Va, Ve, Lnew]  = Morphometricity(y', X, ASM, alg=2)
% flag = 0
% m2 = 0.5232
% SE =       0 + 0.8728i
% Va = 1.6898e+33
% Ve = 1.5398e+33
% Lnew = -81.468

% same bug in py, Va and Ve just kept increasing... need normalization in each update of Va and Ve. 
% pobviously Vy = 1 after normalization, I coded this step within the algorithm.