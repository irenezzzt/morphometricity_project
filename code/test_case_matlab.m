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
beta0i = mvnrnd(mu,8*ASM)

% error
eps = normrnd(0, sqrt(2), [1,N])

% phenotype
y= 1 + beta0i + beta*X' + eps

% normalize X,y
% y_norm = (y - mean(y))/std(y)
% x_norm = (X-mean(X))./std(X)


[flag, m2, SE, Va, Ve, Lnew]  = Morphometricity(y', X, ASM, alg=0) # average
% flag = 1
% m2 = 0.8525
% SE = 0.5075
% Va = 13.150
% Ve = 2.2749
% Lnew = -7.7757


[flag, m2, SE, Va, Ve, Lnew]  = Morphometricity(y', X, ASM, alg=1) # expected

% flag = 1
% m2 = 0.8525
% SE = 0.5075
% Va = 13.150
% Ve = 2.2749
% Lnew = -7.7757

# average and observed fisher information give identical result.


[flag, m2, SE, Va, Ve, Lnew]  = Morphometricity(y', X, ASM, alg=2) # observed
% flag = 0
% m2 = 0.5232
% SE =       0 + 0.8728i
% Va = 1.6898e+33
% Ve = 1.5398e+33
% Lnew = -81.468




% same bug in py, 
% 1. when using observed fisher info, Va and Ve just kept increasing.
% 2. estimated morphometricity can be very different from the fisher info method used
% 3. does not recover the true morphometricity from simulation 





N = 50
M = 100
L = 2
Va = 8
Ve = 2


Z = normrnd(0, 2, [N, M])
ASM = corrcoef(Z')

age = normrnd(56,8, [N,1])
sex = binornd(1, 0.54, [N,1])

X = [age,sex]

beta = normrnd(0, 1, [1, L])
mu = repelem(0, N)
beta0i = mvnrnd(mu,8*ASM)

eps = normrnd(0, sqrt(2), [1,N])

y= 1 + beta0i + beta*X' + eps
[flag, m2, SE, Va, Ve, Lnew]  = Morphometricity(y', X, ASM, alg=0) 
[flag, m2, SE, Va, Ve, Lnew]  = Morphometricity(y', X, ASM, alg=1) 
[flag, m2, SE, Va, Ve, Lnew]  = Morphometricity(y', X, ASM, alg=2) 
% all close to 0.44, not recovering the truth 

morph = 0
se = 0
lik = 0

for i = 1:10
  Z = normrnd(0, 2, [N, M]);
  ASM = corrcoef(Z');

  age = normrnd(56,8, [N,1]);
  sex = binornd(1, 0.54, [N,1]);

  X = [age,sex];

  beta = normrnd(0, 1, [1, L]);
  mu = repelem(0, N);
  beta0i = mvnrnd(mu,2*ASM);

  eps = normrnd(0, sqrt(8), [1,N]);

  y= 1 + beta0i + beta*X' + eps;
  [flag, m2, SE, Va, Ve, Lnew]  = Morphometricity(y', X, ASM, alg=0) ;
  morph(i) = m2;
  se(i) = SE;
  lik(i) = Lnew;
  
end

mean(morph) % 0.7498 when m2 should be 0.8
mean(morph) % 0.1816 when m2 should be 0.2

% still bug in my python code ??


