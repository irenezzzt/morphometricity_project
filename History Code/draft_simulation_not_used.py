

''' old ver of sim()
def sim(N, M, L, m2, n_sim, kernel = "linear", fisher="expected"):
    Va = m2*10; Ve = (1-m2)*10
    res_lin = np.ndarray(shape = (n_sim, 9))
    res_gau0 = np.ndarray(shape = (n_sim, 9))
    res_gau1 = np.ndarray(shape = (n_sim, 9))
    res_gau2 = np.ndarray(shape = (n_sim, 9))
    res_gau3 = np.ndarray(shape = (n_sim, 9))

    beta = np.random.normal(loc=0, scale=1, size = L)  # fixed effect

    # the result we would like to keep for each iter, and their types
    name_type = [('flag', int), ("iter", int), ('estimated m2', float), ('estimated sd', float),
                            ('theoretical var', float), ('residual var', float), ('reML likelihood', float),('aic', float), ('bic', float)]

    if kernel == "linear":
        for j in range(n_sim):
            np.random.seed(j*13+7)
            Z = np.random.normal(0, 2, size = (N,M)) # brain imaging
            ASM = np.corrcoef(Z)

            age = np.random.normal(56, 8 ,size=(N,1))
            sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb

            X = np.concatenate((age, sex), axis=1) # covariates
            beta0i = np.random.multivariate_normal(mean = [0] * N, cov = Va*ASM)  # random effect
            eps = np.random.normal(loc=0, scale=Ve**(1/2), size = N)  # random error
            y = beta0i + beta.dot(X.T) + eps # response

            # simulation is running with 5 kernels as below: linear, Gaussian (bw = 2, 1, 0.5, 0.25)
            # if other bandwidth or kernel is needed, can add here.
            ASM_lin = ASM
            ASM_gau0 = morphometricity.gauss_similarity(Z, width=2)
            ASM_gau1 = morphometricity.gauss_similarity(Z, width=1)
            ASM_gau2 = morphometricity.gauss_similarity(Z, width=1/2)
            ASM_gau3 = morphometricity.gauss_similarity(Z, width=1/4)

            temp = morphometricity.morph_fit(y=y, X=X, K=ASM_lin, method=fisher, max_iter=100)
            res_lin[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
                temp['iteration'], temp['Estimated morphometricity'],
                temp['Estimated standard error'], temp['Morphological variance'],
                temp['Residual variance'], temp['ReML likelihood'],
                temp['AIC'], temp['BIC'] ]

            temp = morphometricity.morph_fit(y=y, X=X, K=ASM_gau0, method=fisher, max_iter=100)
            res_gau0[j] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
                temp['iteration'], temp['Estimated morphometricity'],
                temp['Estimated standard error'], temp['Morphological variance'],
                temp['Residual variance'], temp['ReML likelihood'],
                temp['AIC'], temp['BIC'] ]

            temp = morphometricity.morph_fit(y=y, X=X, K=ASM_gau1, method=fisher, max_iter=100)
            res_gau1[j] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
                temp['iteration'], temp['Estimated morphometricity'],
                temp['Estimated standard error'], temp['Morphological variance'],
                temp['Residual variance'], temp['ReML likelihood'],
                temp['AIC'], temp['BIC'] ]

            temp = morphometricity.morph_fit(y=y, X=X, K=ASM_gau2, method=fisher, max_iter=100)
            res_gau2[j] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
                temp['iteration'], temp['Estimated morphometricity'],
                temp['Estimated standard error'], temp['Morphological variance'],
                temp['Residual variance'], temp['ReML likelihood'],
                temp['AIC'], temp['BIC'] ]

            temp = morphometricity.morph_fit(y=y, X=X, K=ASM_gau3, method=fisher, max_iter=100)
            res_gau3[j] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
                temp['iteration'], temp['Estimated morphometricity'],
                temp['Estimated standard error'], temp['Morphological variance'],
                temp['Residual variance'], temp['ReML likelihood'],
                temp['AIC'], temp['BIC'] ]


    elif kernel =="gaussian":
        for i in range(n_sim):
            np.random.seed(j*13+7)
            Z = np.random.normal(0, 2, size = (N,M)) # brain imaging
            ASM = morphometricity.gauss_similarity(Z, width=1)

            age = np.random.normal(56, 8 ,size=(N,1))
            sex = np.random.binomial(50, 0.54, size=(N,1)) # following the summary stats on ukb

            X = np.concatenate((age, sex), axis=1) # covariates
            beta0i = np.random.multivariate_normal(mean = [0] * N, cov = Va*ASM)  # random effect
            eps = np.random.normal(loc=0, scale=Ve**(1/2), size = N)  # random error
            y = beta0i + beta.dot(X.T) + eps # response

            ASM_lin = np.corrcoef(Z)
            ASM_gau0 = morphometricity.gauss_similarity(Z, width=1/2)
            ASM_gau1 = morphometricity.gauss_similarity(Z, width=1)
            ASM_gau2 = morphometricity.gauss_similarity(Z, width=2)
            ASM_gau3 = morphometricity.gauss_similarity(Z, width=4)

            temp = morphometricity.morph_fit(y=y, X=X, K=ASM_lin, method=fisher, max_iter=100)
            res_lin[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
                temp['iteration'], temp['Estimated morphometricity'],
                temp['Estimated standard error'], temp['Morphological variance'],
                temp['Residual variance'], temp['ReML likelihood'],
                temp['AIC'], temp['BIC'] ]


            temp = morphometricity.morph_fit(y=y, X=X, K=ASM_gau0, method=fisher, max_iter=100)
            res_gau0[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
                temp['iteration'], temp['Estimated morphometricity'],
                temp['Estimated standard error'], temp['Morphological variance'],
                temp['Residual variance'], temp['ReML likelihood'],
                temp['AIC'], temp['BIC'] ]

            temp = morphometricity.morph_fit(y=y, X=X, K=ASM_gau1, method=fisher, max_iter=100)
            res_gau1[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
                temp['iteration'], temp['Estimated morphometricity'],
                temp['Estimated standard error'], temp['Morphological variance'],
                temp['Residual variance'], temp['ReML likelihood'],
                temp['AIC'], temp['BIC'] ]

            temp = morphometricity.morph_fit(y=y, X=X, K=ASM_gau2, method=fisher, max_iter=100)
            res_gau2[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
                temp['iteration'], temp['Estimated morphometricity'],
                temp['Estimated standard error'], temp['Morphological variance'],
                temp['Residual variance'], temp['ReML likelihood'],
                temp['AIC'], temp['BIC'] ]

            temp = morphometricity.morph_fit(y=y, X=X, K=ASM_gau3, method=fisher, max_iter=100)
            res_gau3[i] = [ (temp['flag'] == 'ReML algorithm has converged')*1,
                temp['iteration'], temp['Estimated morphometricity'],
                temp['Estimated standard error'], temp['Morphological variance'],
                temp['Residual variance'], temp['ReML likelihood'],
                temp['AIC'], temp['BIC'] ]

        res_lin = {'flag' : res_lin[:,0],
                   'iteration' : res_lin[:,1],
                   'estimated m2' : res_lin[:,2],
                   'estimated sd' : res_lin[:,3],
                   'theoretical sd': math.sqrt(res_lin[:,4]),
                   'residual var': res_lin[:,5],
                   'ReML likelihood': res_lin[:,6],
                   'aic': res_lin[:,7],
                   'bic': res_lin[:,8]
                   }

        res_gau0 = {'flag' : res_gau0[:,0],
                   'iteration' : res_gau0[:,1],
                   'estimated m2' : res_gau0[:,2],
                   'estimated sd' : res_gau0[:,3],
                   'theoretical sd': math.sqrt(res_gau0[:,4]),
                   'residual var': res_gau0[:,5],
                   'ReML likelihood': res_gau0[:,6],
                   'aic': res_gau0[:,7],
                   'bic': res_gau0[:,8]
                   }

        res_gau1 = {'flag' : res_gau1[:,0],
                   'iteration' : res_gau1[:,1],
                   'estimated m2' : res_gau1[:,2],
                   'estimated sd' : res_gau1[:,3],
                   'theoretical sd': math.sqrt(res_gau1[:,4]),
                   'residual var': res_gau1[:,5],
                   'ReML likelihood': res_gau1[:,6],
                   'aic': res_gau1[:,7],
                   'bic': res_gau1[:,8]
                   }

        res_gau2 = {'flag' : res_gau2[:,0],
                   'iteration' : res_gau2[:,1],
                   'estimated m2' : res_gau2[:,2],
                   'estimated sd' : res_gau2[:,3],
                   'theoretical sd': math.sqrt(res_gau2[:,4]),
                   'residual var': res_gau2[:,5],
                   'ReML likelihood': res_gau2[:,6],
                   'aic': res_gau2[:,7],
                   'bic': res_gau2[:,8]
                   }

        res_gau3 = {'flag' : res_gau3[:,0],
                   'iteration' : res_gau3[:,1],
                   'estimated m2' : res_gau3[:,2],
                   'estimated sd' : res_gau3[:,3],
                   'theoretical sd': math.sqrt(res_gau3[:,4]),
                   'residual var': res_gau3[:,5],
                   'ReML likelihood': res_gau3[:,6],
                   'aic': res_gau3[:,7],
                   'bic': res_gau3[:,8]
                   }


        return{'res_lin': res_lin, 'res_gau0': res_gau0, 'res_gau1':res_gau1, 'res_gau2':res_gau2, 'res_gau3':res_gau3}

    else:
        return['Input kernel is not supported']

'''
# %%
