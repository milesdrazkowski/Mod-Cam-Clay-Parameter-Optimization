import corner
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Model
from lmfit.model import save_modelresult
from concurrent.futures import ThreadPoolExecutor
from numba import njit

@njit
def TriaxMCCFn(M, Lambda, Kappa, N, Poisson, Pc, P0, nitr = 301, dstrn = 0.0005):
    D = np.zeros((6, 6))
    De = np.zeros((6, 6))
    I = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    dpprimedSprime = np.array([1 / 3, 1 / 3, 1 / 3, 0.0, 0.0, 0.0])

    p = np.zeros(nitr)
    q = np.zeros(nitr)
    strV = np.zeros(nitr)

    V = N - Lambda * np.log(Pc) + Kappa * np.log(Pc / P0)
    Stress = np.full(6, P0, dtype='float64')
    Stress[3:] = 0.0
    Strain = np.zeros(6)

    p[0] = (Stress[0] + Stress[1] + Stress[2]) / 3.0
    q[0] = Stress[0] - Stress[2]
    F = q[0] ** 2 + (M * p[0]) ** 2 - p[0] * Pc * M ** 2

    for i in range(nitr):
        p[i] = (Stress[0] + Stress[1] + Stress[2]) / 3.0
        q[i] = Stress[0] - Stress[2]

        if F < 0:
            F = q[i]**2 + (M * p[i])**2 - p[i] * Pc * M**2
        else:
            F = 0.0
            Pc = ((q[i] / M)**2 + p[i]**2) / p[i]

        K = V * p[i] / Kappa
        G = 3.0 * K * (1.0 - 2.0 * Poisson) / (2.0 * (1.0 + Poisson))

        De[:3, :3] = K - 2.0 * G / 3.0
        for j in range(3):
            De[j, j] = K + 4.0 * G / 3.0
        for j in range(3, 6):
            De[j, j] = G

        if F < 0:
            for m in range(6):
                for n in range(6):
                    D[m, n] = De[m, n]
        else:
            dFdpprime = 2 * p[i] - Pc
            dFdq = 2 * q[i] / M**2
            qsafe = q[i] if abs(q[i]) > 1e-12 else 1e-12
            dqds = (3.0 / 2.0) * (1.0 / qsafe) * (Stress - (p[i]) * I)
            dFds = dFdpprime * dpprimedSprime + dFdq * dqds
            dFdpc = -p[i]
            dpcdevp = V * Pc / (Lambda - Kappa)
            DepDENOM1 = -dFdpc * dpcdevp * dFdpprime
            dFdstr = np.dot(De, dFds)
            DepDENOM2 = np.dot(dFds.T, dFdstr).item()
            outer_dFds = np.outer(dFds, dFds)
            DepNUM = np.dot(De, np.dot(outer_dFds, De))
            DepDENOM = DepDENOM1 + DepDENOM2
            if DepDENOM != 0:
                Dep = DepNUM / DepDENOM
            else:
                Dep = np.zeros((6, 6))
            D = De - Dep

        denom1 = D[1, 1] + D[2, 1]
        denom2 = D[1, 2] + D[2, 2]

        dStrain = np.array([
            [dstrn],
            [-dstrn * D[0, 1] / denom1 if denom1 != 0 else 0.0],
            [-dstrn * D[0, 2] / denom2 if denom2 != 0 else 0.0],
            [0.0], [0.0], [0.0]
        ])
        dStress = np.dot(D, dStrain)
        Stress += dStress
        Strain += dStrain
        depstrV = np.sum(dStrain[:3])
        p_safe = p[i] if p[i] > 1e-12 else 1e-12
        V = N - Lambda * np.log(Pc) + Kappa * np.log(Pc / p_safe)
        strV[i] = strV[i - 1] + depstrV if i > 0 else depstrV

    return strV, q

@njit
def IsoconMCCfn(Strpath, Lambda, Kappa, N, Pc):
    stps = len(Strpath)
    isoe = np.zeros((stps))
    pc = Pc

    Vi = N - Lambda * np.log(Pc) + Kappa * np.log(Pc / Strpath[0])

    for i in range(stps):
        if i == 0:
            isoe[i] = Vi - 1
        elif Strpath[i] > pc and Strpath[i - 1] <= pc:
            isoe[i] = isoe[i - 1] - (Kappa * np.log(pc / Strpath[i - 1]) + Lambda * np.log(Strpath[i] / pc))
            pc = Strpath[i]
        else:
            if Strpath[i] <= pc:
                isoe[i] = isoe[i - 1] - Kappa * np.log(Strpath[i] / Strpath[i - 1])
            else:
                isoe[i] = isoe[i - 1] - Lambda * np.log(Strpath[i] / Strpath[i - 1])
                pc = Strpath[i]

    return isoe

def MCCmodel(Strpath, M, Lambda, Kappa, N, Poisson, Pc):
    pressures = [5, 10, 20, 30]
    Strpath = Strpath[:324]
    Calce = IsoconMCCfn(Strpath, Lambda, Kappa, N, Pc)

    results = []
    for P0 in pressures:
        try:
            result = TriaxMCCFn(M, Lambda, Kappa, N, Poisson, Pc, P0)
        except ZeroDivisionError:
            result = (np.full((301,), np.nan), np.full((301,), np.nan))
        results.append(result)

    all_ev = np.concatenate([res[0] for res in results])
    all_sd = np.concatenate([res[1] for res in results])
    Results = np.hstack((np.hstack((all_ev, all_sd)), Calce))
    return Results

nitr = 301
data = np.loadtxt("Triaxial and IC Test Data.csv", dtype='float64', delimiter=',')
evdata = data[:4*nitr, 0]
sddata = data[:4*nitr, 1]
edata = data[4*nitr:, 0]
Strpath = data[4*nitr:, 1].flatten()
tmp = len(data)

ydata = np.concatenate((evdata, sddata, edata), axis=0).flatten()
Strpath = np.pad(Strpath, (0, len(ydata) - len(Strpath)), mode='constant', constant_values=0)
evdata_safe = np.where(np.abs(evdata) < 1e-12, 1e-12, np.abs(evdata))
evweights = 0.01 + np.nan_to_num(np.abs(1/(100*evdata_safe)), nan = 100)
sdweights = 0.01 + np.nan_to_num(np.abs(1/sddata), nan = 100)
eweights = 0.01 + np.nan_to_num(np.abs(1/(100*edata)), nan = 100)
fweights = np.hstack((np.hstack((evweights,sdweights)), eweights))


tparams = [ 'M', 'Lambda', 'Kappa', 'N', 'Poisson', 'Pc']
tijmodel = Model(MCCmodel, param_names=tparams)

paramin = tijmodel.make_params(M=dict(value=1.65, max=5, min=1), Lambda=dict(value=0.022, max=0.05, min=0.01),
                               Kappa=dict(value=0.0016, max=0.0019, min=0.0013),
                               N=dict(value=1.351, max=1.551, min=1.3323), Poisson=dict(value=0.2, max=0.4, min=0.10),
                               Pc=dict(value=35.0, max=120, min=20))

emcee_kws = dict(steps=3500, nwalkers=5000, burn=1000, workers=8, is_weighted=False)

def main():
    paramopt = tijmodel.fit(data=ydata, Strpath=Strpath, nan_policy='omit', params=paramin, method='emcee', fit_kws=emcee_kws)

    with open('Optpara4-26-25.txt', 'w') as opt:
        opt.write(paramopt.fit_report())
        for name in paramopt.var_names:
            values = paramopt.flatchain[name]
            lower = np.percentile(values, 2.5)
            median = np.percentile(values, 50)
            upper = np.percentile(values, 97.5)
            opt.write(f"{name}: {median:.5f} (95% CI: [{lower:.5f}, {upper:.5f}])")
        opt.write('\nMaximum Likelihood Estimation from emcee       ')
        opt.write('-------------------------------------------------')
        opt.write('Parameter  MLE Value   Median Value   Uncertainty')
        fmt = '  {:5s}  {:11.5f} {:11.5f}   {:11.5f}'.format
        for name, param in paramin.items():
            opt.write(fmt(name, param.value, paramopt.params[name].value,
                  paramopt.params[name].stderr))

        opt.write('\nError estimates from emcee:')
        opt.write('------------------------------------------------------')
        opt.write('Parameter  -2sigma  -1sigma   median  +1sigma  +2sigma')

        for name in paramin.keys():
            quantiles = np.percentile(paramopt.flatchain[name],
                                  [2.275, 15.865, 50, 84.135, 97.275])
            median = quantiles[2]
            err_m2 = quantiles[0] - median
            err_m1 = quantiles[1] - median
            err_p1 = quantiles[3] - median
            err_p2 = quantiles[4] - median
            fmt = '  {:5s}   {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}'.format
            opt.write(fmt(name, err_m2, err_m1, median, err_p1, err_p2))

    print(paramopt.fit_report())
    for name in paramopt.var_names:
        values = paramopt.flatchain[name]
        lower = np.percentile(values, 2.5)
        median = np.percentile(values, 50)
        upper = np.percentile(values, 97.5)
        print(f"{name}: {median:.5f} (95% CI: [{lower:.5f}, {upper:.5f}])")

    plt.plot(paramopt.acceptance_fraction, 'o')
    plt.title('ParamOpt')
    plt.xlabel('walker')
    plt.ylabel('acceptance fraction')
    plt.savefig('AcceptanceFrac.4-26-25.png')

    para_plot = corner.corner(paramopt.flatchain, labels=paramopt.var_names, truths=list(paramopt.params.valuesdict().values()))
    para_plot.savefig('paramplot.4-26-25.png')
    highest_prob = np.argmax(paramopt.lnprob)
    hp_loc = np.unravel_index(highest_prob, paramopt.lnprob.shape)
    mle_soln = paramopt.chain[hp_loc]
    for i, par in enumerate(paramin):
        paramin[par].value = mle_soln[i]


    print('\nMaximum Likelihood Estimation from emcee       ')
    print('-------------------------------------------------')
    print('Parameter  MLE Value   Median Value   Uncertainty')
    fmt = '  {:5s}  {:11.5f} {:11.5f}   {:11.5f}'.format
    for name, param in paramin.items():
        print(fmt(name, param.value, paramopt.params[name].value,
              paramopt.params[name].stderr))

    print('\nError estimates from emcee:')
    print('------------------------------------------------------')
    print('Parameter  -2sigma  -1sigma   median  +1sigma  +2sigma')

    for name in paramin.keys():
        quantiles = np.percentile(paramopt.flatchain[name],
                              [2.275, 15.865, 50, 84.135, 97.275])
        median = quantiles[2]
        err_m2 = quantiles[0] - median
        err_m1 = quantiles[1] - median
        err_p1 = quantiles[3] - median
        err_p2 = quantiles[4] - median
        fmt = '  {:5s}   {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}'.format
        print(fmt(name, err_m2, err_m1, median, err_p1, err_p2))

    plt.show()
    para_plot.show()

    save_modelresult(paramopt, 'Paramopt4-26-25.sav')

if __name__ == '__main__':
    main()