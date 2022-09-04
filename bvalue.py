# -*- coding: utf-8 -*-

catalog = r"NKUA_SL_thiva_sample_catalogue.cat"

mc_list = [0.0, 2.2]
mmin = 0.0  # can be different from mc, for plot reasons
mmax = 6.8  # maximum accepted magnitude

mbins = 0.1  # step for magnitude bins

nmin = 5  # minimum number of events per bin

ylim = [0, 3e4]  # 0 will yield warning in loglog plot, but
                 # it should sort itself out.
                 
######################################################
"""

Read a seismic catalogue and estimate the GR-law's b and
a values through MLE and simple regression. Also plots
Frequency-Magnitude Distribution diagrams for visual validation.

Multiple magnitudes of completeness (Mc) can be used.

An extra magnitude of completeness is estimated with the Maximum Curvature
method.

"""

#-- import
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

#--
if catalog.endswith('cat'):
    # read the NKUA-SL `cat` format
    df_raw = pd.read_csv(catalog, delim_whitespace=True)
    # rename columns for compatibility
    df_raw.rename(columns={'Mag': 'Magnitude'}, inplace=True)
else:
    df_raw = pd.read_csv(catalog)

idx_mag = (mmin <= df_raw.Magnitude) & (df_raw.Magnitude <= mmax)

df = df_raw[idx_mag]

np_mag = df.Magnitude.to_numpy()

#-- ordah 66
np_mag.sort()
np_mag = np_mag[::-1]

#-- make histogram (no plot)
if type(mbins) == float:
    bins_range = np.arange(np_mag.min(), np_mag.max() + mbins, mbins)
elif type(mbins) == int:
    bins_range = mbins
freq, bins = np.histogram(np_mag, bins=bins_range)

#-- GR law / FMD
gr_freq = freq[::-1].cumsum()[::-1]  # cumulative
gr_mag = bins[:-1]

#-- Maximum Curvature for MC
grad = abs(np.gradient(gr_freq, gr_mag))
# get magnitude of completness with Maximum Curvature
mc_maxcur = np.round(gr_mag[grad.argmax()], 1)
if not mc_maxcur in mc_list: 
    mc_list.append(mc_maxcur)
mc_list.sort()

#-- plot the Maximum Curvature plot to validate the MC
plt.plot(gr_mag, grad)
plt.axvline(gr_mag[grad.argmax()], label=r'$M_{c}=%.1f$' % mc_maxcur)
plt.gca().set_yscale('log')
plt.grid(ls='-', which='major')
plt.grid(ls='--', which='minor')
plt.ylabel(r'$\frac{dN}{dM}$', rotation=0, fontsize=14)
plt.xlabel('$Magnitude \ [0.1 \ bins]$')
plt.legend(loc='upper right')
plt.savefig('maximum_curvature_mc.png', dpi=300)

#-- start iterating over MCs, find a & b and plot the FMD
for mc in mc_list:
    print(f'>> working on Mc={mc}')
    #-- estimate b and a
    idx_mc = gr_mag >= mc
    xdata = gr_mag[idx_mc]
    ydata = np.log10(gr_freq[idx_mc])

    #-- get clean data (ie remove N<minN)

    if nmin:
        idx_n = ydata >= np.log10(nmin)  # keep these above Nmin
    else:
        idx_n = True

    gr_regr_x = xdata[idx_n]
    gr_regr_y = ydata[idx_n]

    #-- maximum likelhood b (Aki, 1965; Bender, 1983; Utsu, 1999)
    b_ml = np.log(10) * np.log10(np.e) / (gr_regr_x.mean() - (mc - (mbins / 2)))
    a_ml = gr_regr_y[0] + b_ml * gr_regr_x[0]

    print('>> Maximum likelihood a, b: {:.3f}, {:.3f}'.format(a_ml, b_ml))

    xpredict = np.arange(xdata.min(), xdata.max() + 0.1, 0.1)

    _eq_str_ = 'y ~ x'
    model = ols(_eq_str_, dict(x=gr_regr_x, y=gr_regr_y))
    res = model.fit()
    a_value, b_value = res.params
    a_se, b_se = res.bse
    r2 = 100 * res.rsquared
    n_obs = int(res.nobs)

    ypredict = a_value + b_value * xpredict

    ypredict_ml = a_ml - b_ml * xpredict

    #-- get prediction related quantities
    pred_se_r = np.sqrt(res.scale)
    # pred_se_r = np.sqrt(np.sum(res.resid ** 2) / (n_obs - 2))
    pred_se_lo = (a_value + b_value * xpredict) - 2 * pred_se_r
    pred_se_hi = (a_value + b_value * xpredict) + 2 * pred_se_r
    # pred_se_r_perc = np.round(100 * sum((pred_se_lo <= gr_regr_y) & (gr_regr_y <= pred_se_hi)) / gr_regr_y.size, 2)

    _se_str_ = '$δa=%.2f, δb=%.2f$' % (a_se, b_se)

    #-- plot the results
    plt.close()
    plt.gca().set_yscale('log')
    plt.scatter(gr_mag, gr_freq, marker='s', ec='k', color='b', label='cumulative')
    plt.scatter(bins[:-1], freq, marker='s', ec='k', color='w', label='discrete')
    plt.plot(xpredict, 10 ** ypredict, color='r', label=_se_str_)
    plt.plot(xpredict, 10 ** ypredict_ml, color='k', ls='dotted', label='MLE')
    plt.plot(xpredict, 10 ** pred_se_lo, color='r', ls='--', label='regression error')
    plt.plot(xpredict, 10 ** pred_se_hi, color='r', ls='--')
    plt.grid(ls='-', which='major')
    plt.grid(ls='--', which='minor')
    plt.xlabel('Magnitude [0.1 bins]')
    plt.ylabel('# Events')
    if mc != mmin:
        plt.axvline(mc, label='$M_{c}=%.1f$' % mc)
    plt.legend(loc='upper right')
    _eq_title_ = '$\log{N} = %.2f - %.2fM \ [R^{2}=%.0f \%%]$' % (a_value, abs(b_value), r2)

    plt.gca().set_title(_eq_title_)

    if ylim:
        plt.ylim(ylim)

    plt.savefig('b_value_{:.1f}.png'.format(mc), dpi=300)
    plt.show()