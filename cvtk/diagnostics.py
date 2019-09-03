import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

def calc_diagnostics(covs, mean_hets, seqids, tile_depths, 
                     offdiag_k=1, exclude_seqids=None, depth_limits=None):
    """

    Calculate the diagonal and off-diagonal elements of the variance-covariance
    matrix (usually a single tile), load into a dataframe with the average depth of 
    each tile, and run a regression on depth by covariance and varaince. This is run
    in the TiledTemporalFreqs.correction_diagnostics() method 

    offdiag_k: 1 is the adjacent timepoint covariances. Other options are for comparing
               the magnitude of the covariances to.

    """
    offdiag = np.array([np.diag(c, k=offdiag_k).mean() for c in covs])
    diag = np.array([np.diag(c, k=0).mean() for c in covs])

    # load all this into a dataframe for model fitting
    df = pd.DataFrame(dict(mean_hets=mean_hets, offdiag=offdiag, diag=diag,
                           seqid=seqids, depth=tile_depths))
    if depth_limits is not None:
        df = df[(df['depth'] > depth_limits[0]) & (df['depth'] < depth_limits[1])]
    # we fit the model optionally excluding some seqids
    # but we return the *full* dataframe, which we make a 
    # copy of here.
    full_df = df
    if exclude_seqids is not None:
        df = df[~df.seqid.isin(exclude_seqids)]
    diag_fit = smf.ols("diag ~ depth", df).fit()
    depth = df['depth'].dropna()
    diag_ypred = diag_fit.predict(dict(depth=depth))
    assert(len(diag_ypred.index) == len(depth))
    offdiag_fit = smf.ols("offdiag ~ depth", df).fit()
    offdiag_ypred = offdiag_fit.predict(dict(depth=depth))
    models = (diag_fit, offdiag_fit)
    xpreds = depth
    ypreds = (diag_ypred, offdiag_ypred)
    return models, xpreds, ypreds, full_df
