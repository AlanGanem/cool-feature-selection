from toolz import curry
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

@curry
def fast_metric_with_ci_(data, *, n_samples=100, ci_level=0.95,
                         prediction='prediction', target='target', weight='weight', metric_fn = roc_auc_score):

    data = data.assign(weight=lambda df: df[weight] if weight is not None else 1)

    summary = (
        data
            .assign(
            prediction=lambda df: (1000 * df[prediction]).round(),
        )
            .groupby(["weight", 'prediction', target])
            .size().to_frame("sample_size")
            .reset_index()
    )

    estimate = (
        summary
            .assign(weight=lambda df: df["weight"] * df['sample_size'])
            .pipe(lambda df: metric_fn(df[target], df['prediction'], sample_weight=df['weight']))
    )

    bs_values = [
        summary
            .assign(weight=lambda df: df["weight"] * np.random.poisson(df['sample_size']))
            .pipe(lambda df: metric_fn(df[target], df['prediction'], sample_weight=df['weight']))
        for _ in range(n_samples)]

    lo, hi = bootstrap_ci(estimate, bs_values, ci_level=ci_level)

    return pd.Series(dict(
        estimate=estimate,
        ci_upper=hi,
        ci_lower=lo,
        model=prediction
    ))

@curry
def fast_auc_with_ci_sklearn(y_true, y_pred, *, sample_weight=None, n_samples=30, ci_level=0.95,):

    data = pd.DataFrame([y_true,y_pred], columns = ["target", "prediction"])
    if sample_weight is None:
        data["weight"] = 1
    else:
        data["weight"] = sample_weight

    return fast_metric_with_ci(data, n_samples=n_samples, ci_level=ci_level, prediction='prediction', target='defaulted', weight='weight', metric_fn = roc_auc_score)


def bootstrap_ci(sample_estimate, bootstrap_estimates, ci_level=0.95):
    lo = 2 * sample_estimate - np.quantile(bootstrap_estimates, (1 + ci_level) / 2)
    hi = 2 * sample_estimate - np.quantile(bootstrap_estimates, (1 - ci_level) / 2)
    return lo, hi



@curry
def fast_delta_metric_with_ci_(data, baseline, challenger, *, n_samples=100, ci_level=0.95,
                               target='target', weight='weight', metric_fn = roc_auc_score):

    data = data.assign(weight=lambda df: df[weight] if weight is not None else 1)

    summary = (
        data
            .assign(**{
            baseline: lambda df: (1000 * df[baseline]).round(),
            challenger: lambda df: (1000 * df[challenger]).round(),
        })
            .groupby(["weight", baseline, challenger, target])
            .size().to_frame("sample_size")
            .reset_index()
    )

    def delta_auc(df):
        challenger_auc = metric_fn(df[target], df[challenger], sample_weight=df['weight'])
        baseline_auc = metric_fn(df[target], df[baseline], sample_weight=df['weight'])
        return challenger_auc - baseline_auc

    estimate = (
        summary
            .assign(weight=lambda df: df["weight"] * df['sample_size'])
            .pipe(delta_auc)
    )

    bs_values = [
        summary
            .assign(weight=lambda df: df["weight"] * np.random.poisson(df['sample_size']))
            .pipe(delta_auc)
        for _ in range(n_samples)]

    lo, hi = bootstrap_ci(estimate, bs_values, ci_level=ci_level)

    return pd.Series(dict(
        estimate=estimate,
        ci_upper=hi,
        ci_lower=lo,
        model=challenger
    ))


@curry
def fast_delta_metric_with_ci(data, baseline, challengers, target, *, n_samples=100, ci_level=0.95, weight='weight', metric_fn = roc_auc_score):

    fn = fast_delta_metric_with_ci_(
        baseline=baseline,
        n_samples=n_samples,
        ci_level=ci_level,
        target=target,
        weight=weight,
        metric_fn=metric_fn
    )

    all_values = [fn(data=data,challenger=c) for c in challengers]

    return pd.DataFrame(all_values)

@curry
def fast_metric_with_ci(data, predictions, target, *, n_samples=100, ci_level=0.95, weight='weight', metric_fn = roc_auc_score):

    fn = fast_metric_with_ci_(
        target=target,
        n_samples=n_samples,
        ci_level=ci_level,
        weight=weight,
        metric_fn=metric_fn
    )

    all_values = [fn(data=data,prediction=p) for p in predictions]

    return pd.DataFrame(all_values)