import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from metrics import fast_metric_with_ci, fast_delta_metric_with_ci

def log_odds_to_proba(x):
    return 1/(1+np.exp(-x))


def backwards_shap_feature_selection(
        model,
        df_train,
        df_val,
        candidate_features_for_removal,
        target,
        null_hypothesis = "all_features_are_bad",
        fixed_features=[],
        sample_weight=None,
        metric_fn = roc_auc_score,
        n_samples=20,
        ci_level=0.8,
        max_iter = 10,
        max_removals_per_run=None,
        return_estimators=False,
        return_explainers=False,
):


    valid_nulls = ["all_features_are_bad","all_features_are_good"]
    if not null_hypothesis in valid_nulls:
        raise(ValueError(f"null_hypothesis should be one of {valid_nulls}, got {null_hypothesis}"))
    #parse params
    df_train = df_train.assign(weight__ = lambda d: d[sample_weight] if sample_weight is not None else 1)
    df_val = df_val.assign(weight__ = lambda d: d[sample_weight] if sample_weight is not None else 1)

    all_logs = []
    for i in range(max_iter):

        #set all features
        all_features = candidate_features_for_removal + fixed_features

        if len(all_features) == 0:
            break

        run_logs = _backwards_shap_feature_selection(
            model=clone(model),
            df_train=df_train,
            df_val=df_val,
            all_features=all_features,
            target=target,
            sample_weight="weight__",
            metric_fn=metric_fn,
            n_samples=n_samples,
            ci_level=ci_level,
            return_estimator=return_estimators,
            return_explainer=return_explainers
        )

        features_to_remove = (
            run_logs
                .sort_values(by = "ci_lower", ascending=False)
            [lambda d: d["ci_lower"] > 0]
            [lambda d: ~d["model"].isin(fixed_features + ["baseline_prediction"])]
        )

        if max_removals_per_run is not None:
            features_to_remove = features_to_remove.iloc[:max_removals_per_run]

        features_to_remove = features_to_remove["model"].values.tolist() #model means the model without the feature

        run_logs["run_index"] = i
        run_logs["n_features"] = len(run_logs) - 1
        run_logs["removed_features"] = str(features_to_remove)
        run_logs["n_features_removed"] = len(features_to_remove)
        all_logs.append(run_logs)

        if len(features_to_remove) == 0:
            break
        #update features for the next iteration
        candidate_features_for_removal = [i for i in candidate_features_for_removal if not i in features_to_remove]
        i+=1

    return pd.concat(all_logs, ignore_index=True)


def _backwards_shap_feature_selection(
        model,
        df_train,
        df_val,
        all_features,
        target,
        metric_fn,
        n_samples,
        ci_level,
        return_estimator,
        return_explainer,
):

    #train model
    model.fit(df_train[all_features], df_train[target], sample_weight=df_train["weight__"])

    #calculate shap
    explainer = shap.TreeExplainer(model)
    shap_values_val = explainer.shap_values(df_val[all_features])[-1]

    #make raw preds
    raw_preds_val = model.predict(df_val[all_features], raw_score=True)

    #score without feature
    scores_df = pd.DataFrame(
        raw_preds_val.reshape(-1,1) - shap_values_val,
        columns = all_features
    )

    #add extra columns
    scores_df["baseline_prediction"] = raw_preds_val
    scores_df = scores_df.apply(log_odds_to_proba)
    scores_df["weight__"] = df_val["weight__"].values
    scores_df[target] = df_val[target].values

    #deltas
    error_contributions_with_ci = fast_delta_metric_with_ci(
        scores_df,
        baseline="baseline_prediction",
        challengers=all_features,
        n_samples=n_samples,
        ci_level=ci_level,
        target=target,
        weight="weight__",
        metric_fn = metric_fn
    )

    # error_contributions_with_ci["estimate"] = -error_contributions_with_ci["estimate"]
    # error_contributions_with_ci["ci_lower"] = -error_contributions_with_ci["ci_lower"]
    # error_contributions_with_ci["ci_upper"] = -error_contributions_with_ci["ci_upper"]

    #current setup auc
    auc = fast_metric_with_ci(
        scores_df,
        predictions=["baseline_prediction"],
        n_samples=n_samples,
        ci_level=ci_level,
        target=target,
        weight="weight__",
        metric_fn = metric_fn
    )

    log = error_contributions_with_ci.append(auc, ignore_index = True)

    if return_estimator:
        log["estimator"] = model
    if return_explainer:
        log["explainer"] = explainer

    return log