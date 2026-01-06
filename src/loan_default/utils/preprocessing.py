"""Preprocessing helpers to build leakage-safe train/test splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from .imputation import categorical_missing, numerical_missing


def _safe_probs(observed: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return (values, probabilities) ensuring a valid multinomial vector."""

    if observed.empty:
        return np.array([]), np.array([])
    vals = observed.unique()
    counts = observed.value_counts(normalize=True)
    probs = counts.reindex(vals).fillna(0).to_numpy()
    total = probs.sum()
    if total == 0:
        probs = np.full_like(probs, 1 / len(probs))
    else:
        probs = probs / total
    return vals, probs


def prepare_train_test(
    df_base: pd.DataFrame,
    test_size: float = 0.5,
    rn_state: int = 1818,
    rng_seed: int = 1818,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit imputations on train only, apply to test, and return engineered splits.

    Returns:
        train_with_dummies, test_with_dummies, train_without_dummies, test_without_dummies
    """

    rng = np.random.default_rng(rng_seed)

    # Initial split with stratification to keep class balance
    train_df, test_df = train_test_split(
        df_base.copy(), test_size=test_size, random_state=rn_state, stratify=df_base["Status"]
    )

    # Income (grouped by Status and loan_purpose)
    income_est = pd.crosstab(train_df["Status"], train_df["loan_purpose"], values=train_df["income"], aggfunc="median")
    train_0 = numerical_missing(train_df, income_est, "income", ["Status", "loan_purpose"])
    test_0 = numerical_missing(test_df, income_est, "income", ["Status", "loan_purpose"])

    # Categorical multinomial draws (fit probs on train)
    def apply_categorical(train_in: pd.DataFrame, test_in: pd.DataFrame, col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        vals, probs = _safe_probs(train_in[col].dropna())
        if len(vals) == 0:
            return train_in, test_in
        train_out = categorical_missing(train_in, col, probabilities=probs, values=vals, rng=rng)
        test_out = categorical_missing(test_in, col, probabilities=probs, values=vals, rng=rng)
        return train_out, test_out

    train_1, test_1 = train_0.copy(), test_0.copy()
    for col in ["age", "submission_of_application", "approv_in_adv", "loan_purpose", "loan_limit", "Neg_ammortization"]:
        train_1, test_1 = apply_categorical(train_1, test_1, col)

    # Term imputation using train distribution
    term_notna = train_1[train_1["term"].notna()]
    if len(term_notna) == 0:
        term_probs = np.array([0.0, 0.0, 1.0])
    else:
        N = len(term_notna)
        p1 = (term_notna["term"] <= 180).sum() / N
        p2 = ((term_notna["term"] <= 240) & (term_notna["term"] > 180)).sum() / N
        p3 = (term_notna["term"] > 240).sum() / N
        term_probs = np.array([p1, p2, p3])
        total = term_probs.sum()
        term_probs = term_probs / total if total > 0 else np.array([0.0, 0.0, 1.0])
    term_vals = np.array([180, 240, 360])
    train_1 = categorical_missing(train_1, "term", probabilities=term_probs, values=term_vals, rng=rng)
    test_1 = categorical_missing(test_1, "term", probabilities=term_probs, values=term_vals, rng=rng)

    # DTIR imputation (median by Status from train with global fallback)
    ditr_est = train_1.groupby(["Status"])["dtir1"].median()
    global_dtir = train_1["dtir1"].median()
    ditr_est = ditr_est.fillna(global_dtir)
    train_2 = numerical_missing(train_1, ditr_est, "dtir1", ["Status"])
    test_2 = numerical_missing(test_1, ditr_est, "dtir1", ["Status"])

    # Upfront charges coin flip based on train stats
    def apply_upfront(train_in: pd.DataFrame, test_in: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        where_missing_train = np.where(train_in["Upfront_charges"].isna())[0]
        where_missing_test = np.where(test_in["Upfront_charges"].isna())[0]

        non_missing = train_in[train_in["Upfront_charges"].notna()]
        if len(non_missing) == 0:
            pul_vec = np.array([0.5, 0.5])
            mul = 0.0
        else:
            pul = len(non_missing[non_missing["Upfront_charges"] == 0]) / len(non_missing)
            pul_vec = np.array([pul, 1 - pul])
            positive = non_missing[non_missing["Upfront_charges"] > 0]
            if len(positive) == 0:
                mul = 0.0
            else:
                mul = (positive["Upfront_charges"] * 100 / positive["loan_amount"]).mean()
        if pul_vec.sum() == 0:
            pul_vec = np.array([0.5, 0.5])
        vul = np.array([0, mul])

        def fill_missing(frame: pd.DataFrame, missing_idx: np.ndarray) -> pd.DataFrame:
            if len(missing_idx) == 0:
                return frame
            draws = rng.multinomial(1, pul_vec, len(missing_idx))
            sampled = [vul[row.argmax()] for row in draws]
            col_pos = frame.columns.get_loc("Upfront_charges")
            frame.iloc[missing_idx, col_pos] = sampled
            return frame

        return fill_missing(train_in, where_missing_train), fill_missing(test_in, where_missing_test)

    train_3, test_3 = apply_upfront(train_2.copy(), test_2.copy())

    # Interest rates mean impute (fit on train)
    rate_mean = train_3["rate_of_interest"].mean()
    irs_mean = train_3["Interest_rate_spread"].mean()
    train_4 = train_3.copy()
    test_4 = test_3.copy()
    train_4["rate_of_interest"] = train_4["rate_of_interest"].fillna(rate_mean)
    test_4["rate_of_interest"] = test_4["rate_of_interest"].fillna(rate_mean)
    train_4["Interest_rate_spread"] = train_4["Interest_rate_spread"].fillna(irs_mean)
    test_4["Interest_rate_spread"] = test_4["Interest_rate_spread"].fillna(irs_mean)

    # Feature engineering
    def engineer(df_in: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_engineered_0 = df_in.copy()
        df_engineered_0["term"] = pd.cut(df_engineered_0["term"], bins=[0, 180, 240, 360], labels=["short", "medium", "long"], include_lowest=True)

        df_engineered_1 = df_engineered_0.copy()
        df_engineered_1["Upfront_charges_y/n"] = np.where(df_engineered_1["Upfront_charges"] < 300, 0, 1)
        df_engineered_1["Upfront_charges"] = df_engineered_1["Upfront_charges"] * 100 / df_engineered_1["loan_amount"]

        df_engineered_2 = df_engineered_1.copy()
        df_engineered_2["income"] = np.log10(df_engineered_2["income"].clip(lower=1e-6))
        df_engineered_2["property_value"] = np.log10(df_engineered_2["property_value"].clip(lower=1e-6))
        df_engineered_2["loan_amount"] = np.log10(df_engineered_2["loan_amount"].clip(lower=1e-6))
        
        df_engineered_3 = df_engineered_2.drop(["ID", "year", "LTV", "credit_type", "Secured_by"], axis=1)

        df_engineered_4 = df_engineered_3.copy()
        target_mean_age = df_engineered_4.groupby("age")["Status"].mean()
        df_engineered_4["age"] = df_engineered_4["age"].map(target_mean_age)
        target_mean_region = df_engineered_4.groupby("Region")["Status"].mean()
        df_engineered_4["Region"] = df_engineered_4["Region"].map(target_mean_region)

        encoder = OrdinalEncoder()
        df_engineered_4[["term", "total_units"]] = encoder.fit_transform(df_engineered_4[["term", "total_units"]]) + 1

        cardinal_cols = df_engineered_4.select_dtypes(include="object").columns
        if len(cardinal_cols) > 0:
            column_transforms = ColumnTransformer(
                transformers=[("onehot", OneHotEncoder(drop="first", sparse_output=False), cardinal_cols)],
                remainder="drop",
            )
            encoded = column_transforms.fit_transform(df_engineered_4[cardinal_cols])
            onehot_feature_names = column_transforms.named_transformers_["onehot"].get_feature_names_out(cardinal_cols)
            df_tmp = pd.DataFrame(encoded, columns=onehot_feature_names, index=df_engineered_4.index)
        else:
            df_tmp = pd.DataFrame(index=df_engineered_4.index)

        df_engineered_4 = df_engineered_4.drop(cardinal_cols, axis=1)
        df_engineered_4 = pd.concat([df_engineered_4, df_tmp], axis=1)
        df_engineered_4[["Credit_Score", "Status", "Upfront_charges_y/n"]] = df_engineered_4[["Credit_Score", "Status", "Upfront_charges_y/n"]].astype("float64")

        return df_engineered_3, df_engineered_4

    train_without_dummies, train_with_dummies = engineer(train_4)
    test_without_dummies, test_with_dummies = engineer(test_4)

    return train_with_dummies, test_with_dummies, train_without_dummies, test_without_dummies
