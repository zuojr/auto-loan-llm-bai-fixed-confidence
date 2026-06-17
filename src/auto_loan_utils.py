from __future__ import annotations
import json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

from .config import (
    NUM_FEATURES, CAT_FEATURES, FEATURES, TARGET, OFFER_COL, OFFER_RATIO_COL,
    AMOUNT_COL, MULTIPLIERS, RANDOM_SEED, CONTEXT_COLUMNS,
)


def load_chronological_split(csv_path: str | Path, train_n=50_000, eval_n=40_000):
    df = pd.read_csv(csv_path)
    train = df.iloc[:train_n].copy().reset_index(drop=True)
    eval_pool = df.iloc[train_n:train_n + eval_n].copy().reset_index(drop=True)
    train['base_row_id'] = np.arange(len(train))
    eval_pool['base_row_id'] = np.arange(len(eval_pool))
    return df, train, eval_pool


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), NUM_FEATURES),
            ('cat', Pipeline([
                ('imp', SimpleImputer(strategy='most_frequent')),
                ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]), CAT_FEATURES),
        ],
        sparse_threshold=0.0,
    )


def make_xgb_model(seed=RANDOM_SEED):
    return XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.025,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric='logloss',
        random_state=seed,
        n_jobs=4,
    )


def train_environment_acceptance_model(train: pd.DataFrame, seed=RANDOM_SEED):
    pipe = Pipeline([('pre', make_preprocessor()), ('model', make_xgb_model(seed))])
    pipe.fit(train[FEATURES], train[TARGET])
    return pipe


def predict_acceptance(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURES])[:, 1]


def apply_multiplier(df: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    cf = df.copy()
    cf[OFFER_COL] = cf[OFFER_COL].astype(float) * multiplier
    cf[OFFER_RATIO_COL] = cf[OFFER_COL] / cf[AMOUNT_COL].clip(lower=1)
    return cf


def _fmt_money(x) -> str:
    try:
        if x is None or pd.isna(x):
            return "NA"
        return f"${float(x):,.2f}"
    except Exception:
        return "NA"


def _fmt_num(x, ndigits=2) -> str:
    try:
        if x is None or pd.isna(x):
            return "NA"
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return "NA"


def make_profile_text(row) -> str:
    """Natural-language application-offer description for LLM scoring.

    The goal is to let an API LLM use semantic knowledge about the auto-loan
    setting, not just parse terse column names.  The fields are all available
    before the counterfactual reward is simulated in the replay.
    """
    amount = float(row.get('Amount_Approved', 0) or 0)
    offer = float(row.get('offer_mp', row.get('mp', 0)) or 0)
    offer_ratio = offer / max(amount, 1.0)
    fico = int(row.Primary_FICO) if pd.notna(row.get('Primary_FICO', np.nan)) else 'NA'
    tier = int(row.Tier) if pd.notna(row.get('Tier', np.nan)) else 'NA'
    term = int(row.Term) if pd.notna(row.get('Term', np.nan)) else 'NA'
    partnerbin = int(row.partnerbin) if pd.notna(row.get('partnerbin', np.nan)) else 'NA'
    comp = _fmt_num(row.get('Competition_rate', np.nan), 2)
    multiplier = _fmt_num(row.get('multiplier', 1.0), 2)
    policy = row.get('policy', 'NA')
    return (
        f"This auto-loan application has Primary FICO {fico}, internal tier {tier}, "
        f"state {row.get('State','NA')}, applicant type {row.get('Type','NA')}, "
        f"car type {row.get('CarType','NA')}, approved amount {_fmt_money(amount)}, "
        f"and loan term {term} months. The competitive market rate is {comp}. "
        f"The offered margin price is {_fmt_money(offer)}, equal to {offer_ratio:.4%} "
        f"of the approved amount. The partnerbin indicator is {partnerbin}. "
        f"This row is evaluated under policy rule {policy} with multiplier {multiplier} "
        f"relative to the historical offer."
    )


def make_calibration_examples_text(examples: pd.DataFrame, max_rows: int | None = None) -> str:
    """Convert labeled historical examples into a compact natural-language block.

    If `max_rows` is provided and labels are available, select a roughly balanced
    subset rather than blindly taking the first rows.  This keeps the ICL prompt
    informative while the global base-rate paragraph prevents the model from
    treating the example balance as the population prior.
    """
    if max_rows is not None and (TARGET in examples.columns or 'apply' in examples.columns):
        lab_col = TARGET if TARGET in examples.columns else 'apply'
        n_pos = max_rows // 2
        n_neg = max_rows - n_pos
        pos = examples[examples[lab_col] == 1].head(n_pos)
        neg = examples[examples[lab_col] == 0].head(n_neg)
        examples = pd.concat([pos, neg], ignore_index=True)
        # Interleave positives and negatives when possible.
        if len(pos) and len(neg):
            rows = []
            for k in range(max(len(pos), len(neg))):
                if k < len(pos): rows.append(pos.iloc[[k]])
                if k < len(neg): rows.append(neg.iloc[[k]])
            examples = pd.concat(rows, ignore_index=True).head(max_rows)
        else:
            examples = examples.head(max_rows)
    elif max_rows is not None:
        examples = examples.head(max_rows)
    lines = []
    for k, (_, r) in enumerate(examples.iterrows(), start=1):
        row = r.copy()
        if 'offer_mp' not in row:
            row['offer_mp'] = row.get('mp', np.nan)
        if 'offer_ratio' not in row:
            row['offer_ratio'] = row.get('mp_rto_amtfinance', np.nan)
        row['multiplier'] = row.get('multiplier', 1.0)
        row['policy'] = row.get('policy', 'historical_rule')
        desc = make_profile_text(row)
        label = int(r.get(TARGET, r.get('apply', 0)))
        lines.append(f"Example {k}: apply={label}. {desc}")
    return "\n".join(lines)

def make_api_user_batch(rows: pd.DataFrame, global_stats: dict, examples_text: str) -> str:
    """Shared detailed prompt for GPT API and Qwen API ICL scoring."""
    compact = []
    for _, r in rows.iterrows():
        compact.append({
            'record_id': str(r['record_id']),
            'policy_rule': str(r.get('policy', 'NA')),
            'multiplier_relative_to_historical_offer': round(float(r.get('multiplier', 1.0)), 3),
            'application_offer_description': str(r['profile_text']),
        })
    return (
        "We are running an auto-loan pricing experiment.\n\n"
        "A lender observes a loan application and offers a margin price. The customer may accept "
        "the offered contract or not accept it. The observed binary outcome is apply=1 if the "
        "customer accepts/applies under the offered contract and apply=0 otherwise.\n\n"
        "Your task is to estimate the acceptance probability for each application-offer pair:\n"
        "pred_prob = P(apply = 1 | application profile, offered contract).\n\n"
        "Downstream use: this probability will be converted into an auxiliary proxy reward, "
        "proxy = offered margin price × pred_prob. The bandit algorithm will still use realized "
        "margin revenue as the true reward; your prediction is not used as a reward or as a direct "
        "policy ranking signal.\n\n"
        "Calibration discipline:\n"
        "- The labeled examples below may be class-balanced for demonstration, so their positive "
        "rate is not the true base rate.\n"
        "- Use the global historical apply rate as the prior base rate, then adjust upward or "
        "downward based on the application profile and offered contract.\n"
        "- Do not output 0.50 merely because the examples are balanced.\n"
        "- Return calibrated probabilities, not binary decisions.\n\n"
        "Global historical calibration stats:\n" + json.dumps(global_stats, ensure_ascii=False, indent=2) + "\n\n"
        "Feature interpretation:\n"
        "- Primary_FICO is a credit-quality signal, but this is an acceptance task rather than a default-risk task.\n"
        "- Tier is an internal segmentation variable.\n"
        "- Amount_Approved is the approved loan amount and Term is the loan term in months.\n"
        "- Competition_rate summarizes the competitive lending environment.\n"
        "- offer_mp is the offered margin price; offer_ratio is offer_mp divided by approved amount.\n"
        "- A higher offer may increase revenue if accepted, but it can reduce acceptance probability.\n"
        "- State, Type, CarType, and partnerbin may shift customer composition and acceptance behavior.\n"
        "- Avoid extreme probabilities unless the row is clearly much stronger or weaker than typical historical applications.\n\n"
        "Representative labeled historical examples:\n" + examples_text + "\n\n"
        "Now score the following unlabeled application-offer rows.\n"
        "For each row, return record_id copied exactly and pred_prob between 0.01 and 0.99.\n\n"
        "Rows:\n" + json.dumps(compact, ensure_ascii=False, indent=2) + "\n\n"
        "Output format: return only a JSON array. Each object must have exactly these keys:\n"
        "{\"record_id\": \"...\", \"pred_prob\": 0.123}\n"
        "Do not include explanations, markdown, comments, or any extra text. Preserve every record_id."
    )

def build_two_policy_pool(model, eval_pool: pd.DataFrame, multipliers=MULTIPLIERS) -> pd.DataFrame:
    base = eval_pool.copy().reset_index(drop=True)
    hist_prob = predict_acceptance(model, base)
    hist_offer = base[OFFER_COL].to_numpy(float)
    hist = pd.DataFrame({
        'base_row_id': base['base_row_id'].to_numpy(),
        'arm': 0,
        'policy': 'historical_rule',
        'multiplier': 1.0,
        'offer_mp': hist_offer,
        'offer_ratio': hist_offer / np.clip(base[AMOUNT_COL].to_numpy(float), 1, None),
        'prob_accept': hist_prob,
        'observed_apply': base[TARGET].to_numpy(),
        'historical_offer_mp': hist_offer,
    })

    prob_mat, offer_mat, rev_mat = [], [], []
    for m in multipliers:
        cf = apply_multiplier(base, m)
        prob = predict_acceptance(model, cf)
        offer = cf[OFFER_COL].to_numpy(float)
        prob_mat.append(prob)
        offer_mat.append(offer)
        rev_mat.append(prob * offer)
    prob_mat = np.vstack(prob_mat).T
    offer_mat = np.vstack(offer_mat).T
    rev_mat = np.vstack(rev_mat).T
    best_idx = rev_mat.argmax(axis=1)
    mult_arr = np.array(multipliers)
    rows = np.arange(len(base))
    imp = pd.DataFrame({
        'base_row_id': base['base_row_id'].to_numpy(),
        'arm': 1,
        'policy': 'support_constrained_improved_rule',
        'multiplier': mult_arr[best_idx],
        'offer_mp': offer_mat[rows, best_idx],
        'offer_ratio': offer_mat[rows, best_idx] / np.clip(base[AMOUNT_COL].to_numpy(float), 1, None),
        'prob_accept': prob_mat[rows, best_idx],
        'observed_apply': base[TARGET].to_numpy(),
        'historical_offer_mp': hist_offer,
    })
    paired = pd.concat([hist, imp], ignore_index=True)
    context = base[['base_row_id'] + CONTEXT_COLUMNS].copy()
    paired = paired.merge(context, on='base_row_id', how='left')
    paired['expected_reward_env'] = paired['offer_mp'] * paired['prob_accept']
    paired['record_id'] = paired['policy'].astype(str) + '_' + paired['base_row_id'].astype(str)
    paired['profile_text'] = paired.apply(make_profile_text, axis=1)
    return paired


def policy_summary_from_environment(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, policy), g in paired.groupby(['arm','policy']):
        offer = g['offer_mp'].to_numpy(float)
        p = g['prob_accept'].to_numpy(float)
        mu = float(np.mean(offer * p))
        varx = float(np.mean((offer ** 2) * p) - mu ** 2)
        rows.append({
            'arm': arm,
            'policy': policy,
            'n_units': len(g),
            'mean_reward_env': mu,
            'mean_accept_prob_env': float(np.mean(p)),
            'sd_reward_env': float(np.sqrt(max(varx, 0))),
            'mean_offer_mp': float(np.mean(offer)),
        })
    return pd.DataFrame(rows)


def make_prompt_for_acceptance(row, include_policy=True) -> str:
    """Short prompt used for fine-tuned Qwen training/inference.

    API ICL prompts include calibration examples and global stats.  Fine-tuned
    Qwen should instead receive a stable task prompt and one application-offer
    profile, because the supervision is already in the LoRA training data.
    """
    policy_line = ''
    if include_policy:
        policy_line = (
            f"Pricing rule: {row.get('policy','NA')}; multiplier relative to historical offer: "
            f"{float(row.get('multiplier',1.0)):.2f}.\n"
        )
    return (
        "Estimate the probability that the customer accepts this offered auto-loan contract.\n\n"
        "Target:\n"
        "pred_prob = P(apply = 1 | application profile, offered contract).\n\n"
        "This is an acceptance-probability task, not a default-risk task and not a policy-ranking task.\n"
        "Return only JSON like {\"pred_prob\": 0.123}.\n\n"
        f"{policy_line}"
        f"Application-offer profile:\n{row['profile_text']}\n"
    )

def parse_probability(text: str, default=np.nan) -> float:
    if text is None:
        return default
    s = str(text).strip()
    # JSON-like pred_prob first
    m = re.search(r'pred[_ ]?prob["\'\s:]*([0-9]*\.?[0-9]+)', s, flags=re.I)
    if not m:
        m = re.search(r'([0-9]*\.?[0-9]+)', s)
    if not m:
        return default
    val = float(m.group(1))
    # Interpret percentages defensively.
    if val > 1 and val <= 100:
        val = val / 100.0
    return float(np.clip(val, 0.01, 0.99))
