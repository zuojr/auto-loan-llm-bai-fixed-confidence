TRAIN_N = 50_000
EVAL_N = 40_000
MULTIPLIERS = [0.8, 0.9, 1.0, 1.1, 1.2]
RANDOM_SEED = 20260610
TARGET = 'apply'
OFFER_COL = 'mp'
OFFER_RATIO_COL = 'mp_rto_amtfinance'
AMOUNT_COL = 'Amount_Approved'
NUM_FEATURES = [
    'Tier', 'Primary_FICO', 'Term', 'Amount_Approved', 'Competition_rate',
    'mp', 'mp_rto_amtfinance', 'partnerbin', 'CarType_id',
    'days', 'weeks', 'months', 'termclass',
]
CAT_FEATURES = ['State', 'Type', 'CarType']
FEATURES = NUM_FEATURES + CAT_FEATURES
CONTEXT_COLUMNS = [
    'Tier', 'Primary_FICO', 'State', 'Type', 'Term', 'Amount_Approved',
    'Competition_rate', 'CarType', 'partnerbin', 'CarType_id',
    'days', 'weeks', 'months', 'termclass'
]
