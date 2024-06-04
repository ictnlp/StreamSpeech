from .latency_scorer import LATENCY_SCORERS_DICT
from .quality_scorer import QUALITY_SCORERS_DICT


def get_scorer_class(scorer_type, name):
    if scorer_type == "quality":
        scorer_dict = QUALITY_SCORERS_DICT
    else:
        scorer_dict = LATENCY_SCORERS_DICT

    if name not in scorer_dict:
        raise RuntimeError(f"No {scorer_type} metric called {name}")

    return scorer_dict[name]
