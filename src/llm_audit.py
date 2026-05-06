import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score

from .config import Config, POLICY_ORDER
from .utils import maybe_float, sample_users, stable_hash, write_jsonl
def default_llm_registry_entry(model_name: str) -> Optional[Dict[str, Any]]:
    defaults = {
        "openai_gpt4o": {
            "api_type": "openai_chat_completions",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "model": "gpt-4o-mini",
        }
    }
    return defaults.get(model_name)


def load_llm_registry(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("llm_registry_path must contain a JSON object.")
    return data


def resolve_llm_spec(cfg: Config) -> Optional[Dict[str, Any]]:
    registry = load_llm_registry(cfg.llm_registry_path)
    spec = registry.get(cfg.llm_model_name)
    if spec is None:
        spec = default_llm_registry_entry(cfg.llm_model_name)
    return spec


def load_llm_cache(path: str) -> Dict[str, Dict[str, Any]]:
    cache = {}
    if not os.path.exists(path):
        return cache
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "key" in row:
                    cache[row["key"]] = row
            except Exception:
                continue
    return cache


def append_llm_cache_row(path: str, row: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_text_from_chat_response(data: Dict[str, Any]) -> str:
    choices = data.get("choices", [])
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and "text" in part:
                    texts.append(part["text"])
        return "\n".join(texts)
    return str(content)


def extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def llm_chat_json(
    cfg: Config,
    llm_spec: Dict[str, Any],
    cache: Dict[str, Dict[str, Any]],
    messages: List[Dict[str, str]],
    max_tokens: int,
):
    key_payload = {
        "llm_model_name": cfg.llm_model_name,
        "resolved_model": llm_spec.get("model"),
        "temperature": cfg.llm_temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    key = stable_hash(key_payload)
    if key in cache:
        row = cache[key]
        return row.get("text", ""), row.get("parsed")

    api_type = llm_spec.get("api_type", "openai_chat_completions")
    if api_type != "openai_chat_completions":
        raise ValueError(f"Unsupported api_type: {api_type}")

    api_key_env = llm_spec.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing environment variable for LLM API key: {api_key_env}")

    base_url = llm_spec.get("base_url", "https://api.openai.com/v1").rstrip("/")
    model = llm_spec["model"]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": cfg.llm_temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    extra_body = llm_spec.get("extra_body")
    if isinstance(extra_body, dict):
        payload.update(extra_body)

    resp = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=cfg.llm_timeout_sec,
    )
    resp.raise_for_status()
    data = resp.json()
    text = extract_text_from_chat_response(data)
    parsed = extract_first_json_obj(text)

    row = {
        "key": key,
        "text": text,
        "parsed": parsed,
        "created_at": time.time(),
        "llm_model_name": cfg.llm_model_name,
        "resolved_model": model,
    }
    cache[key] = row
    append_llm_cache_row(cfg.llm_cache_path, row)
    return text, parsed
def build_randomized_state_prompt(cfg: Config, rng: np.random.RandomState) -> Dict[str, Any]:
    if rng.rand() < 0.5:
        label_to_state = {"A": 0, "B": 1}
        desc_A = cfg.state_desc_0
        desc_B = cfg.state_desc_1
    else:
        label_to_state = {"A": 1, "B": 0}
        desc_A = cfg.state_desc_1
        desc_B = cfg.state_desc_0

    state_to_label = {v: k for k, v in label_to_state.items()}

    return {
        "label_to_state": label_to_state,
        "state_to_label": state_to_label,
        "desc_A": desc_A,
        "desc_B": desc_B,
    }

def format_recommendation_list(
    topk_items: np.ndarray,
    titles: List[str],
    item_genre_strings: List[str],
    max_items: int,
) -> str:
    lines = []
    use_items = topk_items[:max_items]
    for rank, i in enumerate(use_items, start=1):
        lines.append(f"{rank}. {titles[int(i)]} [{item_genre_strings[int(i)]}]")
    return "\n".join(lines)


def build_llm_single_messages(
    cfg: Config,
    rec_list_text: str,
    state_prompt: Dict[str, Any],
    persona: Optional[str] = None,
) -> List[Dict[str, str]]:
    persona_text = ""
    if persona is not None:
        persona_text = f"You should answer from the perspective of this evaluator persona: {persona}.\n\n"

    return [
        {
            "role": "system",
            "content": (
                "You are a careful privacy auditor. "
                "Given one recommendation list, infer which of two hidden short-term intents is more likely. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{persona_text}"
                "There are two possible hidden short-term intents.\n\n"
                f"State A: {state_prompt['desc_A']}\n"
                f"State B: {state_prompt['desc_B']}\n\n"
                "Observed recommendation list:\n"
                f"{rec_list_text}\n\n"
                "Return JSON with this schema exactly:\n"
                '{"guess": "A" or "B", "confidence": 0.0 to 1.0, "brief_reason": "short"}'
            ),
        },
    ]


def build_llm_pair_messages(
    cfg: Config,
    list_a_text: str,
    list_b_text: str,
    state_prompt: Dict[str, Any],
    target_state: int = 1,
    persona: Optional[str] = None,
) -> List[Dict[str, str]]:
    persona_text = ""
    if persona is not None:
        persona_text = f"You should answer from the perspective of this evaluator persona: {persona}.\n\n"

    target_label = state_prompt["state_to_label"][int(target_state)]

    return [
        {
            "role": "system",
            "content": (
                "You are a careful privacy auditor. "
                "You will compare two recommendation lists for the same user. "
                "Infer which list is more likely to correspond to the target hidden state. "
                "If the two lists are indistinguishable, you may say so. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{persona_text}"
                "There are two possible hidden short-term intents.\n\n"
                f"State A: {state_prompt['desc_A']}\n"
                f"State B: {state_prompt['desc_B']}\n\n"
                "Recommendation list A:\n"
                f"{list_a_text}\n\n"
                "Recommendation list B:\n"
                f"{list_b_text}\n\n"
                f"Which list is more likely to have been generated under State {target_label}?\n"
                'Return JSON with this schema exactly:\n'
                '{"guess": "A" or "B" or "indistinguishable", "confidence": 0.0 to 1.0, "brief_reason": "short"}'
            ),
        },
    ]
def build_llm_rating_messages(
    cfg: Config,
    rec_list_text: str,
    state_prompt: Dict[str, Any],
    persona: Optional[str] = None,
) -> List[Dict[str, str]]:
    persona_text = ""
    if persona is not None:
        persona_text = f"You should answer from the perspective of this evaluator persona: {persona}.\n\n"

    return [
        {
            "role": "system",
            "content": (
                "You are a careful privacy auditor. "
                "You will read one recommendation list and rate semantic privacy-related perceptions. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{persona_text}"
                "There are two possible hidden short-term intents.\n\n"
                f"State A: {state_prompt['desc_A']}\n"
                f"State B: {state_prompt['desc_B']}\n\n"
                "Observed recommendation list:\n"
                f"{rec_list_text}\n\n"
                "Using integer ratings from 1 to 7, where 1 means not at all and 7 means very strongly, "
                "rate the following:\n"
                "1. The list reveals the user's private state.\n"
                "2. The list makes the user feel monitored.\n"
                "3. The list feels overly personalized to a sensitive condition.\n"
                "4. The list feels privacy-invasive.\n\n"
                'Return JSON with this exact schema:\n'
                '{"state_reveal": 1-7, "monitoring": 1-7, "over_personalized": 1-7, "privacy_invasive": 1-7, "brief_reason": "short"}'
            ),
        },
    ]

def normalize_llm_ratings(parsed: Optional[Dict[str, Any]], raw_text: str) -> Dict[str, Optional[float]]:
    out = {
        "state_reveal": np.nan,
        "monitoring": np.nan,
        "over_personalized": np.nan,
        "privacy_invasive": np.nan,
    }

    key_aliases = {
        "state_reveal": ["state_reveal", "reveal", "state_reveal_score", "private_state_reveal"],
        "monitoring": ["monitoring", "monitored", "feels_monitored", "monitoring_score"],
        "over_personalized": ["over_personalized", "overpersonalized", "sensitive_personalization"],
        "privacy_invasive": ["privacy_invasive", "privacyinvasive", "invasive", "privacy_invasiveness"],
    }

    def _coerce_1to7(v):
        if isinstance(v, (int, float)):
            x = float(v)
            if 1.0 <= x <= 7.0:
                return x
            return np.nan
        if isinstance(v, str):
            m = re.search(r'([1-7])(?:\.0+)?', v.strip())
            if m:
                return float(m.group(1))
        return np.nan

    if isinstance(parsed, dict):
        for target_key, aliases in key_aliases.items():
            for a in aliases:
                if a in parsed:
                    out[target_key] = _coerce_1to7(parsed[a])
                    if not pd.isna(out[target_key]):
                        break

    if any(pd.isna(v) for v in out.values()) and isinstance(raw_text, str):
        patterns = {
            "state_reveal": r'(?i)(state[_\s-]*reveal|reveals? the user.?s private state)[^0-9]*([1-7])',
            "monitoring": r'(?i)(monitoring|feel monitored)[^0-9]*([1-7])',
            "over_personalized": r'(?i)(over[_\s-]*personalized|overly personalized)[^0-9]*([1-7])',
            "privacy_invasive": r'(?i)(privacy[_\s-]*invasive|privacy invasive|invasive)[^0-9]*([1-7])',
        }

        for key, pat in patterns.items():
            if pd.isna(out[key]):
                m = re.search(pat, raw_text)
                if m:
                    try:
                        out[key] = float(m.group(2))
                    except Exception:
                        pass

    return out
def normalize_single_guess(
    parsed: Optional[Dict[str, Any]],
    raw_text: str,
    label_map: Optional[Dict[str, int]] = None,
) -> Optional[int]:
    vals = []
    if isinstance(parsed, dict):
        for k in ["guess", "label", "state", "answer", "prediction"]:
            if k in parsed:
                vals.append(parsed[k])
    vals.append(raw_text)

    for v in vals:
        if isinstance(v, (int, float)):
            vv = int(v)
            if vv in (0, 1):
                return vv

        if isinstance(v, str):
            s = v.strip().lower()
            s_norm = s.replace(" ", "").replace("_", "").replace("-", "")

            if label_map is not None:
                if s_norm in {"a", "statea", "intenta", "labela"}:
                    return int(label_map["A"])
                if s_norm in {"b", "stateb", "intentb", "labelb"}:
                    return int(label_map["B"])
                if "state a" in s or "label a" in s or "intent a" in s:
                    return int(label_map["A"])
                if "state b" in s or "label b" in s or "intent b" in s:
                    return int(label_map["B"])

            if s_norm in {"0", "state0", "intent0", "label0"}:
                return 0
            if s_norm in {"1", "state1", "intent1", "label1"}:
                return 1
            if "romance" in s_norm or "drama" in s_norm:
                return 0
            if "action" in s_norm or "thriller" in s_norm:
                return 1

    return None


def normalize_pair_guess(parsed: Optional[Dict[str, Any]], raw_text: str) -> Optional[str]:
    vals = []
    if isinstance(parsed, dict):
        for k in ["guess", "label", "answer", "prediction", "list"]:
            if k in parsed:
                vals.append(parsed[k])
    vals.append(raw_text)

    for v in vals:
        if isinstance(v, str):
            s = v.strip()
            s_upper = s.upper()
            s_norm = s.strip().lower().replace(" ", "").replace("_", "").replace("-", "")

            if s_upper in {"A", "B"}:
                return s_upper
            if s_upper.startswith("A"):
                return "A"
            if s_upper.startswith("B"):
                return "B"

            if s_norm in {
                "indistinguishable", "cannottell", "canttell", "uncertain",
                "unknown", "same", "tie", "equal", "both", "neither"
            }:
                return "INDISTINGUISHABLE"

            if "indistinguishable" in s.lower():
                return "INDISTINGUISHABLE"
            if "cannot tell" in s.lower() or "can't tell" in s.lower():
                return "INDISTINGUISHABLE"

    return None


def normalize_single_score(
    parsed: Optional[Dict[str, Any]],
    raw_text: str,
    label_map: Optional[Dict[str, int]] = None,
) -> float:
    pred = normalize_single_guess(parsed, raw_text, label_map=label_map)

    conf = None
    candidates = []

    if isinstance(parsed, dict):
        for k in ["confidence", "conf", "prob", "probability", "score"]:
            if k in parsed:
                candidates.append(parsed[k])

    candidates.append(raw_text)

    for v in candidates:
        if isinstance(v, (int, float)):
            conf = float(v)
            break

        if isinstance(v, str):
            m = re.search(r'(?i)(confidence|prob|probability|score)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)', v)
            if m:
                try:
                    conf = float(m.group(2))
                    break
                except Exception:
                    pass

            m2 = re.search(r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)', v)
            if m2:
                try:
                    conf = float(m2.group(1))
                    break
                except Exception:
                    pass

    if conf is None:
        conf = 0.5

    if conf > 1.0 and conf <= 100.0:
        conf = conf / 100.0
    conf = float(np.clip(conf, 0.0, 1.0))

    if pred is None:
        return 0.5
    if pred == 1:
        return conf
    return 1.0 - conf

def compute_llm_single_auc(raw_rows: List[Dict[str, Any]]) -> float:
    ys = []
    scores = []

    for r in raw_rows:
        if r.get("audit_type") != "single":
            continue

        if r.get("pred_label") is None:
            continue

        y = r.get("true_label")
        s = r.get("pred_score_z1")

        if y is None or s is None:
            continue

        try:
            y = int(y)
            s = float(s)
        except Exception:
            continue

        if y not in (0, 1):
            continue

        ys.append(y)
        scores.append(s)

    if len(ys) == 0:
        return np.nan
    if len(set(ys)) < 2:
        return np.nan

    try:
        return float(roc_auc_score(ys, scores))
    except Exception:
        return np.nan
def run_llm_audit_policy(
    cfg: Config,
    llm_spec: Dict[str, Any],
    cache: Dict[str, Dict[str, Any]],
    data_dict,
    users: List[int],
    topk_z0: Dict[int, np.ndarray],
    topk_z1: Dict[int, np.ndarray],
    seed: int,
    policy_name: str,
    persona: Optional[str] = None,
    mc_tag: Optional[int] = None,
):
    titles = data_dict["titles"]
    item_genre_strings = data_dict["item_genre_strings"]

    rng = np.random.RandomState(seed)
    raw_rows = []

    single_correct = 0
    single_total = 0

    for u in users:
        for true_z, topk_items in [(0, topk_z0[u]), (1, topk_z1[u])]:
            state_prompt = build_randomized_state_prompt(cfg, rng)

            rec_text = format_recommendation_list(
                topk_items=topk_items,
                titles=titles,
                item_genre_strings=item_genre_strings,
                max_items=cfg.llm_list_k,
            )
            messages = build_llm_single_messages(
                cfg, rec_text, state_prompt=state_prompt, persona=persona
            )

            try:
                raw_text, parsed = llm_chat_json(
                    cfg, llm_spec, cache, messages, cfg.llm_max_tokens_single
                )
                pred = normalize_single_guess(
                    parsed, raw_text, label_map=state_prompt["label_to_state"]
                )
                pred_score_z1 = normalize_single_score(
                    parsed, raw_text, label_map=state_prompt["label_to_state"]
                )

                if pred in (0, 1):
                    confidence_used = float(pred_score_z1) if pred == 1 else float(1.0 - pred_score_z1)
                else:
                    confidence_used = 0.5

            except Exception as e:
                raw_text = f"ERROR: {repr(e)}"
                parsed = None
                pred = None
                pred_score_z1 = None
                confidence_used = None

            ok = pred is not None
            if ok:
                single_total += 1
                single_correct += int(pred == true_z)

            raw_rows.append(
                {
                    "policy": policy_name,
                    "mc_tag": mc_tag,
                    "persona": persona,
                    "audit_type": "single",
                    "user": int(u),
                    "true_label": int(true_z),
                    "prompt_label_A_state": int(state_prompt["label_to_state"]["A"]),
                    "prompt_label_B_state": int(state_prompt["label_to_state"]["B"]),
                    "pred_label": None if pred is None else int(pred),
                    "pred_score_z1": float(pred_score_z1),
                    "confidence_used": float(confidence_used),
                    "correct": None if pred is None else int(pred == true_z),
                    "raw_text": raw_text,
                    "parsed": parsed,
                }
            )

    pair_credit_sum = 0.0
    pair_total = 0

    for u in users:
        t0 = topk_z0[u]
        t1 = topk_z1[u]

        if rng.rand() < 0.5:
            list_a = t0
            list_b = t1
            true_guess = "B"
        else:
            list_a = t1
            list_b = t0
            true_guess = "A"

        state_prompt = build_randomized_state_prompt(cfg, rng)

        text_a = format_recommendation_list(
            topk_items=list_a,
            titles=titles,
            item_genre_strings=item_genre_strings,
            max_items=cfg.llm_list_k,
        )
        text_b = format_recommendation_list(
            topk_items=list_b,
            titles=titles,
            item_genre_strings=item_genre_strings,
            max_items=cfg.llm_list_k,
        )
        messages = build_llm_pair_messages(
            cfg,
            text_a,
            text_b,
            state_prompt=state_prompt,
            target_state=1,
            persona=persona,
        )

        try:
            raw_text, parsed = llm_chat_json(cfg, llm_spec, cache, messages, cfg.llm_max_tokens_pair)
            pred = normalize_pair_guess(parsed, raw_text)
        except Exception as e:
            raw_text = f"ERROR: {repr(e)}"
            parsed = None
            pred = None

        pair_credit = None
        if pred in {"A", "B"}:
            pair_total += 1
            pair_credit = float(pred == true_guess)
            pair_credit_sum += pair_credit
        elif pred == "INDISTINGUISHABLE":
            pair_total += 1
            pair_credit = 0.5
            pair_credit_sum += pair_credit

        raw_rows.append(
            {
                "policy": policy_name,
                "mc_tag": mc_tag,
                "persona": persona,
                "audit_type": "pair",
                "user": int(u),
                "true_label": true_guess,
                "pred_label": pred,
                "pair_credit": pair_credit,
                "correct": None if pair_credit is None else float(pair_credit),
                "raw_text": raw_text,
                "parsed": parsed,
            }
        )

    summary = {
        "policy": policy_name,
        "mc_tag": mc_tag,
        "persona": persona,
        "single_acc": float(single_correct / single_total) if single_total > 0 else np.nan,
        "single_auc": compute_llm_single_auc(raw_rows),
        "single_n": int(single_total),
        "pair_acc": float(pair_credit_sum / pair_total) if pair_total > 0 else np.nan,
        "pair_n": int(pair_total),
    }
    return summary, raw_rows
def run_llm_rating_audit_policy(
    cfg: Config,
    llm_spec: Dict[str, Any],
    cache: Dict[str, Dict[str, Any]],
    data_dict,
    users: List[int],
    topk_z0: Dict[int, np.ndarray],
    topk_z1: Dict[int, np.ndarray],
    seed: int,
    policy_name: str,
    persona: Optional[str] = None,
    mc_tag: Optional[int] = None,
):
    titles = data_dict["titles"]
    item_genre_strings = data_dict["item_genre_strings"]

    rng = np.random.RandomState(seed)
    raw_rows = []

    state_reveal_vals = []
    monitoring_vals = []
    over_personalized_vals = []
    privacy_invasive_vals = []

    for u in users:
        for true_z, topk_items in [(0, topk_z0[u]), (1, topk_z1[u])]:
            state_prompt = build_randomized_state_prompt(cfg, rng)

            rec_text = format_recommendation_list(
                topk_items=topk_items,
                titles=titles,
                item_genre_strings=item_genre_strings,
                max_items=cfg.llm_list_k,
            )
            messages = build_llm_rating_messages(
                cfg, rec_text, state_prompt=state_prompt, persona=persona
            )

            try:
                raw_text, parsed = llm_chat_json(
                    cfg, llm_spec, cache, messages, cfg.llm_max_tokens_single
                )
                ratings = normalize_llm_ratings(parsed, raw_text)
            except Exception as e:
                raw_text = f"ERROR: {repr(e)}"
                parsed = None
                ratings = {
                    "state_reveal": np.nan,
                    "monitoring": np.nan,
                    "over_personalized": np.nan,
                    "privacy_invasive": np.nan,
                }

            if not pd.isna(ratings["state_reveal"]):
                state_reveal_vals.append(float(ratings["state_reveal"]))
            if not pd.isna(ratings["monitoring"]):
                monitoring_vals.append(float(ratings["monitoring"]))
            if not pd.isna(ratings["over_personalized"]):
                over_personalized_vals.append(float(ratings["over_personalized"]))
            if not pd.isna(ratings["privacy_invasive"]):
                privacy_invasive_vals.append(float(ratings["privacy_invasive"]))

            raw_rows.append(
                {
                    "policy": policy_name,
                    "mc_tag": mc_tag,
                    "persona": persona,
                    "audit_type": "rating",
                    "user": int(u),
                    "true_label": int(true_z),
                    "prompt_label_A_state": int(state_prompt["label_to_state"]["A"]),
                    "prompt_label_B_state": int(state_prompt["label_to_state"]["B"]),
                    "state_reveal": None if pd.isna(ratings["state_reveal"]) else float(ratings["state_reveal"]),
                    "monitoring": None if pd.isna(ratings["monitoring"]) else float(ratings["monitoring"]),
                    "over_personalized": None if pd.isna(ratings["over_personalized"]) else float(ratings["over_personalized"]),
                    "privacy_invasive": None if pd.isna(ratings["privacy_invasive"]) else float(ratings["privacy_invasive"]),
                    "raw_text": raw_text,
                    "parsed": parsed,
                }
            )

    summary = {
        "policy": policy_name,
        "mc_tag": mc_tag,
        "persona": persona,
        "state_reveal_mean": float(np.mean(state_reveal_vals)) if len(state_reveal_vals) > 0 else np.nan,
        "state_reveal_n": int(len(state_reveal_vals)),
        "monitoring_mean": float(np.mean(monitoring_vals)) if len(monitoring_vals) > 0 else np.nan,
        "monitoring_n": int(len(monitoring_vals)),
        "over_personalized_mean": float(np.mean(over_personalized_vals)) if len(over_personalized_vals) > 0 else np.nan,
        "over_personalized_n": int(len(over_personalized_vals)),
        "privacy_invasive_mean": float(np.mean(privacy_invasive_vals)) if len(privacy_invasive_vals) > 0 else np.nan,
        "privacy_invasive_n": int(len(privacy_invasive_vals)),
    }
    return summary, raw_rows


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.RandomState(seed)
    boots = []
    n = arr.size
    for _ in range(int(n_boot)):
        samp = arr[rng.randint(0, n, size=n)]
        boots.append(float(np.mean(samp)))

    mean_val = float(np.mean(arr))
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return mean_val, lo, hi

def build_llm_bootstrap_summary(
    raw_rows: List[Dict[str, Any]],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
):
    if len(raw_rows) == 0:
        return pd.DataFrame(
            columns=["policy", "persona", "metric", "mean", "ci_low", "ci_high", "n_users"]
        )

    df = pd.DataFrame(raw_rows)
    rows = []

    group_cols = ["policy", "persona"]
    for (policy, persona), sub in df.groupby(group_cols, dropna=False):
        users = sorted(sub["user"].dropna().unique().tolist())
        if len(users) == 0:
            continue

        # single acc
        ss = sub[sub["audit_type"] == "single"].copy()
        if len(ss) > 0 and "correct" in ss.columns:
            user_vals = ss.groupby("user")["correct"].mean().astype(float).values
            m, lo, hi = bootstrap_mean_ci(user_vals, n_boot=n_boot, alpha=alpha, seed=seed + 11)
            rows.append({
                "policy": policy,
                "persona": persona,
                "metric": "single_acc",
                "mean": m,
                "ci_low": lo,
                "ci_high": hi,
                "n_users": len(user_vals),
            })

        # pair acc
        pp = sub[sub["audit_type"] == "pair"].copy()
        if len(pp) > 0:
            use_col = "pair_credit" if "pair_credit" in pp.columns else "correct"
            if use_col in pp.columns:
                user_vals = pp.groupby("user")[use_col].mean().astype(float).values
                m, lo, hi = bootstrap_mean_ci(user_vals, n_boot=n_boot, alpha=alpha, seed=seed + 17)
                rows.append({
                    "policy": policy,
                    "persona": persona,
                    "metric": "pair_acc",
                    "mean": m,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_users": len(user_vals),
                })

        # ratings
        rr = sub[sub["audit_type"] == "rating"].copy()
        for metric in ["state_reveal", "monitoring", "over_personalized", "privacy_invasive"]:
            if len(rr) > 0 and metric in rr.columns:
                user_vals = rr.groupby("user")[metric].mean().astype(float).values
                user_vals = user_vals[np.isfinite(user_vals)]
                if len(user_vals) == 0:
                    continue
                m, lo, hi = bootstrap_mean_ci(
                    user_vals, n_boot=n_boot, alpha=alpha, seed=seed + abs(hash(metric)) % 1000
                )
                rows.append({
                    "policy": policy,
                    "persona": persona,
                    "metric": metric,
                    "mean": m,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_users": len(user_vals),
                })

    return pd.DataFrame(rows)

def aggregate_llm_policy_from_persona_df(persona_df: pd.DataFrame):
    if persona_df is None or len(persona_df) == 0:
        return pd.DataFrame(
            columns=[
                "policy",
                "single_acc", "single_auc", "pair_acc",
                "single_n", "pair_n",
                "monitoring_mean", "state_reveal_mean",
                "over_personalized_mean", "privacy_invasive_mean",
                "monitoring_n", "state_reveal_n",
                "over_personalized_n", "privacy_invasive_n",
            ]
        )

    df = persona_df.copy()
    rows = []

    mean_cols = [
        "single_acc", "single_auc", "pair_acc",
        "monitoring_mean", "state_reveal_mean",
        "over_personalized_mean", "privacy_invasive_mean",
    ]
    count_cols = [
        "single_n", "pair_n",
        "monitoring_n", "state_reveal_n",
        "over_personalized_n", "privacy_invasive_n",
    ]

    for policy, sub in df.groupby("policy", dropna=False):
        row = {"policy": policy}
        for c in mean_cols:
            row[c] = float(sub[c].mean()) if c in sub.columns and sub[c].notna().any() else np.nan
        for c in count_cols:
            if c in sub.columns and sub[c].notna().any():
                row[c] = int(pd.to_numeric(sub[c], errors="coerce").dropna().iloc[0])
            else:
                row[c] = 0
        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) > 0:
        order_map = {p: i for i, p in enumerate(POLICY_ORDER)}
        out["policy_order"] = out["policy"].map(order_map)
        out = out.sort_values("policy_order").drop(columns=["policy_order"]).reset_index(drop=True)
    return out
def aggregate_llm_persona_rows(llm_summary_df: pd.DataFrame):
    if llm_summary_df is None or len(llm_summary_df) == 0:
        return pd.DataFrame(
            columns=[
                "policy", "persona",
                "single_acc", "single_auc", "pair_acc",
                "single_n", "pair_n",
                "monitoring_mean", "state_reveal_mean",
                "over_personalized_mean", "privacy_invasive_mean",
                "monitoring_n", "state_reveal_n",
                "over_personalized_n", "privacy_invasive_n",
            ]
        )

    df = llm_summary_df.copy()

    numeric_cols = [
        "single_acc", "single_auc", "pair_acc",
        "single_n", "pair_n",
        "monitoring_mean", "state_reveal_mean",
        "over_personalized_mean", "privacy_invasive_mean",
        "monitoring_n", "state_reveal_n",
        "over_personalized_n", "privacy_invasive_n",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    for (policy, persona), sub in df.groupby(["policy", "persona"], dropna=False):
        row = {
            "policy": policy,
            "persona": persona,
        }

        mean_cols = [
            "single_acc", "single_auc", "pair_acc",
            "monitoring_mean", "state_reveal_mean",
            "over_personalized_mean", "privacy_invasive_mean",
        ]
        for c in mean_cols:
            row[c] = float(sub[c].mean()) if c in sub.columns and sub[c].notna().any() else np.nan

        # 这里按同一 persona 下 MC 平均/复用，不跨 persona 累加
        count_cols = [
            "single_n", "pair_n",
            "monitoring_n", "state_reveal_n",
            "over_personalized_n", "privacy_invasive_n",
        ]
        for c in count_cols:
            if c in sub.columns and sub[c].notna().any():
                row[c] = int(pd.to_numeric(sub[c], errors="coerce").dropna().iloc[0])
            else:
                row[c] = 0

        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out["persona_sort"] = out["persona"].astype(str)
        out = out.sort_values(["policy", "persona_sort"]).drop(columns=["persona_sort"]).reset_index(drop=True)
    return out

def run_llm_audit_suite(
    cfg: Config,
    data_dict,
    seed: int,
    seed_dir: str,
    test_users: List[int],
    topk_aware_z0: Dict[int, np.ndarray],
    topk_aware_z1: Dict[int, np.ndarray],
    topk_agn: Dict[int, np.ndarray],
    topk_ra_sel_z0: Dict[int, np.ndarray],
    topk_ra_sel_z1: Dict[int, np.ndarray],
    best_mix_cand: Dict[str, Any],
    best_mix_alpha: float,
):
    llm_summary_by_policy = {
        p: {
            "LLM.AUC": np.nan,
            "LLM.SingleAcc": np.nan,
            "LLM.PairAcc": np.nan,
            "LLM.SingleN": 0,
            "LLM.PairN": 0,
            "Monitor.Score": np.nan,
            "StateReveal.Score": np.nan,
            "PrivacyInvasive.Score": np.nan,
        }
        for p in POLICY_ORDER
    }

    if not cfg.llm_enabled:
        return llm_summary_by_policy

    try:
        llm_spec = resolve_llm_spec(cfg)
        if llm_spec is None:
            raise RuntimeError(
                f"Could not resolve llm_model_name={cfg.llm_model_name}. "
                f"Provide it in {cfg.llm_registry_path} or use a supported default alias."
            )

        llm_cache = load_llm_cache(cfg.llm_cache_path)
        llm_users = sample_users(test_users, cfg.llm_max_users, seed * 997 + 11)
        print(f"[seed={seed}] running LLM audit on {len(llm_users)} test users with model alias={cfg.llm_model_name}")

        llm_summary_rows_default = []
        llm_summary_rows_persona = []
        llm_raw_rows = []

        def _run_llm_policy_pair(
            target_summary_rows: List[Dict[str, Any]],
            policy_name: str,
            z0_dict: Dict[int, np.ndarray],
            z1_dict: Dict[int, np.ndarray],
            infer_seed: int,
            rating_seed: int,
            persona: Optional[str],
            mc_tag: Optional[int],
        ):
            sm, raw = run_llm_audit_policy(
                cfg, llm_spec, llm_cache, data_dict, llm_users,
                z0_dict, z1_dict,
                seed=infer_seed,
                policy_name=policy_name,
                persona=persona,
                mc_tag=mc_tag,
            )
            target_summary_rows.append(sm)
            llm_raw_rows.extend(raw)

            sm_rate, raw_rate = run_llm_rating_audit_policy(
                cfg, llm_spec, llm_cache, data_dict, llm_users,
                z0_dict, z1_dict,
                seed=rating_seed,
                policy_name=policy_name,
                persona=persona,
                mc_tag=mc_tag,
            )
            target_summary_rows.append(sm_rate)
            llm_raw_rows.extend(raw_rate)

        # default auditor，保留原始记录
        _run_llm_policy_pair(llm_summary_rows_default, "State-aware", topk_aware_z0, topk_aware_z1, seed * 1000 + 1, seed * 1000 + 101, None, None)
        _run_llm_policy_pair(llm_summary_rows_default, "State-independent", topk_agn, topk_agn, seed * 1000 + 2, seed * 1000 + 102, None, None)
        _run_llm_policy_pair(llm_summary_rows_default, "RA", topk_ra_sel_z0, topk_ra_sel_z1, seed * 1000 + 3, seed * 1000 + 103, None, None)

        for mc in range(cfg.llm_mix_mc):
            mx_z0, mx_z1 = sample_mix_transcripts(
                llm_users,
                topk_agn=topk_agn,
                topk_ra_z0=best_mix_cand["topk_z0"],
                topk_ra_z1=best_mix_cand["topk_z1"],
                alpha=best_mix_alpha,
                seed=seed * 20000 + 100 * mc + 17,
                shared_coin=cfg.mix_shared_coin_in_audit,
            )
            _run_llm_policy_pair(
                llm_summary_rows_default, "RA+Mix",
                mx_z0, mx_z1,
                seed * 30000 + 100 * mc + 19,
                seed * 40000 + 100 * mc + 23,
                None, mc,
            )

        # persona auditors：主表以后用这个平均
        if len(cfg.llm_personas) > 0:
            for p_idx, persona in enumerate(cfg.llm_personas):
                offset = 1000000 + 100000 * p_idx
                _run_llm_policy_pair(llm_summary_rows_persona, "State-aware", topk_aware_z0, topk_aware_z1, seed * 1000 + offset + 1, seed * 1000 + offset + 101, persona, None)
                _run_llm_policy_pair(llm_summary_rows_persona, "State-independent", topk_agn, topk_agn, seed * 1000 + offset + 2, seed * 1000 + offset + 102, persona, None)
                _run_llm_policy_pair(llm_summary_rows_persona, "RA", topk_ra_sel_z0, topk_ra_sel_z1, seed * 1000 + offset + 3, seed * 1000 + offset + 103, persona, None)

                for mc in range(cfg.llm_mix_mc):
                    mx_z0, mx_z1 = sample_mix_transcripts(
                        llm_users,
                        topk_agn=topk_agn,
                        topk_ra_z0=best_mix_cand["topk_z0"],
                        topk_ra_z1=best_mix_cand["topk_z1"],
                        alpha=best_mix_alpha,
                        seed=seed * 500000 + 10000 * p_idx + 100 * mc + 17,
                        shared_coin=cfg.mix_shared_coin_in_audit,
                    )
                    _run_llm_policy_pair(
                        llm_summary_rows_persona, "RA+Mix",
                        mx_z0, mx_z1,
                        seed * 600000 + 10000 * p_idx + 100 * mc + 19,
                        seed * 700000 + 10000 * p_idx + 100 * mc + 23,
                        persona, mc,
                    )

        llm_summary_df_all = pd.DataFrame(llm_summary_rows_default + llm_summary_rows_persona)
        llm_summary_df_all.to_csv(os.path.join(seed_dir, "llm_audit_summary_raw.csv"), index=False)
        write_jsonl(os.path.join(seed_dir, "llm_audit_raw.jsonl"), llm_raw_rows)

        llm_persona_df = aggregate_llm_persona_rows(pd.DataFrame(llm_summary_rows_persona))
        llm_persona_df.to_csv(os.path.join(seed_dir, "llm_audit_summary_persona.csv"), index=False)

        # 主表优先用 persona average；若没有 persona，再 fallback 到 default auditor
        if len(llm_persona_df) > 0:
            llm_policy_df = aggregate_llm_policy_from_persona_df(llm_persona_df)
            llm_policy_df["summary_source"] = "persona_average"
        else:
            llm_default_df = pd.DataFrame(llm_summary_rows_default)
            llm_policy_df = aggregate_llm_policy_from_persona_df(
                aggregate_llm_persona_rows(llm_default_df.assign(persona="default"))
            )
            llm_policy_df["summary_source"] = "default_single_auditor"

        llm_policy_df.to_csv(os.path.join(seed_dir, "llm_audit_summary_policy.csv"), index=False)

        llm_boot_df = build_llm_bootstrap_summary(
            llm_raw_rows,
            n_boot=getattr(cfg, "llm_bootstrap_n", 1000),
            alpha=getattr(cfg, "llm_bootstrap_alpha", 0.05),
            seed=seed * 313 + 7,
        )
        llm_boot_df.to_csv(os.path.join(seed_dir, "llm_audit_bootstrap_summary.csv"), index=False)

        for p in POLICY_ORDER:
            sub = llm_policy_df[llm_policy_df["policy"] == p].copy()
            if len(sub) == 0:
                continue
            row = sub.iloc[0].to_dict()
            llm_summary_by_policy[p] = {
                "LLM.AUC": maybe_float(row.get("single_auc")),
                "LLM.SingleAcc": maybe_float(row.get("single_acc")),
                "LLM.PairAcc": maybe_float(row.get("pair_acc")),
                "LLM.SingleN": 0 if pd.isna(row.get("single_n", np.nan)) else int(row.get("single_n", 0)),
                "LLM.PairN": 0 if pd.isna(row.get("pair_n", np.nan)) else int(row.get("pair_n", 0)),
                "Monitor.Score": maybe_float(row.get("monitoring_mean")),
                "StateReveal.Score": maybe_float(row.get("state_reveal_mean")),
                "PrivacyInvasive.Score": maybe_float(row.get("privacy_invasive_mean")),
            }

    except Exception as e:
        print(f"[seed={seed}] WARNING: LLM audit skipped due to error: {repr(e)}")

    return llm_summary_by_policy

