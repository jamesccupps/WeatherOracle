"""Claude API Advisor v2 — Closed-loop weather regime analysis.

Periodically sends context to Claude and gets model weight adjustments.
Tracks whether its advice actually improved predictions (feedback loop).
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

log = logging.getLogger("WeatherOracle.advisor")

SYSTEM_PROMPT = """You are a meteorological advisor for WeatherOracle, a hyperlocal ML weather prediction system.

Analyze the current weather regime and recent model performance, then recommend which weather models to trust more or less.

You receive:
1. NWS forecast discussion from GYX (Gray, Maine)
2. Current model predictions and recent accuracy scores
3. Current observations from Tempest weather stations
4. Your past advice and whether it helped or hurt (feedback)

Return ONLY valid JSON:
{
  "regime": "brief description of current weather pattern",
  "reasoning": "2-3 sentences explaining your analysis",
  "adjustments": {
    "gfs": 1.0, "hrrr": 1.0, "ecmwf": 1.0,
    "icon": 1.0, "gem": 1.0, "jma": 1.0, "tempest_bf": 1.0
  },
  "confidence_note": "any concerns about forecast confidence",
  "watch_for": "what to expect in next 6-12 hours"
}

Adjustments are weight MULTIPLIERS (1.0 = no change, 1.5 = trust more, 0.5 = trust less).

Guidelines:
- Frontal passages: HRRR and ECMWF handle timing best
- Coastal effects (sea breeze, fog): Tempest BF and HRRR are better
- Synoptic-scale (2-3 day): GFS and ECMWF dominate
- JMA is often an outlier for New England — usually downweight
- ICON handles European-origin systems well
- GEM is relevant for Canadian air masses
- Tempest BetterForecast is already localized — trust for short-term
- If your past advice HURT accuracy, explain why and adjust strategy

Return ONLY JSON, no markdown."""


class ClaudeAdvisor:
    """Consults Claude API with feedback loop."""

    API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.adjustments = {}
        self.regime = ""
        self.reasoning = ""
        self.watch_for = ""
        self.last_run = None

        # Feedback tracking
        self.advice_history = []  # [{timestamp, adjustments, regime, ...}]
        self.feedback_scores = []  # [{timestamp, helped: bool, delta_mae: float}]

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get_adjustments(self) -> dict:
        return self.adjustments

    def record_feedback(self, helped: bool, delta_mae: float,
                        details: str = ""):
        """Record whether the last advice improved accuracy.

        helped: True if adjusted predictions were better than baseline
        delta_mae: negative = improved, positive = worse
        """
        self.feedback_scores.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "helped": helped,
            "delta_mae": round(delta_mae, 3),
            "details": details,
        })
        # Keep last 20 feedback entries
        self.feedback_scores = self.feedback_scores[-20:]
        log.info("[ADVISOR] Feedback: %s (ΔMAE=%+.2f°F) %s",
                 "helped" if helped else "hurt", delta_mae, details)

    def _build_feedback_text(self) -> str:
        """Format feedback history for Claude."""
        if not self.feedback_scores:
            return "No feedback yet — this is the first advisory cycle."

        total = len(self.feedback_scores)
        helped = sum(1 for f in self.feedback_scores if f["helped"])
        avg_delta = (sum(f["delta_mae"] for f in self.feedback_scores) / total
                     if total else 0)

        text = (f"FEEDBACK ON YOUR PAST ADVICE:\n"
                f"  Your advice helped {helped}/{total} times "
                f"(avg ΔMAE = {avg_delta:+.2f}°F)\n")

        # Show last 5 entries
        for f in self.feedback_scores[-5:]:
            text += (f"  [{f['timestamp'][:16]}] "
                     f"{'✓ Helped' if f['helped'] else '✗ Hurt'}: "
                     f"ΔMAE={f['delta_mae']:+.2f}°F {f['details']}\n")

        if avg_delta > 0:
            text += "  ⚠ Your recent advice has been HURTING accuracy. Adjust strategy.\n"
        elif avg_delta < -0.5:
            text += "  ✓ Your advice is consistently improving predictions. Keep going.\n"

        return text

    def analyze(self, nws_discussion: str, observations: dict,
                model_forecasts: dict, scoreboard_summary: dict,
                location_name: str) -> Optional[dict]:
        """Send context to Claude and get weight adjustments."""
        if not self.api_key:
            return None

        # Build observation summary
        obs_text = ""
        for loc, obs in observations.items():
            if obs:
                obs_text += (f"  {loc}: {obs.get('temp_f')}°F, "
                             f"{obs.get('humidity')}% RH, "
                             f"wind {obs.get('wind_mph')} mph\n")

        # Model forecasts (next 6h summary)
        fc_text = "Next 6-hour predictions:\n"
        for model, data in model_forecasts.items():
            if isinstance(data, list) and data:
                temps = [f.get("temp_f") for f in data[:6] if f.get("temp_f")]
                if temps:
                    fc_text += f"  {model}: {temps[0]:.0f}→{temps[-1]:.0f}°F\n"

        # Scoreboard summary
        score_text = "Recent model accuracy (lower MAE = better):\n"
        for key, stats in sorted(scoreboard_summary.items()):
            score_text += (f"  {key}: MAE={stats['mae']}°F, "
                           f"bias={stats['bias']:+.1f}°F "
                           f"({stats['n']} samples)\n")

        # Trim NWS discussion
        afd = nws_discussion or "Not available"
        if len(afd) > 2500:
            sections = []
            for section in afd.split("\n\n"):
                lower = section.lower()
                if any(k in lower for k in ["synopsis", "near term",
                                              "short term", "discussion"]):
                    sections.append(section[:600])
            afd = "\n\n".join(sections[:3]) if sections else afd[:2000]

        feedback_text = self._build_feedback_text()

        user_msg = f"""Analyze weather for {location_name}:

OBSERVATIONS:
{obs_text}

NWS DISCUSSION (GYX):
{afd[:2000]}

{fc_text}

{score_text}

{feedback_text}

Which models should be trusted more or less right now? Return JSON only."""

        try:
            resp = requests.post(
                self.API_URL,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 500,
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": user_msg}],
                },
                timeout=30,
            )

            if resp.status_code != 200:
                log.warning("Claude API HTTP %d: %s",
                            resp.status_code, resp.text[:200])
                return None

            data = resp.json()
            text = data.get("content", [{}])[0].get("text", "").strip()

            # Clean markdown wrapping
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            advice = json.loads(text)

            # Apply
            self.adjustments = advice.get("adjustments", {})
            self.regime = advice.get("regime", "")
            self.reasoning = advice.get("reasoning", "")
            self.watch_for = advice.get("watch_for", "")
            self.last_run = datetime.now(timezone.utc).isoformat()

            # Store in history for feedback tracking
            self.advice_history.append({
                "timestamp": self.last_run,
                "adjustments": self.adjustments.copy(),
                "regime": self.regime,
            })
            self.advice_history = self.advice_history[-10:]

            log.info("[ADVISOR] Regime: %s", self.regime)
            log.info("[ADVISOR] %s", self.reasoning)
            for mk, mult in self.adjustments.items():
                if abs(mult - 1.0) > 0.05:
                    log.info("[ADVISOR] %s: ×%.2f", mk, mult)

            return advice

        except json.JSONDecodeError as e:
            log.warning("Claude JSON error: %s", e)
            return None
        except Exception as e:
            log.error("Claude advisor error: %s", e)
            return None

    def get_status(self) -> dict:
        total = len(self.feedback_scores)
        helped = sum(1 for f in self.feedback_scores if f["helped"])
        return {
            "configured": self.is_configured(),
            "last_run": self.last_run,
            "regime": self.regime,
            "reasoning": self.reasoning,
            "watch_for": self.watch_for,
            "adjustments": self.adjustments,
            "feedback_total": total,
            "feedback_helped": helped,
            "feedback_pct": round(helped / total * 100, 0) if total else 0,
        }
