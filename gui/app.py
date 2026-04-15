"""WeatherOracle tkinter GUI — dual-location real-time weather + independent forecasts."""

import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
from datetime import datetime

from core.config import LOCATIONS, WEATHER_MODELS, WMO_CODES, WMO_ICONS
from core.config import load_config, save_config, DB_PATH
from core.orchestrator import WeatherOracle
from ml.deep_backfill import DeepBackfill, BackfillThread

log = logging.getLogger("WeatherOracle.gui")

# ─── Colors ───────────────────────────────────────────────────────────────────

BG       = "#0f1923"
BG2      = "#1a2733"
CARD     = "#172a3a"
CARD_HI  = "#1f3547"
FG       = "#e0e8f0"
FG_DIM   = "#7a8a9a"
ACCENT_G = "#00d4aa"  # location 1 (green)
ACCENT_B = "#4a90d9"  # location 2 (blue)
WARN     = "#ff6b6b"
OK       = "#4ade80"
BORDER   = "#2a3a4a"


class WeatherOracleGUI:
    """Main application window with dual-pane real-time weather."""

    def __init__(self):
        self.config = load_config()
        self.oracle = None
        self._backfill_thread = None

        self.root = tk.Tk()
        self.root.title("WeatherOracle v2.0 — Hyperlocal ML Weather")
        self.root.geometry("1400x900")
        self.root.minsize(1100, 700)
        self.root.configure(bg=BG)

        self._setup_styles()
        self._build_ui()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Styles ────────────────────────────────────────────────────────────

    def _setup_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TNotebook", background=BG)
        s.configure("TNotebook.Tab", background=CARD, foreground=FG,
                    padding=[14, 6], font=("Segoe UI", 10))
        s.map("TNotebook.Tab",
              background=[("selected", ACCENT_B)],
              foreground=[("selected", "#fff")])
        s.configure("TFrame", background=BG)
        s.configure("Card.TFrame", background=CARD)
        s.configure("TLabel", background=BG, foreground=FG,
                    font=("Segoe UI", 10))
        s.configure("Card.TLabel", background=CARD, foreground=FG)
        s.configure("Dim.TLabel", background=CARD, foreground=FG_DIM,
                    font=("Segoe UI", 9))
        s.configure("Hdr.TLabel", background=BG, foreground=ACCENT_G,
                    font=("Segoe UI", 11, "bold"))
        s.configure("HdrB.TLabel", background=BG, foreground=ACCENT_B,
                    font=("Segoe UI", 11, "bold"))
        s.configure("BigG.TLabel", background=CARD, foreground=ACCENT_G,
                    font=("Segoe UI", 32, "bold"))
        s.configure("BigB.TLabel", background=CARD, foreground=ACCENT_B,
                    font=("Segoe UI", 32, "bold"))
        s.configure("Med.TLabel", background=CARD, foreground=FG,
                    font=("Segoe UI", 14))
        s.configure("Sm.TLabel", background=CARD, foreground=FG_DIM,
                    font=("Segoe UI", 8))
        s.configure("TButton", font=("Segoe UI", 10))

    # ── UI Build ──────────────────────────────────────────────────────────

    def _build_ui(self):
        # Title bar
        title_bar = ttk.Frame(self.root)
        title_bar.pack(fill="x", padx=10, pady=(8, 2))
        ttk.Label(title_bar, text="WEATHERORACLE",
                 font=("Segoe UI", 16, "bold"),
                 foreground=ACCENT_G).pack(side="left")
        ttk.Label(title_bar, text="  Hyperlocal ML Weather Prediction",
                 foreground=FG_DIM, font=("Segoe UI", 10)).pack(side="left", padx=10)

        self.status_var = tk.StringVar(value="Configure API tokens in Settings")
        ttk.Label(title_bar, textvariable=self.status_var,
                 foreground=FG_DIM, font=("Segoe UI", 9)).pack(side="right")

        # Notebook
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=8, pady=4)

        self._build_dashboard_tab()
        self._build_forecast_tab()
        self._build_accuracy_tab()
        self._build_training_tab()
        self._build_data_tab()
        self._build_settings_tab()
        self._build_log_tab()

    # ── Dashboard Tab (dual real-time) ────────────────────────────────────

    def _build_dashboard_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Dashboard  ")

        # Two side-by-side panels
        panes = ttk.Frame(tab)
        panes.pack(fill="both", expand=True, padx=5, pady=5)
        panes.columnconfigure(0, weight=1)
        panes.columnconfigure(1, weight=1)

        self.loc_panels = {}
        for col, (loc_key, loc) in enumerate(LOCATIONS.items()):
            accent = loc.get("color", ACCENT_G)
            big_style = "BigG.TLabel" if col == 0 else "BigB.TLabel"
            hdr_style = "Hdr.TLabel" if col == 0 else "HdrB.TLabel"

            panel = ttk.Frame(panes, style="Card.TFrame")
            panel.grid(row=0, column=col, sticky="nsew", padx=4, pady=4)
            panes.rowconfigure(0, weight=1)

            # Header
            hdr = ttk.Frame(panel, style="Card.TFrame")
            hdr.pack(fill="x", padx=12, pady=(12, 4))
            ttk.Label(hdr, text=loc["name"].upper(),
                     foreground=accent, background=CARD,
                     font=("Segoe UI", 12, "bold")).pack(side="left")
            station_lbl = ttk.Label(hdr, text=f"Tempest #{loc['tempest_station']}",
                                    style="Dim.TLabel")
            station_lbl.pack(side="right")

            # Current conditions grid
            vars_dict = {}
            metrics = [
                ("temp", "Temperature", "°F", big_style),
                ("feels", "Feels Like", "°F", "Med.TLabel"),
                ("humidity", "Humidity", "%", "Med.TLabel"),
                ("dewpoint", "Dewpoint", "°F", "Med.TLabel"),
                ("wind", "Wind", "mph", "Med.TLabel"),
                ("gust", "Gusts", "mph", "Med.TLabel"),
                ("pressure", "Pressure", "mb", "Med.TLabel"),
                ("precip", "Precip Today", "in", "Med.TLabel"),
                ("solar", "Solar", "W/m²", "Med.TLabel"),
                ("uv", "UV Index", "", "Med.TLabel"),
            ]

            # Big temp
            temp_frame = ttk.Frame(panel, style="Card.TFrame")
            temp_frame.pack(fill="x", padx=12, pady=(8, 2))
            vars_dict["temp"] = tk.StringVar(value="--")
            ttk.Label(temp_frame, textvariable=vars_dict["temp"],
                     style=big_style).pack(side="left")
            ttk.Label(temp_frame, text="°F", background=CARD,
                     foreground=FG_DIM, font=("Segoe UI", 16)).pack(side="left", pady=(10, 0))

            vars_dict["condition"] = tk.StringVar(value="--")
            ttk.Label(temp_frame, textvariable=vars_dict["condition"],
                     background=CARD, foreground=FG,
                     font=("Segoe UI", 14)).pack(side="right")

            # Metrics grid
            grid = ttk.Frame(panel, style="Card.TFrame")
            grid.pack(fill="x", padx=12, pady=4)
            for i in range(3):
                grid.columnconfigure(i, weight=1)

            for idx, (key, label, unit, style) in enumerate(metrics[1:]):  # skip temp
                r, c = divmod(idx, 3)
                cell = ttk.Frame(grid, style="Card.TFrame")
                cell.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
                vars_dict[key] = tk.StringVar(value="--")
                ttk.Label(cell, textvariable=vars_dict[key],
                         background=CARD, foreground=FG,
                         font=("Segoe UI", 13, "bold")).pack()
                ttk.Label(cell, text=f"{label} ({unit})" if unit else label,
                         style="Sm.TLabel").pack()

            # Cross-check section (first location with HA configured)
            if col == 0:
                xcheck_frame = ttk.Frame(panel, style="Card.TFrame")
                xcheck_frame.pack(fill="x", padx=12, pady=(4, 2))
                ttk.Label(xcheck_frame, text="SENSOR CROSS-CHECK",
                         foreground=accent, background=CARD,
                         font=("Segoe UI", 9, "bold")).pack(anchor="w")
                vars_dict["xcheck"] = tk.StringVar(value="Waiting for data...")
                ttk.Label(xcheck_frame, textvariable=vars_dict["xcheck"],
                         style="Dim.TLabel", wraplength=500).pack(anchor="w")

            # Alerts
            alert_frame = ttk.Frame(panel, style="Card.TFrame")
            alert_frame.pack(fill="x", padx=12, pady=(4, 2))
            ttk.Label(alert_frame, text="NWS ALERTS",
                     foreground=accent, background=CARD,
                     font=("Segoe UI", 9, "bold")).pack(anchor="w")
            vars_dict["alerts"] = tk.StringVar(value="No active alerts")
            ttk.Label(alert_frame, textvariable=vars_dict["alerts"],
                     background=CARD, foreground=WARN,
                     font=("Segoe UI", 9), wraplength=500).pack(anchor="w")

            # Updated timestamp
            vars_dict["updated"] = tk.StringVar(value="")
            ttk.Label(panel, textvariable=vars_dict["updated"],
                     style="Dim.TLabel").pack(side="bottom", anchor="e",
                                               padx=12, pady=(0, 8))

            self.loc_panels[loc_key] = vars_dict

        # Refresh button
        btn_row = ttk.Frame(tab)
        btn_row.pack(fill="x", padx=10, pady=4)
        ttk.Button(btn_row, text="↻ Refresh Now",
                  command=self._refresh_dashboard).pack(side="right")

    def _refresh_dashboard(self):
        if not self.oracle:
            return
        for loc_key, vars_dict in self.loc_panels.items():
            obs = self.oracle.get_current(loc_key)
            if obs:
                vars_dict["temp"].set(f"{obs.get('temp_f', '--')}")
                vars_dict["feels"].set(f"{obs.get('feels_like_f', '--')}")
                vars_dict["humidity"].set(f"{obs.get('humidity', '--')}")
                vars_dict["dewpoint"].set(f"{obs.get('dewpoint_f', '--')}")
                vars_dict["wind"].set(f"{obs.get('wind_mph', '--')}")
                vars_dict["gust"].set(f"{obs.get('wind_gust_mph', '--')}")
                vars_dict["pressure"].set(f"{obs.get('pressure_mb', '--')}")
                vars_dict["precip"].set(f"{obs.get('precip_in', '0')}")
                vars_dict["solar"].set(f"{obs.get('solar_radiation', '--')}")
                vars_dict["uv"].set(f"{obs.get('uv_index', '--')}")
                ts = obs.get("timestamp", "")[:19]
                vars_dict["updated"].set(f"Updated: {ts}")

                # Condition from nearest forecast weather code
                raw = self.oracle.get_raw_forecasts(loc_key)
                if raw:
                    wc = raw[0].get("weather_code")
                    icon = WMO_ICONS.get(wc, "")
                    desc = WMO_CODES.get(wc, "")
                    vars_dict["condition"].set(f"{icon} {desc}")

            # Cross-check (first location)
            if "xcheck" in vars_dict:
                xc = self.oracle.get_crosscheck(loc_key)
                tempest_t = obs.get("temp_f") if obs else None
                parts = []
                for src, data in xc.items():
                    t = data.get("temp_f")
                    delta = f" (Δ{abs(t - tempest_t):.1f}°)" if t and tempest_t else ""
                    parts.append(f"{src}: {t}°F{delta}")
                vars_dict["xcheck"].set(" │ ".join(parts) if parts else "No cross-check data")

            # Alerts
            alerts = self.oracle.get_alerts_for(loc_key)
            if alerts:
                vars_dict["alerts"].set(
                    " │ ".join(f"⚠ {a['event']}" for a in alerts))
            else:
                vars_dict["alerts"].set("No active alerts")

    # ── Forecast Tab (dual independent) ───────────────────────────────────

    def _build_forecast_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Forecasts  ")

        self.fc_trees = {}

        panes = ttk.Frame(tab)
        panes.pack(fill="both", expand=True, padx=5, pady=5)
        panes.columnconfigure(0, weight=1)
        panes.columnconfigure(1, weight=1)

        for col, (loc_key, loc) in enumerate(LOCATIONS.items()):
            accent = loc.get("color", ACCENT_G)

            frame = ttk.Frame(panes, style="Card.TFrame")
            frame.grid(row=0, column=col, sticky="nsew", padx=4, pady=4)
            panes.rowconfigure(0, weight=1)

            hdr = ttk.Frame(frame, style="Card.TFrame")
            hdr.pack(fill="x", padx=8, pady=(8, 4))
            ttk.Label(hdr, text=f"{loc['name'].upper()} — 72HR FORECAST",
                     foreground=accent, background=CARD,
                     font=("Segoe UI", 10, "bold")).pack(side="left")

            ml_status = tk.StringVar(value="")
            ttk.Label(hdr, textvariable=ml_status,
                     background=CARD, foreground=OK,
                     font=("Segoe UI", 9)).pack(side="right")

            cols = ("time", "temp", "hum", "wind", "precip", "sky", "conf")
            tree = ttk.Treeview(frame, columns=cols, show="headings", height=22)
            hdrs = {"time": "Time", "temp": "Temp°F", "hum": "RH%",
                    "wind": "Wind", "precip": "Precip%", "sky": "Sky",
                    "conf": "Conf%"}
            widths = {"time": 115, "temp": 65, "hum": 55, "wind": 60,
                      "precip": 60, "sky": 90, "conf": 55}
            for c in cols:
                tree.heading(c, text=hdrs[c])
                tree.column(c, width=widths[c], anchor="center")
            tree.column("time", anchor="w")

            sb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=sb.set)
            tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=(0, 8))
            sb.pack(side="right", fill="y", padx=(0, 8), pady=(0, 8))

            self.fc_trees[loc_key] = {"tree": tree, "ml_status": ml_status}

        btn_row = ttk.Frame(tab)
        btn_row.pack(fill="x", padx=10, pady=4)
        ttk.Button(btn_row, text="↻ Refresh Forecasts",
                  command=self._refresh_forecasts).pack(side="right")

    def _refresh_forecasts(self):
        if not self.oracle:
            return
        for loc_key, widgets in self.fc_trees.items():
            tree = widgets["tree"]
            tree.delete(*tree.get_children())

            # ML status
            n_models = self.oracle.ml.get_model_count(loc_key)
            if n_models > 0:
                widgets["ml_status"].set(f"✓ {n_models} ML models active")
            else:
                widgets["ml_status"].set("Weighted avg (no ML yet)")

            ensemble = self.oracle.get_forecast(loc_key)
            raw = self.oracle.get_raw_forecasts(loc_key)

            # Build weather code lookup from raw
            wc_map = {}
            for r in raw:
                if r.get("weather_code") is not None:
                    wc_map.setdefault(r["valid_at"], r["weather_code"])

            for fc in ensemble:
                valid = fc.get("valid_at", "")
                try:
                    dt = datetime.fromisoformat(valid)
                    ts = dt.strftime("%a %I%p").replace(" 0", " ")
                except:
                    ts = valid[:16]

                wc = fc.get("weather_code") or wc_map.get(valid)
                sky = WMO_CODES.get(wc, "") if wc is not None else "—"
                icon = WMO_ICONS.get(wc, "") if wc is not None else ""

                tree.insert("", "end", values=(
                    ts,
                    fc.get("temp_f", "—"),
                    fc.get("humidity", "—"),
                    fc.get("wind_mph", "—"),
                    fc.get("precip_prob", "—"),
                    f"{icon}{sky}",
                    fc.get("confidence", "—"),
                ))

    # ── Accuracy Tab ──────────────────────────────────────────────────────

    def _build_accuracy_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Model Accuracy  ")

        top = ttk.Frame(tab)
        top.pack(fill="x", padx=10, pady=8)
        self.acc_loc = tk.StringVar(value=list(LOCATIONS.keys())[0])
        for key, loc in LOCATIONS.items():
            ttk.Radiobutton(top, text=loc["name"], variable=self.acc_loc,
                           value=key).pack(side="left", padx=8)
        ttk.Button(top, text="Compute Accuracy",
                  command=self._refresh_accuracy).pack(side="right", padx=5)
        ttk.Button(top, text="Retrain ML",
                  command=self._retrain).pack(side="right", padx=5)
        ttk.Button(top, text="Backfill 14 Days",
                  command=self._backfill).pack(side="right", padx=5)
        self.acc_status = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.acc_status,
                 foreground=ACCENT_G).pack(side="right", padx=10)

        cols = ("model", "variable", "mae", "rmse", "bias", "n", "rank")
        self.acc_tree = ttk.Treeview(tab, columns=cols, show="headings", height=22)
        for c in cols:
            self.acc_tree.heading(c, text=c.upper())
            self.acc_tree.column(c, width=120, anchor="center")
        self.acc_tree.column("model", width=180, anchor="w")
        self.acc_tree.pack(fill="both", expand=True, padx=10, pady=(0, 8))

    def _refresh_accuracy(self):
        if not self.oracle:
            return
        loc = self.acc_loc.get()
        self.acc_status.set("Computing...")

        def _do():
            acc = self.oracle.ml.compute_accuracy(loc)
            self.root.after(0, lambda: self._display_accuracy(acc))

        threading.Thread(target=_do, daemon=True).start()

    def _display_accuracy(self, accuracy):
        self.acc_tree.delete(*self.acc_tree.get_children())
        for var in ("temp_f", "humidity", "wind_mph"):
            var_data = accuracy.get(var, {})
            ranked = sorted(var_data.items(), key=lambda x: x[1]["mae"])
            for rank, (model, stats) in enumerate(ranked, 1):
                name = WEATHER_MODELS.get(model, {}).get("name", model)
                if model == "ensemble":
                    name = "★ ML Ensemble"
                vdisplay = {"temp_f": "Temp°F", "humidity": "RH%",
                            "wind_mph": "Wind mph"}.get(var, var)
                self.acc_tree.insert("", "end", values=(
                    name, vdisplay, stats["mae"], stats["rmse"],
                    stats["bias"], stats["n"], f"#{rank}"))
        self.acc_status.set("Done")

    def _retrain(self):
        if not self.oracle:
            return
        self.acc_status.set("Training...")
        def _do():
            self.oracle.retrain()
            self.root.after(0, lambda: self.acc_status.set("Training complete!"))
        threading.Thread(target=_do, daemon=True).start()

    def _backfill(self):
        if not self.oracle:
            return
        self.acc_status.set("Backfilling...")
        def _do():
            self.oracle.backfill_observations(14)
            self.oracle.backfill_forecasts(14)
            stats = self.oracle.db.get_db_stats()
            self.root.after(0, lambda: self.acc_status.set(
                f"Done! {stats['observations']} obs, {stats['forecasts']} forecasts"))
        threading.Thread(target=_do, daemon=True).start()

    # ── Training Tab (deep backfill + ML training) ────────────────────────

    def _build_training_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Training  ")

        # ── Deep Backfill Section ──
        bf_frame = ttk.Frame(tab, style="Card.TFrame")
        bf_frame.pack(fill="x", padx=10, pady=(8, 4))

        hdr = ttk.Frame(bf_frame, style="Card.TFrame")
        hdr.pack(fill="x", padx=12, pady=(10, 4))
        ttk.Label(hdr, text="DEEP HISTORICAL BACKFILL",
                 foreground=ACCENT_G, background=CARD,
                 font=("Segoe UI", 11, "bold")).pack(side="left")

        ttk.Label(bf_frame, text=(
            "Pull years of Tempest observations + model forecasts to train "
            "the ML ensemble. More data = better predictions."
        ), style="Dim.TLabel", wraplength=900).pack(anchor="w", padx=12, pady=(0, 8))

        # Controls row
        ctrl = ttk.Frame(bf_frame, style="Card.TFrame")
        ctrl.pack(fill="x", padx=12, pady=4)

        ttk.Label(ctrl, text="Days back:", style="Card.TLabel").pack(side="left")
        self.bf_days = tk.StringVar(value="730")
        days_entry = ttk.Entry(ctrl, textvariable=self.bf_days, width=8)
        days_entry.pack(side="left", padx=6)

        # Quick presets
        for label, days in [("6 mo", 180), ("1 yr", 365), ("2 yr", 730), ("3 yr", 1095)]:
            ttk.Button(ctrl, text=label,
                      command=lambda d=days: self.bf_days.set(str(d))
                      ).pack(side="left", padx=3)

        ttk.Button(ctrl, text="Estimate",
                  command=self._estimate_backfill).pack(side="left", padx=10)

        self.bf_include_archive = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Include Open-Meteo archive (supplementary)",
                        variable=self.bf_include_archive).pack(side="left", padx=10)

        # Estimate display
        self.bf_estimate = tk.StringVar(value="Click 'Estimate' to see time and data projections")
        ttk.Label(bf_frame, textvariable=self.bf_estimate,
                 style="Dim.TLabel").pack(anchor="w", padx=12, pady=4)

        # Progress bars
        prog_frame = ttk.Frame(bf_frame, style="Card.TFrame")
        prog_frame.pack(fill="x", padx=12, pady=4)

        self.bf_phase_var = tk.StringVar(value="Ready")
        ttk.Label(prog_frame, textvariable=self.bf_phase_var,
                 background=CARD, foreground=ACCENT_G,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")

        self.bf_progress_var = tk.DoubleVar(value=0)
        self.bf_progress_bar = ttk.Progressbar(
            prog_frame, variable=self.bf_progress_var,
            maximum=100, length=400, mode="determinate")
        self.bf_progress_bar.pack(fill="x", pady=4)

        self.bf_detail_var = tk.StringVar(value="")
        ttk.Label(prog_frame, textvariable=self.bf_detail_var,
                 style="Dim.TLabel").pack(anchor="w")

        # Action buttons
        btn_row = ttk.Frame(bf_frame, style="Card.TFrame")
        btn_row.pack(fill="x", padx=12, pady=(4, 10))

        self.bf_start_btn = ttk.Button(btn_row, text="▶ Start Deep Backfill",
                                        command=self._start_deep_backfill)
        self.bf_start_btn.pack(side="left", padx=4)

        self.bf_cancel_btn = ttk.Button(btn_row, text="■ Cancel",
                                         command=self._cancel_backfill,
                                         state="disabled")
        self.bf_cancel_btn.pack(side="left", padx=4)

        ttk.Button(btn_row, text="Retrain After Backfill",
                  command=self._retrain_after_backfill).pack(side="left", padx=12)

        # ── Training Log Section ──
        log_frame = ttk.Frame(tab, style="Card.TFrame")
        log_frame.pack(fill="both", expand=True, padx=10, pady=(4, 8))

        log_hdr = ttk.Frame(log_frame, style="Card.TFrame")
        log_hdr.pack(fill="x", padx=12, pady=(10, 4))
        ttk.Label(log_hdr, text="ML TRAINING HISTORY",
                 foreground=ACCENT_B, background=CARD,
                 font=("Segoe UI", 11, "bold")).pack(side="left")
        ttk.Button(log_hdr, text="Refresh",
                  command=self._refresh_training_log).pack(side="right")

        cols = ("time", "loc", "var", "bucket", "samples",
                "cv_mae", "ens_mae", "best", "best_mae", "improv")
        self.train_tree = ttk.Treeview(log_frame, columns=cols, show="headings",
                                        height=10)
        headers = {"time": "Timestamp", "loc": "Location", "var": "Variable",
                   "bucket": "Lead Bucket", "samples": "Samples",
                   "cv_mae": "CV MAE", "ens_mae": "Ens MAE",
                   "best": "Best Model", "best_mae": "Best MAE",
                   "improv": "Improvement"}
        widths = {"time": 140, "loc": 80, "var": 70, "bucket": 70,
                  "samples": 65, "cv_mae": 65, "ens_mae": 65,
                  "best": 80, "best_mae": 65, "improv": 75}
        for c in cols:
            self.train_tree.heading(c, text=headers[c])
            self.train_tree.column(c, width=widths[c], anchor="center")
        self.train_tree.column("time", anchor="w")

        tsb = ttk.Scrollbar(log_frame, orient="vertical",
                            command=self.train_tree.yview)
        self.train_tree.configure(yscrollcommand=tsb.set)
        self.train_tree.pack(side="left", fill="both", expand=True,
                             padx=(12, 0), pady=(0, 10))
        tsb.pack(side="right", fill="y", padx=(0, 12), pady=(0, 10))

    def _estimate_backfill(self):
        try:
            days = int(self.bf_days.get())
        except ValueError:
            self.bf_estimate.set("Enter a valid number of days")
            return
        bf = DeepBackfill(None, "")
        est = bf.estimate_time(days)
        self.bf_estimate.set(
            f"~{est['est_minutes']:.0f} min  │  "
            f"~{est['est_observations']:,} observations  │  "
            f"~{est['est_forecasts']:,} forecast hours  │  "
            f"{est['tempest_api_calls']:,} Tempest calls  │  "
            f"{est['openmeteo_api_calls']:,} Open-Meteo calls"
        )

    def _start_deep_backfill(self):
        if not self.oracle or not self.config.get("tempest_api_token"):
            self.bf_phase_var.set("Configure Tempest API token first!")
            return

        if self._backfill_thread and self._backfill_thread.is_alive():
            self.bf_phase_var.set("Backfill already running")
            return

        try:
            days = int(self.bf_days.get())
        except ValueError:
            self.bf_phase_var.set("Enter valid days")
            return

        self.bf_start_btn.config(state="disabled")
        self.bf_cancel_btn.config(state="normal")
        self.bf_phase_var.set("Starting deep backfill...")
        self.bf_progress_var.set(0)

        def _on_progress(phase, pct, msg):
            self.root.after(0, lambda: self._update_bf_progress(phase, pct, msg))

        def _on_complete(results):
            self.root.after(0, lambda: self._backfill_complete(results))

        self._backfill_thread = BackfillThread(
            db=self.oracle.db,
            tempest_token=self.config["tempest_api_token"],
            days_back=days,
            include_archive=self.bf_include_archive.get(),
            include_ha=True,
            ha_url=self.config.get("ha_url", ""),
            ha_token=self.config.get("ha_token", ""),
            on_progress=_on_progress,
            on_complete=_on_complete,
        )
        self._backfill_thread.start()

    def _update_bf_progress(self, phase, pct, msg):
        phase_labels = {
            "tempest": "Phase 1: Tempest Observations",
            "archive": "Phase 2: Open-Meteo Archive",
            "forecasts": "Phase 3: Model Forecasts",
            "ha_history": "Phase 4: Home Assistant History",
            "overall": "Overall",
        }
        self.bf_phase_var.set(phase_labels.get(phase, phase))
        self.bf_progress_var.set(pct)
        self.bf_detail_var.set(msg)

    def _backfill_complete(self, results):
        self.bf_start_btn.config(state="normal")
        self.bf_cancel_btn.config(state="disabled")
        self.bf_progress_var.set(100)

        if "error" in results:
            self.bf_phase_var.set(f"Error: {results['error']}")
            return

        parts = []
        for loc_key, data in results.items():
            name = LOCATIONS[loc_key]["short"]
            parts.append(
                f"{name}: {data.get('tempest_obs', 0):,} obs, "
                f"{data.get('forecasts', 0):,} forecasts"
            )
        self.bf_phase_var.set("✓ Complete!")
        self.bf_detail_var.set(" │ ".join(parts))

        # Refresh stats
        try:
            self._refresh_stats()
        except:
            pass

    def _cancel_backfill(self):
        if self._backfill_thread and self._backfill_thread.is_alive():
            self._backfill_thread.cancel()
            self.bf_phase_var.set("Cancelling...")
            self.bf_cancel_btn.config(state="disabled")

    def _retrain_after_backfill(self):
        if not self.oracle:
            return
        self.bf_phase_var.set("Retraining ML models with all available data...")
        self.bf_progress_var.set(0)

        def _do():
            self.oracle.retrain()
            stats = self.oracle.db.get_db_stats()
            self.root.after(0, lambda: (
                self.bf_phase_var.set("✓ Retrain complete!"),
                self.bf_progress_var.set(100),
                self.bf_detail_var.set(
                    f"DB: {stats['observations']:,} obs, "
                    f"{stats['forecasts']:,} forecasts, "
                    f"ML models: " +
                    ", ".join(f"{lc['short']}={self.oracle.ml.get_model_count(lk)}"
                              for lk, lc in LOCATIONS.items())
                ),
                self._refresh_training_log(),
            ))

        threading.Thread(target=_do, daemon=True).start()

    def _refresh_training_log(self):
        if not self.oracle:
            return
        self.train_tree.delete(*self.train_tree.get_children())
        for loc_key in LOCATIONS:
            entries = self.oracle.ml.get_training_summary(loc_key)
            for e in entries:
                self.train_tree.insert("", "end", values=(
                    e.get("timestamp", "")[:19],
                    loc_key,
                    e.get("variable", ""),
                    e.get("bucket", ""),
                    e.get("n_samples", ""),
                    e.get("cv_mae", "—"),
                    e.get("ensemble_mae", "—"),
                    e.get("best_individual", ""),
                    e.get("best_individual_mae", "—"),
                    f"{e.get('improvement_pct', 0):+.1f}%",
                ))

    # ── Data Tab ──────────────────────────────────────────────────────────

    def _build_data_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Data & Stats  ")

        stats_frame = ttk.Frame(tab, style="Card.TFrame")
        stats_frame.pack(fill="x", padx=10, pady=8)
        self.stats_text = tk.Text(stats_frame, height=10, bg=CARD, fg=FG,
                                   font=("Consolas", 10), wrap="word",
                                   relief="flat", bd=0)
        self.stats_text.pack(fill="x", padx=10, pady=10)

        btn = ttk.Frame(tab)
        btn.pack(fill="x", padx=10, pady=4)
        ttk.Button(btn, text="Refresh", command=self._refresh_stats).pack(side="left")
        ttk.Button(btn, text="Export CSV", command=self._export_csv).pack(side="left", padx=8)

        cols = ("time", "loc", "source", "temp", "hum", "wind", "press", "dew")
        self.obs_tree = ttk.Treeview(tab, columns=cols, show="headings", height=15)
        for c in cols:
            self.obs_tree.heading(c, text=c.title())
            self.obs_tree.column(c, width=100, anchor="center")
        self.obs_tree.column("time", width=160)
        self.obs_tree.pack(fill="both", expand=True, padx=10, pady=(0, 8))

    def _refresh_stats(self):
        if not self.oracle:
            return
        stats = self.oracle.db.get_db_stats()
        self.stats_text.delete("1.0", "end")
        lines = [
            f"  {'Total Observations:':<25} {stats.get('observations', 0):,}",
        ]
        for lk, lc in LOCATIONS.items():
            lines.append(f"    {lc['short'] + ':':<23} {stats.get(f'obs_{lk}', 0):,}")
        lines.extend([
            f"  {'Total Forecasts:':<25} {stats.get('forecasts', 0):,}",
            f"  {'Ensemble Forecasts:':<25} {stats.get('ensemble_forecasts', 0):,}",
            f"  {'Training Runs:':<25} {stats.get('training_log', 0):,}",
            f"  {'Date Range:':<25} {(stats.get('obs_start') or 'N/A')[:10]}"
            f" → {(stats.get('obs_end') or 'N/A')[:10]}",
        ])
        for lk, lc in LOCATIONS.items():
            lines.append(f"  {'ML Models (' + lc['short'] + '):':<25} "
                        f"{self.oracle.ml.get_model_count(lk)}")
        try:
            sz = DB_PATH.stat().st_size
            lines.append(f"  {'DB Size:':<25} {sz/1024:.0f} KB")
        except:
            pass
        self.stats_text.insert("end", "\n".join(lines))

        self.obs_tree.delete(*self.obs_tree.get_children())
        for loc in LOCATIONS:
            for o in self.oracle.db.get_observations(loc, limit=25):
                self.obs_tree.insert("", "end", values=(
                    o["timestamp"][:19], loc, o["source"],
                    o.get("temp_f", "—"), o.get("humidity", "—"),
                    o.get("wind_mph", "—"), o.get("pressure_mb", "—"),
                    o.get("dewpoint_f", "—")))

    def _export_csv(self):
        if not self.oracle:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialfile="weather_oracle_export.csv")
        if not path:
            return
        try:
            import csv
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "location", "source", "temp_f",
                            "humidity", "dewpoint_f", "wind_mph",
                            "wind_gust_mph", "pressure_mb", "precip_in",
                            "solar_radiation", "uv_index", "feels_like_f"])
                for loc in LOCATIONS:
                    for o in self.oracle.db.get_observations(loc, limit=100000):
                        w.writerow([o["timestamp"], loc, o["source"],
                                    o.get("temp_f"), o.get("humidity"),
                                    o.get("dewpoint_f"), o.get("wind_mph"),
                                    o.get("wind_gust_mph"), o.get("pressure_mb"),
                                    o.get("precip_in"), o.get("solar_radiation"),
                                    o.get("uv_index"), o.get("feels_like_f")])
            messagebox.showinfo("Export", f"Exported to {path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    # ── Settings Tab ──────────────────────────────────────────────────────

    def _build_settings_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Settings  ")

        canvas = tk.Canvas(tab, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self.setting_vars = {}

        def _add(parent, label, key, width=50):
            row = ttk.Frame(parent)
            row.pack(fill="x", padx=20, pady=3)
            ttk.Label(row, text=label, width=28, anchor="w").pack(side="left")
            var = tk.StringVar(value=str(self.config.get(key, "")))
            self.setting_vars[key] = var
            e = ttk.Entry(row, textvariable=var, width=width)
            if "token" in key:
                e.config(show="•")
            e.pack(side="left", fill="x", expand=True)

        ttk.Label(inner, text="API CONFIGURATION",
                 style="Hdr.TLabel").pack(anchor="w", padx=20, pady=(15, 5))
        _add(inner, "Tempest API Token:", "tempest_api_token")
        _add(inner, "Home Assistant URL:", "ha_url")
        _add(inner, "Home Assistant Token:", "ha_token")
        _add(inner, "Claude API Key (optional):", "claude_api_key")

        ttk.Label(inner, text="HA CROSS-CHECK SENSORS (optional)",
                 style="Hdr.TLabel").pack(anchor="w", padx=20, pady=(15, 5))
        ttk.Label(inner, text="  Enter HA entity IDs for outdoor sensors to cross-check Tempest readings.",
                 foreground=FG_DIM, background=BG).pack(anchor="w", padx=20)
        _add(inner, "Sensor 1 Temp Entity:", "yolink_temp_entity")
        _add(inner, "Sensor 1 Humidity Entity:", "yolink_humidity_entity")
        _add(inner, "Sensor 2 Temp Entity:", "outdoor_temp_entity")
        _add(inner, "Sensor 2 Humidity Entity:", "outdoor_humidity_entity")
        _add(inner, "Avg Outdoor Temp Entity:", "avg_outdoor_temp_entity")
        _add(inner, "Avg Outdoor Humidity Entity:", "avg_outdoor_humidity_entity")

        ttk.Label(inner, text="COLLECTION TIMING",
                 style="Hdr.TLabel").pack(anchor="w", padx=20, pady=(15, 5))
        _add(inner, "Collection Interval (min):", "collection_interval_min")
        _add(inner, "Retrain Interval (hours):", "retrain_interval_hours")
        _add(inner, "Advisor Interval (hours):", "advisor_interval_hours")
        _add(inner, "Min Training Samples:", "min_training_samples")
        _add(inner, "Forecast Hours:", "forecast_hours")

        btn_frame = ttk.Frame(inner)
        btn_frame.pack(fill="x", padx=20, pady=15)
        ttk.Button(btn_frame, text="Save & Apply",
                  command=self._save_settings).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Test Tempest",
                  command=self._test_tempest).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Test HA",
                  command=self._test_ha).pack(side="left", padx=5)
        self.set_status = tk.StringVar(value="")
        ttk.Label(btn_frame, textvariable=self.set_status,
                 foreground=ACCENT_G).pack(side="left", padx=20)

        ttk.Label(inner, text="LOCATIONS",
                 style="Hdr.TLabel").pack(anchor="w", padx=20, pady=(15, 5))
        ttk.Label(inner, text="  Configure your Tempest stations. Changes require restart.",
                 foreground=FG_DIM, background=BG).pack(anchor="w", padx=20)

        self.loc_vars = {}
        loc_fields = [
            ("name", "Name:", 30),
            ("short", "Short Code:", 10),
            ("tempest_station", "Tempest Station ID:", 15),
            ("lat", "Latitude:", 15),
            ("lon", "Longitude:", 15),
            ("metar_station", "METAR Station (e.g. KJFK):", 10),
            ("nws_office", "NWS Office (e.g. OKX):", 10),
        ]

        cfg = load_config()
        locations = cfg.get("locations", {})
        if not locations:
            locations = {"location_1": {"name": "Home", "short": "HOME",
                                        "tempest_station": 0, "lat": 0.0,
                                        "lon": 0.0, "metar_station": "",
                                        "nws_office": ""}}

        for loc_idx, (loc_key, loc_data) in enumerate(locations.items()):
            lbl = f"Location {loc_idx + 1}"
            ttk.Label(inner, text=lbl.upper(),
                     foreground=ACCENT_G if loc_idx == 0 else ACCENT_B,
                     background=BG,
                     font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=30, pady=(10, 2))

            self.loc_vars[loc_key] = {}
            for field, label, width in loc_fields:
                row = ttk.Frame(inner)
                row.pack(fill="x", padx=40, pady=2)
                ttk.Label(row, text=label, width=28, anchor="w").pack(side="left")
                var = tk.StringVar(value=str(loc_data.get(field, "")))
                self.loc_vars[loc_key][field] = var
                ttk.Entry(row, textvariable=var, width=width).pack(side="left")

        # Add Location button
        def _add_location():
            n = len(self.loc_vars) + 1
            key = f"location_{n}"
            self.loc_vars[key] = {}
            ttk.Label(inner, text=f"LOCATION {n}",
                     foreground=ACCENT_B, background=BG,
                     font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=30, pady=(10, 2))
            for field, label, width in loc_fields:
                row = ttk.Frame(inner)
                row.pack(fill="x", padx=40, pady=2)
                ttk.Label(row, text=label, width=28, anchor="w").pack(side="left")
                var = tk.StringVar(value="")
                self.loc_vars[key][field] = var
                ttk.Entry(row, textvariable=var, width=width).pack(side="left")
            canvas.configure(scrollregion=canvas.bbox("all"))

        ttk.Button(inner, text="+ Add Location",
                  command=_add_location).pack(anchor="w", padx=30, pady=(10, 5))

    def _save_settings(self):
        for key, var in self.setting_vars.items():
            val = var.get()
            if key in ("collection_interval_min", "retrain_interval_hours",
                       "advisor_interval_hours", "min_training_samples",
                       "forecast_hours"):
                try: val = int(val)
                except ValueError: pass
            self.config[key] = val

        # Save location configuration
        if hasattr(self, 'loc_vars') and self.loc_vars:
            locations = {}
            for loc_key, fields in self.loc_vars.items():
                loc = {}
                for field, var in fields.items():
                    val = var.get().strip()
                    if field == "tempest_station":
                        try: val = int(val)
                        except ValueError: val = 0
                    elif field in ("lat", "lon"):
                        try: val = float(val)
                        except ValueError: val = 0.0
                    loc[field] = val
                # Only save locations that have a name and station ID
                if loc.get("name") and loc.get("tempest_station"):
                    locations[loc_key] = loc
            if locations:
                self.config["locations"] = locations

        save_config(self.config)
        self._init_oracle()
        self.set_status.set("✓ Saved! Restart app to apply location changes.")
        self.root.after(5000, lambda: self.set_status.set(""))

    def _test_tempest(self):
        token = self.setting_vars.get("tempest_api_token", tk.StringVar()).get()
        if not token:
            self.set_status.set("No token"); return
        def _do():
            from collectors.tempest import TempestCollector
            tc = TempestCollector(token)
            results = []
            for lk, lc in LOCATIONS.items():
                ok, msg = tc.test_connection(lc["tempest_station"])
                results.append(f"{lc['short']}: {msg}")
            self.root.after(0, lambda: self.set_status.set(" │ ".join(results)))
        threading.Thread(target=_do, daemon=True).start()

    def _test_ha(self):
        url = self.setting_vars.get("ha_url", tk.StringVar()).get()
        token = self.setting_vars.get("ha_token", tk.StringVar()).get()
        if not url or not token:
            self.set_status.set("Enter URL and token"); return
        def _do():
            from collectors.homeassistant import HACollector
            ha = HACollector(url, token)
            ok, msg = ha.test_connection()
            self.root.after(0, lambda: self.set_status.set(f"{'✓' if ok else '✗'} {msg}"))
        threading.Thread(target=_do, daemon=True).start()

    # ── Log Tab ───────────────────────────────────────────────────────────

    def _build_log_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Log  ")

        self.log_text = scrolledtext.ScrolledText(
            tab, bg="#0a0f14", fg=FG, font=("Consolas", 10),
            wrap="word", relief="flat", bd=0)
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(10, 4))

        btn = ttk.Frame(tab)
        btn.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Button(btn, text="Clear",
                  command=lambda: self.log_text.delete("1.0", "end")).pack(side="left")
        ttk.Button(btn, text="Force Cycle",
                  command=self._force_cycle).pack(side="left", padx=8)
        ttk.Button(btn, text="NWS Discussion",
                  command=self._show_discussion).pack(side="left", padx=8)

    def _show_discussion(self):
        if self.oracle and self.oracle.nws_discussion:
            win = tk.Toplevel(self.root)
            win.title("NWS Forecast Discussion — GYX (Gray, ME)")
            win.geometry("800x600")
            txt = scrolledtext.ScrolledText(win, font=("Consolas", 10),
                                            wrap="word")
            txt.pack(fill="both", expand=True)
            txt.insert("1.0", self.oracle.nws_discussion)
            txt.config(state="disabled")

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def _log_cb(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        def _up():
            self.log_text.insert("end", f"[{ts}] {msg}\n")
            self.log_text.see("end")
            self.status_var.set(msg)
        self.root.after(0, _up)

    def _force_cycle(self):
        if self.oracle:
            threading.Thread(target=self.oracle.run_cycle, daemon=True).start()

    def _init_oracle(self):
        if self.oracle:
            self.oracle.stop()
        self.oracle = WeatherOracle(self.config)
        self.oracle.callbacks.append(self._log_cb)

        if self.config.get("tempest_api_token"):
            self.oracle.start()
            self.status_var.set("Running — collecting data for both locations")
            self._schedule_refresh()
        else:
            self.status_var.set("Enter Tempest API token in Settings")

    def _schedule_refresh(self):
        if self.oracle and self.oracle._running:
            self._refresh_dashboard()
            self._refresh_forecasts()
            self.root.after(300_000, self._schedule_refresh)  # 5 min

    def _on_close(self):
        if self.oracle:
            self.oracle.stop()
        self.root.destroy()

    def run(self):
        if self.config.get("tempest_api_token"):
            self.root.after(500, self._init_oracle)
        self.root.mainloop()
