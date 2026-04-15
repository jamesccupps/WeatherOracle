// ============================================================
// WeatherOracle Lovelace Card v1.0
// Drop in: /config/www/weatheroracle-card.js
//
// Dashboard YAML:
//   type: custom:weatheroracle-card
//   show_forecast: true    # optional, default true
//   show_daily: true       # optional, default true
// ============================================================

const WMO_ICON = {
  0:"☀️",1:"🌤",2:"⛅",3:"☁️",45:"🌫",48:"🌫",
  51:"🌦",53:"🌧",55:"🌧",56:"🌧",57:"🌧",
  61:"🌦",63:"🌧",65:"🌧",66:"🌧",67:"🌧",
  71:"🌨",73:"❄️",75:"❄️",77:"❄️",80:"🌦",81:"🌧",82:"⛈",
  85:"🌨",86:"🌨",95:"⛈",96:"⛈",99:"⛈"
};
const WMO = {
  0:"Clear",1:"Mainly Clear",2:"Partly Cloudy",3:"Overcast",
  45:"Fog",48:"Rime Fog",51:"Lt Drizzle",53:"Drizzle",55:"Hvy Drizzle",
  61:"Lt Rain",63:"Rain",65:"Heavy Rain",66:"Frzg Rain",67:"Hvy Frzg Rain",
  71:"Lt Snow",73:"Snow",75:"Heavy Snow",80:"Lt Showers",81:"Showers",82:"Heavy Showers",
  85:"Lt Snow Shwrs",86:"Snow Showers",95:"Thunderstorm",96:"T-Storm+Hail",99:"Severe T-Storm"
};

class WeatherOracleCard extends HTMLElement {
  set hass(hass) {
    if (!this.content) {
      this._setupCard();
    }
    this._hass = hass;
    this._update();
  }

  setConfig(config) {
    this._config = {
      show_forecast: true,
      show_daily: true,
      ...config,
    };
  }

  _setupCard() {
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.innerHTML = `
      <style>${this._getStyles()}</style>
      <ha-card>
        <div class="wo-container">
          <div class="wo-header">
            <span class="wo-title">WeatherOracle</span>
            <span class="wo-sub">ML-Powered Hyperlocal</span>
            <span class="wo-status" id="status"></span>
          </div>
          <div class="wo-tabs">
            <button class="wo-tab active" data-tab="current">Current</button>
            <button class="wo-tab" data-tab="daily">3-Day</button>
            <button class="wo-tab" data-tab="hourly">Hourly</button>
          </div>
          <div class="wo-panel active" id="panel-current">
            <div class="wo-dual">
              <div class="wo-card apt" id="cur-apt"></div>
              <div class="wo-card occ" id="cur-occ"></div>
            </div>
          </div>
          <div class="wo-panel" id="panel-daily">
            <div class="wo-section-title apt-color">Apartment — 3-Day Outlook</div>
            <div class="wo-daily-grid" id="daily-apt"></div>
            <div class="wo-section-title occ-color" style="margin-top:16px">Location 2 — 3-Day Outlook</div>
            <div class="wo-daily-grid" id="daily-occ"></div>
          </div>
          <div class="wo-panel" id="panel-hourly">
            <div class="wo-section-title apt-color">Apartment — Hourly Forecast</div>
            <div class="wo-scroll" id="hourly-apt"></div>
            <div class="wo-section-title occ-color" style="margin-top:16px">Location 2 — Hourly Forecast</div>
            <div class="wo-scroll" id="hourly-occ"></div>
          </div>
        </div>
      </ha-card>
    `;
    this.content = this.shadowRoot.querySelector('.wo-container');

    // Tab switching
    this.shadowRoot.querySelectorAll('.wo-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        this.shadowRoot.querySelectorAll('.wo-tab').forEach(t => t.classList.remove('active'));
        this.shadowRoot.querySelectorAll('.wo-panel').forEach(p => p.classList.remove('active'));
        tab.classList.add('active');
        this.shadowRoot.getElementById('panel-' + tab.dataset.tab).classList.add('active');
      });
    });
  }

  _update() {
    if (!this._hass) return;

    // Current conditions
    for (const [loc, prefix] of [['apartment', 'apt'], ['occ', 'occ']]) {
      const entity = this._hass.states[`sensor.weatheroracle_${prefix}_current`];
      const el = this.shadowRoot.getElementById(`cur-${prefix}`);
      if (el && entity) {
        this._renderCurrent(el, entity, loc, prefix);
      } else if (el) {
        el.innerHTML = '<div class="wo-empty">Waiting for data...</div>';
      }
    }

    // Daily forecast
    for (const [loc, prefix] of [['apartment', 'apt'], ['occ', 'occ']]) {
      const entity = this._hass.states[`sensor.weatheroracle_${prefix}_daily`];
      const el = this.shadowRoot.getElementById(`daily-${prefix}`);
      if (el && entity) {
        this._renderDaily(el, entity);
      } else if (el) {
        el.innerHTML = '<div class="wo-empty">Waiting for forecast...</div>';
      }
    }

    // Hourly forecast
    for (const [loc, prefix] of [['apartment', 'apt'], ['occ', 'occ']]) {
      const entity = this._hass.states[`sensor.weatheroracle_${prefix}_forecast`];
      const el = this.shadowRoot.getElementById(`hourly-${prefix}`);
      if (el && entity) {
        this._renderHourly(el, entity);
      } else if (el) {
        el.innerHTML = '<div class="wo-empty">Waiting for forecast...</div>';
      }
    }

    // Status
    const statusEntity = this._hass.states['sensor.weatheroracle_status'];
    const statusEl = this.shadowRoot.getElementById('status');
    if (statusEl && statusEntity) {
      const ml = statusEntity.attributes.ml_models_total || 0;
      statusEl.textContent = ml > 0 ? `✓ ${ml} ML models` : 'Collecting...';
      statusEl.className = 'wo-status ' + (ml > 0 ? 'active' : '');
    }
  }

  _renderCurrent(el, entity, loc, prefix) {
    const a = entity.attributes;
    const temp = entity.state;
    const locName = loc;
    const stationId = a.tempest_station || '';

    let xcheckHtml = '';
    if (a.xcheck_yolink_temp != null || a.xcheck_outdoor_sensor_temp != null) {
      const parts = [];
      for (const src of ['yolink', 'outdoor_sensor', 'ha_average']) {
        const t = a[`xcheck_${src}_temp`];
        if (t != null) {
          const delta = Math.abs(t - parseFloat(temp)).toFixed(1);
          const cls = delta < 2 ? 'good' : 'warn';
          parts.push(`<span class="xc-src">${src}:</span> ${t}°F <span class="xc-delta ${cls}">Δ${delta}°</span>`);
        }
      }
      if (parts.length) {
        xcheckHtml = `<div class="wo-xcheck">${parts.join(' │ ')}</div>`;
      }
    }

    el.innerHTML = `
      <div class="wo-card-title ${prefix}-color">${locName}</div>
      <div class="wo-current-grid">
        <div class="wo-big-temp ${prefix}-color">${Math.round(parseFloat(temp))}°</div>
        <div class="wo-feels">Feels like ${a.feels_like_f || '--'}°F</div>
      </div>
      <div class="wo-metrics">
        <div class="wo-metric"><div class="wo-val">${a.humidity || '--'}%</div><div class="wo-lbl">Humidity</div></div>
        <div class="wo-metric"><div class="wo-val">${a.dewpoint_f || '--'}°</div><div class="wo-lbl">Dewpoint</div></div>
        <div class="wo-metric"><div class="wo-val">${a.wind_mph || '0'}</div><div class="wo-lbl">Wind mph</div></div>
        <div class="wo-metric"><div class="wo-val">${a.pressure_mb || '--'}</div><div class="wo-lbl">Pressure</div></div>
      </div>
      ${xcheckHtml}
      <div class="wo-ts">Tempest #${stationId} • ${(a.timestamp || '').slice(0,19)}</div>
    `;
  }

  _renderDaily(el, entity) {
    let days;
    try { days = JSON.parse(entity.attributes.days || '[]'); } catch { days = []; }
    if (!days.length) { el.innerHTML = '<div class="wo-empty">No daily data</div>'; return; }

    el.innerHTML = days.map(d => `
      <div class="wo-day">
        <div class="wo-day-name">${d.day}</div>
        <div class="wo-day-date">${d.label}</div>
        <div class="wo-day-icon">${d.icon || '—'}</div>
        <div class="wo-day-cond">${d.condition || ''}</div>
        <div class="wo-day-temps">
          <span class="wo-hi ${this._tempCls(d.hi)}">▲ ${d.hi || '--'}°</span>
          <span class="wo-lo">▼ ${d.lo || '--'}°</span>
        </div>
        <div class="wo-day-details">
          <span>💧 ${d.hum || '--'}%</span>
          <span>💨 ${d.wind || '--'} mph</span>
          <span>🌧 ${d.precip || '0'}%</span>
        </div>
      </div>
    `).join('');
  }

  _renderHourly(el, entity) {
    let hours;
    try { hours = JSON.parse(entity.attributes.hours || '[]'); } catch { hours = []; }
    if (!hours.length) { el.innerHTML = '<div class="wo-empty">No forecast data</div>'; return; }

    let html = `<table class="wo-fc-table"><thead><tr>
      <th>Time</th><th>Temp</th><th>RH</th><th>Wind</th><th>Precip</th><th>Sky</th><th>Conf</th>
    </tr></thead><tbody>`;

    for (const h of hours.slice(0, 48)) {
      const wc = h.wc;
      const icon = WMO_ICON[wc] || '';
      const conf = h.conf || 0;
      const confCls = conf >= 70 ? 'hi' : conf >= 40 ? 'md' : 'lo';
      html += `<tr>
        <td class="time-col">${this._fmtTime(h.t)}</td>
        <td class="${this._tempCls(h.temp)}">${h.temp != null ? h.temp + '°' : '—'}</td>
        <td>${h.hum != null ? Math.round(h.hum) : '—'}</td>
        <td>${h.wind != null ? h.wind : '—'}</td>
        <td class="precip-col">${h.pp != null ? Math.round(h.pp) + '%' : '—'}</td>
        <td class="sky-col">${icon}</td>
        <td><span class="wo-conf ${confCls}">${Math.round(conf)}</span></td>
      </tr>`;
    }
    html += '</tbody></table>';
    el.innerHTML = html;
  }

  _tempCls(t) {
    if (t == null) return '';
    if (t <= 20) return 'tc-cold';
    if (t <= 40) return 'tc-cool';
    if (t <= 70) return 'tc-mild';
    if (t <= 85) return 'tc-warm';
    return 'tc-hot';
  }

  _fmtTime(iso) {
    try {
      const d = new Date(iso);
      const days = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
      let h = d.getHours();
      const ampm = h >= 12 ? 'PM' : 'AM';
      h = h % 12 || 12;
      return `${days[d.getDay()]} ${h}${ampm}`;
    } catch { return iso || ''; }
  }

  _getStyles() {
    return `
      :host { display: block; }
      ha-card { background: var(--card-background-color, #1c1c1c); }
      .wo-container { padding: 16px; font-family: var(--paper-font-body1_-_font-family, sans-serif); }
      .wo-header { display: flex; align-items: baseline; gap: 10px; margin-bottom: 14px; }
      .wo-title { font-size: 18px; font-weight: 700;
        background: linear-gradient(135deg, #34d399, #60a5fa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
      .wo-sub { font-size: 11px; color: var(--secondary-text-color); }
      .wo-status { margin-left: auto; font-size: 11px; color: var(--secondary-text-color); }
      .wo-status.active { color: #34d399; }

      .wo-tabs { display: flex; gap: 4px; margin-bottom: 14px; }
      .wo-tab {
        padding: 6px 14px; border-radius: 6px; border: 1px solid var(--divider-color, #333);
        background: transparent; color: var(--secondary-text-color); cursor: pointer;
        font-size: 12px; font-weight: 500; }
      .wo-tab.active { background: var(--primary-color, #4a90d9); color: #fff;
        border-color: var(--primary-color); }

      .wo-panel { display: none; }
      .wo-panel.active { display: block; }

      .wo-dual { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
      @media (max-width: 600px) { .wo-dual { grid-template-columns: 1fr; } }

      .wo-card {
        background: var(--ha-card-background, rgba(0,0,0,0.2));
        border: 1px solid var(--divider-color, #333);
        border-radius: 10px; padding: 14px;
        position: relative; overflow: hidden; }
      .wo-card.apt::before, .wo-card.occ::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; }
      .wo-card.apt::before { background: #34d399; }
      .wo-card.occ::before { background: #60a5fa; }

      .wo-card-title { font-size: 11px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1px; margin-bottom: 12px; }
      .apt-color { color: #34d399; }
      .occ-color { color: #60a5fa; }

      .wo-current-grid { display: flex; align-items: baseline; gap: 12px; }
      .wo-big-temp { font-size: 48px; font-weight: 700; line-height: 1; }
      .wo-feels { font-size: 12px; color: var(--secondary-text-color); }

      .wo-metrics {
        display: grid; grid-template-columns: repeat(4, 1fr);
        gap: 8px; margin-top: 14px; padding-top: 12px;
        border-top: 1px solid var(--divider-color, #333); }
      .wo-metric { text-align: center; }
      .wo-val { font-size: 16px; font-weight: 700; }
      .wo-lbl { font-size: 9px; color: var(--secondary-text-color);
        text-transform: uppercase; letter-spacing: 0.5px; margin-top: 1px; }

      .wo-xcheck {
        margin-top: 10px; padding: 6px 10px; border-radius: 6px;
        background: rgba(0,0,0,0.2); font-size: 11px;
        color: var(--secondary-text-color); }
      .xc-src { color: var(--primary-text-color); }
      .xc-delta { padding: 1px 5px; border-radius: 3px; font-size: 10px; font-weight: 600; }
      .xc-delta.good { background: rgba(52,211,153,0.2); color: #34d399; }
      .xc-delta.warn { background: rgba(248,113,113,0.2); color: #f87171; }

      .wo-ts { margin-top: 10px; font-size: 10px; color: var(--secondary-text-color); }

      .wo-section-title { font-size: 12px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1px; margin-bottom: 10px; }

      /* Daily */
      .wo-daily-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
      .wo-day {
        background: rgba(0,0,0,0.2); border: 1px solid var(--divider-color, #333);
        border-radius: 10px; padding: 12px; text-align: center; }
      .wo-day-name { font-size: 13px; font-weight: 700; }
      .wo-day-date { font-size: 10px; color: var(--secondary-text-color); margin-bottom: 6px; }
      .wo-day-icon { font-size: 28px; margin-bottom: 4px; }
      .wo-day-cond { font-size: 10px; color: var(--secondary-text-color);
        margin-bottom: 8px; min-height: 14px; }
      .wo-day-temps { display: flex; justify-content: center; gap: 12px;
        font-size: 18px; font-weight: 700; margin-bottom: 8px; }
      .wo-hi { }
      .wo-lo { color: var(--secondary-text-color); }
      .wo-day-details { display: flex; justify-content: center; gap: 10px;
        font-size: 10px; color: var(--secondary-text-color); }

      /* Hourly table */
      .wo-scroll { max-height: 360px; overflow-y: auto; }
      .wo-fc-table { width: 100%; border-collapse: collapse; font-size: 12px; }
      .wo-fc-table th {
        text-align: center; padding: 6px 4px; font-size: 9px;
        text-transform: uppercase; letter-spacing: 0.5px;
        color: var(--secondary-text-color);
        border-bottom: 1px solid var(--divider-color, #333); font-weight: 500; }
      .wo-fc-table td {
        text-align: center; padding: 5px 4px;
        border-bottom: 1px solid rgba(255,255,255,0.05); }
      .wo-fc-table .time-col { text-align: left; color: var(--secondary-text-color); }
      .wo-fc-table .precip-col { color: #818cf8; }
      .wo-fc-table .sky-col { font-size: 14px; }

      .wo-conf { padding: 1px 5px; border-radius: 3px; font-size: 10px; font-weight: 700; }
      .wo-conf.hi { background: rgba(52,211,153,0.15); color: #34d399; }
      .wo-conf.md { background: rgba(251,191,36,0.15); color: #fbbf24; }
      .wo-conf.lo { background: rgba(248,113,113,0.15); color: #f87171; }

      .tc-cold { color: #38bdf8; }
      .tc-cool { color: #67e8f9; }
      .tc-mild { color: var(--primary-text-color); }
      .tc-warm { color: #fb923c; font-weight: 700; }
      .tc-hot { color: #ef4444; font-weight: 700; }

      .wo-empty { color: var(--secondary-text-color); font-size: 13px;
        text-align: center; padding: 20px; }
    `;
  }

  getCardSize() { return 6; }

  static getStubConfig() { return {}; }
}

if (!customElements.get('weatheroracle-card')) {
  customElements.define('weatheroracle-card', WeatherOracleCard);
}

window.customCards = window.customCards || [];
window.customCards.push({
  type: 'weatheroracle-card',
  name: 'WeatherOracle',
  description: 'ML-powered hyperlocal weather for Southern Maine',
});
