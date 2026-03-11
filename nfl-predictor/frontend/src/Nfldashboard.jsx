import { useState, useEffect, useCallback } from "react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  LineChart, Line, ResponsiveContainer, Cell
} from "recharts";

const API = "http://localhost:5000/api";

// ── Helpers ───────────────────────────────────────────────────────────────────

async function apiFetch(path) {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

function normalizeRadar(stats, allTeamStats) {
  if (!stats || Object.keys(allTeamStats).length === 0) return {};
  const keys = ["passing_yards", "rushing_yards", "yards_per_play", "points", "turnovers", "sacks_suffered"];
  const out = {};
  keys.forEach(k => {
    const vals = Object.values(allTeamStats).map(s => s[k]).filter(Boolean);
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const range = max - min || 1;
    // turnovers & sacks_suffered: lower is better → invert
    const invert = k === "turnovers" || k === "sacks_suffered";
    const norm = (stats[k] - min) / range;
    out[k] = Math.round((invert ? 1 - norm : norm) * 100);
  });
  return out;
}

// ── Sub-components ────────────────────────────────────────────────────────────

function StatCard({ label, value, sub, accent }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.04)",
      border: `1px solid ${accent || "rgba(255,180,50,0.25)"}`,
      borderRadius: 12,
      padding: "20px 24px",
      flex: 1,
      minWidth: 130,
    }}>
      <div style={{ fontSize: 11, letterSpacing: 2, color: "#888", textTransform: "uppercase", marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 32, fontFamily: "'Bebas Neue', sans-serif", color: accent || "#F5B942", letterSpacing: 1 }}>
        {value ?? <span style={{ fontSize: 18, color: "#555" }}>Loading…</span>}
      </div>
      {sub && <div style={{ fontSize: 11, color: "#666", marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function SectionTitle({ children }) {
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{
        fontFamily: "'Bebas Neue', sans-serif",
        fontSize: 22,
        letterSpacing: 3,
        color: "#F5B942",
        display: "flex",
        alignItems: "center",
        gap: 12,
      }}>
        <div style={{ width: 4, height: 22, background: "#F5B942", borderRadius: 2 }} />
        {children}
      </div>
    </div>
  );
}

function Spinner() {
  return (
    <div style={{ display: "flex", justifyContent: "center", padding: 40 }}>
      <div style={{
        width: 32, height: 32,
        border: "3px solid rgba(245,185,66,0.2)",
        borderTop: "3px solid #F5B942",
        borderRadius: "50%",
        animation: "spin 0.8s linear infinite",
      }} />
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────────

export default function NFLDashboard() {
  const [activeTab, setActiveTab]         = useState("overview");
  const [metrics, setMetrics]             = useState(null);
  const [coefs, setCoefs]                 = useState([]);
  const [homeWinRate, setHomeWinRate]     = useState([]);
  const [teams, setTeams]                 = useState([]);
  const [homeTeam, setHomeTeam]           = useState(null);
  const [awayTeam, setAwayTeam]           = useState(null);
  const [prediction, setPrediction]       = useState(null);
  const [allTeamStats, setAllTeamStats]   = useState({});
  const [predLoading, setPredLoading]     = useState(false);
  const [loadingStats, setLoadingStats]   = useState(false);
  const [error, setError]                 = useState(null);

  // Load overview data
  useEffect(() => {
    Promise.all([
      apiFetch("/metrics"),
      apiFetch("/coefficients"),
      apiFetch("/home-win-rate"),
      apiFetch("/teams"),
    ]).then(([m, c, h, t]) => {
      setMetrics(m);
      setCoefs(c);
      setHomeWinRate(h);
      setTeams(t);
      if (t.length >= 2) {
        setHomeTeam(t[0]);
        setAwayTeam(t[1]);
      }
    }).catch(e => setError(e.message));
  }, []);

  // Fetch team stats for radar chart whenever teams list loads
  useEffect(() => {
    if (teams.length === 0) return;
    setLoadingStats(true);
    Promise.all(
      teams.map(team =>
        apiFetch(`/team-stats?team=${team}&season=2023`).then(s => [team, s])
      )
    ).then(results => {
      const map = {};
      results.forEach(([team, stats]) => { map[team] = stats; });
      setAllTeamStats(map);
      setLoadingStats(false);
    }).catch(() => setLoadingStats(false));
  }, [teams]);

  // Predict whenever teams change
  useEffect(() => {
    if (!homeTeam || !awayTeam || homeTeam === awayTeam) return;
    setPredLoading(true);
    setPrediction(null);
    apiFetch(`/predict?home=${homeTeam}&away=${awayTeam}&week=18&season=2023`)
      .then(d => { setPrediction(d); setPredLoading(false); })
      .catch(e => { setError(e.message); setPredLoading(false); });
  }, [homeTeam, awayTeam]);

  // Build radar data
  const homeNorm = homeTeam ? normalizeRadar(allTeamStats[homeTeam], allTeamStats) : {};
  const awayNorm = awayTeam ? normalizeRadar(allTeamStats[awayTeam], allTeamStats) : {};
  const radarLabels = {
    passing_yards: "Passing", rushing_yards: "Rushing",
    yards_per_play: "Efficiency", points: "Scoring",
    turnovers: "Ball Security", sacks_suffered: "O-Line",
  };
  const radarData = Object.keys(radarLabels).map(k => ({
    subject: radarLabels[k],
    [homeTeam]: homeNorm[k] ?? 0,
    [awayTeam]: awayNorm[k] ?? 0,
  }));

  const cm  = metrics?.confusion_matrix;
  const prec = cm ? cm.tp / (cm.tp + cm.fp) : 0;
  const rec  = cm ? cm.tp / (cm.tp + cm.fn) : 0;
  const f1   = prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0;

  const winProb  = prediction?.home_win_prob ?? 0.5;
  const awayProb = prediction?.away_win_prob ?? 0.5;

  const tabs = ["overview", "predictor", "model"];

  if (error) return (
    <div style={{ minHeight: "100vh", background: "#0A0C10", color: "#F87171", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "monospace", fontSize: 14 }}>
      ⚠ API Error: {error}<br/>Make sure your Flask server is running at http://localhost:5000
    </div>
  );

  return (
    <div style={{ minHeight: "100vh", background: "#0A0C10", color: "#E8E8E8", fontFamily: "'DM Sans', sans-serif", paddingBottom: 60 }}>
      <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet" />

      {/* Header */}
      <div style={{
        background: "linear-gradient(135deg, #0D1117 0%, #151B26 100%)",
        borderBottom: "1px solid rgba(245,185,66,0.2)",
        padding: "28px 48px 24px",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "flex-end",
      }}>
        <div>
          <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 48, letterSpacing: 5, color: "#F5B942", lineHeight: 1 }}>
            GRIDIRON ML
          </div>
          <div style={{ fontSize: 13, color: "#666", letterSpacing: 2, marginTop: 6 }}>
            NFL GAME OUTCOME PREDICTOR · LOGISTIC REGRESSION MODEL
          </div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {tabs.map(t => (
            <button key={t} onClick={() => setActiveTab(t)} style={{
              background: activeTab === t ? "#F5B942" : "transparent",
              color: activeTab === t ? "#0A0C10" : "#888",
              border: `1px solid ${activeTab === t ? "#F5B942" : "rgba(255,255,255,0.1)"}`,
              borderRadius: 8, padding: "8px 20px",
              fontFamily: "'Bebas Neue', sans-serif", fontSize: 15, letterSpacing: 2,
              cursor: "pointer", transition: "all 0.2s",
            }}>
              {t.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      <div style={{ padding: "36px 48px"}}>

        {/* ── OVERVIEW ── */}
        {activeTab === "overview" && (
          <div>
            <div style={{ display: "flex", gap: 16, marginBottom: 40, flexWrap: "wrap" }}>
              <StatCard label="Accuracy"       value={metrics ? `${(metrics.accuracy * 100).toFixed(1)}%` : null} sub="Test set" />
              <StatCard label="ROC AUC"        value={metrics?.roc_auc} sub="Discriminative power" accent="#6EE7B7" />
              <StatCard label="Train Samples"  value={metrics?.train_samples?.toLocaleString()} sub="80% split" accent="#93C5FD" />
              <StatCard label="Test Samples"   value={metrics?.test_samples?.toLocaleString()} sub="20% split" accent="#C4B5FD" />
              <StatCard label="Seasons"        value={metrics?.seasons} sub="NFL regular season" accent="#FDA4AF" />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 28, marginBottom: 32 }}>
              <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16, padding: 28 }}>
                <SectionTitle>Feature Coefficients</SectionTitle>
                {coefs.length === 0 ? <Spinner /> : (
                  <ResponsiveContainer width="100%" height={320}>
                    <BarChart data={coefs} layout="vertical" margin={{ left: 10, right: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                      <XAxis type="number" tick={{ fill: "#666", fontSize: 11 }} tickLine={false} axisLine={false} />
                      <YAxis type="category" dataKey="feature" tick={{ fill: "#aaa", fontSize: 10 }} tickLine={false} axisLine={false} width={120} />
                      <Tooltip contentStyle={{ background: "#151B26", border: "1px solid #333", borderRadius: 8, fontSize: 12 }} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
                      <Bar dataKey="coef" radius={[0, 4, 4, 0]}>
                        {coefs.map((e, i) => <Cell key={i} fill={e.coef >= 0 ? "#F5B942" : "#F87171"} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                )}
                <div style={{ fontSize: 11, color: "#555", marginTop: 8 }}>Gold = positive influence · Red = negative</div>
              </div>

              <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16, padding: 28 }}>
                <SectionTitle>Home Win Rate by Week</SectionTitle>
                {homeWinRate.length === 0 ? <Spinner /> : (
                  <ResponsiveContainer width="100%" height={320}>
                    <LineChart data={homeWinRate}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="week" tick={{ fill: "#666", fontSize: 11 }} tickLine={false}
                        label={{ value: "Week", position: "insideBottom", offset: -2, fill: "#666", fontSize: 11 }} />
                      <YAxis domain={[0.4, 0.7]} tickFormatter={v => `${(v * 100).toFixed(0)}%`}
                        tick={{ fill: "#666", fontSize: 11 }} tickLine={false} axisLine={false} />
                      <Tooltip formatter={v => [`${(v * 100).toFixed(1)}%`, "Home Win Rate"]}
                        contentStyle={{ background: "#151B26", border: "1px solid #333", borderRadius: 8, fontSize: 12 }} />
                      <Line type="monotone" dataKey="rate" stroke="#F5B942" strokeWidth={2.5}
                        dot={{ fill: "#F5B942", r: 3 }} activeDot={{ r: 6 }} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
            </div>

            {/* Confusion matrix */}
            {cm && (
              <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16, padding: 28, display: "inline-flex", flexDirection: "column", gap: 16, minWidth: 380 }}>
                <SectionTitle>Confusion Matrix</SectionTitle>
                <div style={{ display: "grid", gridTemplateColumns: "auto 1fr 1fr", gap: 6, alignItems: "center" }}>
                  <div /><div style={{ textAlign: "center", fontSize: 11, color: "#888", letterSpacing: 1, paddingBottom: 8 }}>Pred: Win</div>
                  <div style={{ textAlign: "center", fontSize: 11, color: "#888", letterSpacing: 1, paddingBottom: 8 }}>Pred: Loss</div>
                  {[["Actual: Win", cm.tp, cm.fn, "#6EE7B7", "#F87171"], ["Actual: Loss", cm.fp, cm.tn, "#F87171", "#6EE7B7"]].map(([label, v1, v2, c1, c2]) => (
                    <>
                      <div key={label} style={{ fontSize: 11, color: "#888", paddingRight: 12 }}>{label}</div>
                      {[[v1, c1], [v2, c2]].map(([val, col], i) => (
                        <div key={i} style={{ background: `${col}18`, border: `1px solid ${col}44`, borderRadius: 10, padding: "16px 0", textAlign: "center", fontFamily: "'Bebas Neue', sans-serif", fontSize: 28, color: col }}>{val}</div>
                      ))}
                    </>
                  ))}
                </div>
                <div style={{ display: "flex", gap: 24 }}>
                  {[["Precision", prec], ["Recall", rec], ["F1 Score", f1]].map(([l, v]) => (
                    <div key={l}>
                      <div style={{ fontSize: 10, color: "#666", letterSpacing: 1 }}>{l}</div>
                      <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 22, color: "#F5B942" }}>{(v * 100).toFixed(1)}%</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── PREDICTOR ── */}
        {activeTab === "predictor" && (
          <div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 28, marginBottom: 28 }}>
              {[["HOME TEAM", homeTeam, setHomeTeam, "#F5B942"], ["AWAY TEAM", awayTeam, setAwayTeam, "#93C5FD"]].map(([label, val, setter, col]) => (
                <div key={label} style={{ background: "rgba(255,255,255,0.03)", border: `1px solid ${col}33`, borderRadius: 16, padding: 28 }}>
                  <div style={{ fontSize: 12, letterSpacing: 3, color: col, marginBottom: 16, fontWeight: 600 }}>{label}</div>
                  {teams.length === 0 ? <Spinner /> : (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                      {teams.map(t => (
                        <button key={t} onClick={() => setter(t)} style={{
                          background: val === t ? col : "rgba(255,255,255,0.04)",
                          color: val === t ? "#0A0C10" : "#aaa",
                          border: `1px solid ${val === t ? col : "rgba(255,255,255,0.1)"}`,
                          borderRadius: 8, padding: "8px 14px",
                          fontFamily: "'Bebas Neue', sans-serif", fontSize: 15, letterSpacing: 2,
                          cursor: "pointer", transition: "all 0.15s",
                          opacity: t === (label === "HOME TEAM" ? awayTeam : homeTeam) ? 0.3 : 1,
                        }}>{t}</button>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Win probability */}
            <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16, padding: 36, marginBottom: 28 }}>
              <SectionTitle>Win Probability</SectionTitle>
              {predLoading ? <Spinner /> : prediction ? (
                <>
                  <div style={{ display: "flex", alignItems: "center", gap: 24, marginBottom: 20 }}>
                    <div style={{ textAlign: "center", minWidth: 120 }}>
                      <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 56, color: "#F5B942", lineHeight: 1 }}>
                        {(winProb * 100).toFixed(1)}%
                      </div>
                      <div style={{ fontSize: 12, color: "#888", letterSpacing: 2 }}>{homeTeam} (HOME)</div>
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ height: 28, background: "rgba(255,255,255,0.06)", borderRadius: 14, overflow: "hidden", display: "flex" }}>
                        <div style={{ width: `${winProb * 100}%`, background: "linear-gradient(90deg,#F5B942,#F59E0B)", borderRadius: "14px 0 0 14px", transition: "width 0.6s cubic-bezier(0.16,1,0.3,1)" }} />
                        <div style={{ flex: 1, background: "linear-gradient(90deg,#3B82F6,#60A5FA)", borderRadius: "0 14px 14px 0" }} />
                      </div>
                    </div>
                    <div style={{ textAlign: "center", minWidth: 120 }}>
                      <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 56, color: "#93C5FD", lineHeight: 1 }}>
                        {(awayProb * 100).toFixed(1)}%
                      </div>
                      <div style={{ fontSize: 12, color: "#888", letterSpacing: 2 }}>{awayTeam} (AWAY)</div>
                    </div>
                  </div>

                  {/* Verdict */}
                  <div style={{ background: winProb > 0.5 ? "rgba(245,185,66,0.08)" : "rgba(147,197,253,0.08)", border: `1px solid ${winProb > 0.5 ? "rgba(245,185,66,0.3)" : "rgba(147,197,253,0.3)"}`, borderRadius: 12, padding: "16px 24px", textAlign: "center", marginBottom: 24 }}>
                    <span style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 20, letterSpacing: 3, color: winProb > 0.5 ? "#F5B942" : "#93C5FD" }}>
                      {winProb > 0.5 ? `${homeTeam} favored at home` : `${awayTeam} favored on the road`}
                      {Math.abs(winProb - 0.5) > 0.15 ? " · HIGH CONFIDENCE" : Math.abs(winProb - 0.5) > 0.07 ? " · MODERATE CONFIDENCE" : " · TOSS-UP"}
                    </span>
                  </div>

                  {/* Key stats comparison */}
                  {prediction.home_stats && (
                    <div>
                      <div style={{ fontSize: 12, letterSpacing: 2, color: "#666", marginBottom: 12 }}>ROLLING 5-GAME AVERAGES</div>
                      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 12 }}>
                        {[
                          ["Points", "points"],
                          ["Pass Yds", "pass_yards"],
                          ["Rush Yds", "rush_yards"],
                          ["Yds/Play", "yards_per_play"],
                          ["Turnovers", "turnovers"],
                        ].map(([label, key]) => (
                          <div key={key} style={{ background: "rgba(255,255,255,0.03)", borderRadius: 10, padding: "14px 16px", textAlign: "center" }}>
                            <div style={{ fontSize: 10, color: "#666", letterSpacing: 1, marginBottom: 8 }}>{label}</div>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                              <span style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 22, color: "#F5B942" }}>{prediction.home_stats[key]}</span>
                              <span style={{ fontSize: 10, color: "#555" }}>vs</span>
                              <span style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 22, color: "#93C5FD" }}>{prediction.away_stats[key]}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : <div style={{ color: "#666", fontSize: 14 }}>Select two different teams to see prediction</div>}
            </div>

            {/* Radar */}
            <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16, padding: 28 }}>
              <SectionTitle>Team Stats Radar</SectionTitle>
              {loadingStats ? <Spinner /> : (
                <ResponsiveContainer width="100%" height={360}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="rgba(255,255,255,0.1)" />
                    <PolarAngleAxis dataKey="subject" tick={{ fill: "#aaa", fontSize: 12 }} />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                    <Radar name={homeTeam} dataKey={homeTeam} stroke="#F5B942" fill="#F5B942" fillOpacity={0.2} strokeWidth={2} />
                    <Radar name={awayTeam} dataKey={awayTeam} stroke="#93C5FD" fill="#93C5FD" fillOpacity={0.2} strokeWidth={2} />
                    <Legend wrapperStyle={{ color: "#aaa", fontSize: 13 }} />
                    <Tooltip contentStyle={{ background: "#151B26", border: "1px solid #333", borderRadius: 8, fontSize: 12 }} />
                  </RadarChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        )}

        {/* ── MODEL ── */}
        {activeTab === "model" && (
          <div>
            <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16, padding: 32, marginBottom: 28 }}>
              <SectionTitle>How Predictions Work</SectionTitle>
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                {[
                  { step: "01", title: "Data Collection", desc: "3 seasons of NFL schedules and team stats loaded via nflreadpy, covering ~2,300 regular-season games.", col: "#F5B942" },
                  { step: "02", title: "Feature Engineering", desc: "Rolling 5-game averages computed per team (passing yards, rushing yards, turnovers, efficiency, points). All features expressed as home − away differences.", col: "#6EE7B7" },
                  { step: "03", title: "Model Training", desc: "L2-regularized Logistic Regression with StandardScaler normalization. Trained on 80% of data, evaluated on the remaining 20%.", col: "#93C5FD" },
                  { step: "04", title: "Live Prediction", desc: "Flask API serves predictions on demand. The React frontend calls /api/predict with team names and gets win probabilities back in real time.", col: "#C4B5FD" },
                ].map(({ step, title, desc, col }) => (
                  <div key={step} style={{ display: "flex", gap: 20, alignItems: "flex-start" }}>
                    <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 36, color: col, opacity: 0.4, lineHeight: 1, minWidth: 40 }}>{step}</div>
                    <div>
                      <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 18, letterSpacing: 2, color: col, marginBottom: 4 }}>{title}</div>
                      <div style={{ fontSize: 14, color: "#888", lineHeight: 1.6 }}>{desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(245,185,66,0.25)", borderRadius: 16, padding: 32 }}>
              <SectionTitle>Resume Bullet Points</SectionTitle>
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {[
                  "Built end-to-end NFL game outcome predictor in Python (pandas, scikit-learn) across 3 NFL seasons (~2,300 games)",
                  `Achieved ${metrics ? (metrics.accuracy * 100).toFixed(1) : "—"}% test accuracy and ${metrics?.roc_auc ?? "—"} ROC AUC using L2-regularized Logistic Regression with StandardScaler pipeline`,
                  "Engineered 10 predictive features including rolling 5-game averages for passing yards, turnovers, and scoring efficiency; prevented data leakage by rolling strictly on prior-week data",
                  "Built REST API with Flask serving real-time win probability predictions consumed by a React/Recharts dashboard",
                  "Designed interactive dashboard with live team radar comparisons, confusion matrix, and feature coefficient visualization",
                ].map((bullet, i) => (
                  <div key={i} style={{ display: "flex", gap: 14, alignItems: "flex-start" }}>
                    <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#F5B942", marginTop: 7, flexShrink: 0 }} />
                    <div style={{ fontSize: 14, color: "#ccc", lineHeight: 1.6 }}>{bullet}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}