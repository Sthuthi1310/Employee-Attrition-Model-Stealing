/**
 * Model Stealing Analysis Dashboard
 * Stable Chart.js rendering — single init, no scroll re-renders, no animations
 */

(function () {
  "use strict";

  const charts = {};
  let initialized = false;

  // Disable all Chart.js animations for stable rendering
  Chart.defaults.animation = false;
  Chart.defaults.animations = {};
  Chart.defaults.transitions = { active: { animation: { duration: 0 } } };
  Chart.defaults.responsive = true;
  Chart.defaults.maintainAspectRatio = false;
  Chart.defaults.color = "#94a3b8";
  Chart.defaults.borderColor = "#2a3548";
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.devicePixelRatio = 1; // Prevent DPI-related resize jitter

  const COLORS = {
    victim: "#3b82f6",
    surrogate: "#8b5cf6",
    cyan: "#22d3ee",
  };

  const CHART_OPTS = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: { legend: { labels: { boxWidth: 12, padding: 12 } } },
  };

  async function fetchAnalysisData() {
    const response = await fetch("/api/analysis", { cache: "no-store" });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${response.status}`);
    }
    return response.json();
  }

  function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  function populateOverview(data) {
    const o = data.overview;
    setText("original-model-name", o.original_model_name);
    setText("substitute-model-name", o.substitute_model_name);
    setText("total-queries", o.total_attack_queries.toLocaleString());
    setText("stolen-size", o.stolen_dataset_size.toLocaleString());
    setText("generated-at", new Date(data.generated_at).toLocaleString());
  }

  function populateMetricCards(data) {
    const container = document.getElementById("metric-cards");
    const metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"];
    const orig = data.performance.original;
    const sub = data.performance.substitute;

    container.innerHTML = metrics.map((metric) => {
      const oPct = ((orig[metric] ?? 0) * 100).toFixed(1);
      const sPct = ((sub[metric] ?? 0) * 100).toFixed(1);
      return `
        <div class="col-sm-6 col-lg">
          <div class="metric-compare-card h-100">
            <div class="metric-name">${metric}</div>
            <div class="metric-values">
              <div><span class="val-label">Original</span><span class="val-original">${oPct}%</span></div>
              <div class="text-end"><span class="val-label">Substitute</span><span class="val-substitute">${sPct}%</span></div>
            </div>
            <div class="metric-bar">
              <div class="bar-original" style="width:${oPct}%"></div>
              <div class="bar-substitute" style="width:${sPct}%"></div>
            </div>
          </div>
        </div>`;
    }).join("");
  }

  function populateAgreement(data) {
    const a = data.agreement;
    setText("agreement-rate", `${a.agreement_rate}%`);
    setText("prediction-similarity", `${a.prediction_similarity}%`);
    setText("matching-preds", a.matching_predictions.toLocaleString());
    setText("different-preds", a.different_predictions.toLocaleString());
  }

  function populateAttackAnalytics(data) {
    const atk = data.attack_analytics;
    setText("api-queries", atk.total_api_queries.toLocaleString());
    setText("success-queries", atk.successful_queries.toLocaleString());
    setText("failed-queries", atk.failed_queries.toLocaleString());
    setText("avg-response", atk.average_response_time_ms > 0 ? `${atk.average_response_time_ms} ms` : "N/A");
  }

  function populateConclusions(data) {
    const c = data.conclusions;
    const badge = document.getElementById("verdict-badge");
    badge.textContent = c.stealing_status;
    badge.className = `verdict-badge verdict-${c.status_class}`;
    setText("agreement-interpretation", c.agreement_interpretation);
    setText("performance-difference", c.performance_difference);
    setText("recommendation", c.recommendation);
  }

  function populateConfusionMatrices(data) {
    document.getElementById("cm-original").src = data.confusion_matrices.original;
    document.getElementById("cm-substitute").src = data.confusion_matrices.substitute;
  }

  function createChart(canvasId, config) {
    if (charts[canvasId]) return charts[canvasId];
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    charts[canvasId] = new Chart(ctx, config);
    return charts[canvasId];
  }

  function renderAllCharts(data) {
    if (initialized) return;
    initialized = true;

    const orig = data.performance.original;
    const sub = data.performance.substitute;

    createChart("accuracyChart", {
      type: "bar",
      data: {
        labels: ["Accuracy"],
        datasets: [
          { label: "Original", data: [orig.Accuracy || 0], backgroundColor: COLORS.victim, borderRadius: 4 },
          { label: "Substitute", data: [sub.Accuracy || 0], backgroundColor: COLORS.surrogate, borderRadius: 4 },
        ],
      },
      options: {
        ...CHART_OPTS,
        scales: {
          y: { beginAtZero: true, max: 1.05, ticks: { callback: (v) => `${(v * 100).toFixed(0)}%` }, grid: { color: "#2a3548" } },
          x: { grid: { display: false } },
        },
      },
    });

    createChart("prfChart", {
      type: "bar",
      data: {
        labels: ["Precision", "Recall", "F1-Score"],
        datasets: [
          { label: "Original", data: [orig.Precision, orig.Recall, orig["F1-Score"]], backgroundColor: COLORS.victim, borderRadius: 3 },
          { label: "Substitute", data: [sub.Precision, sub.Recall, sub["F1-Score"]], backgroundColor: COLORS.surrogate, borderRadius: 3 },
        ],
      },
      options: {
        ...CHART_OPTS,
        scales: {
          y: { beginAtZero: true, max: 1.05, ticks: { callback: (v) => `${(v * 100).toFixed(0)}%` }, grid: { color: "#2a3548" } },
          x: { grid: { display: false } },
        },
      },
    });

    createChart("rocChart", {
      type: "bar",
      data: {
        labels: ["ROC-AUC"],
        datasets: [
          { label: "Original", data: [orig["ROC-AUC"] || 0], backgroundColor: COLORS.victim, borderRadius: 4 },
          { label: "Substitute", data: [sub["ROC-AUC"] || 0], backgroundColor: COLORS.surrogate, borderRadius: 4 },
        ],
      },
      options: {
        ...CHART_OPTS,
        scales: {
          y: { beginAtZero: true, max: 1.05, ticks: { callback: (v) => `${(v * 100).toFixed(0)}%` }, grid: { color: "#2a3548" } },
          x: { grid: { display: false } },
        },
      },
    });

    const cd = data.class_distribution;
    const origLabels = cd.original_labels || { Stay: 0, Leave: 0 };
    const subLabels = cd.substitute_labels || { Stay: 0, Leave: 0 };

    createChart("classDistChart", {
      type: "bar",
      data: {
        labels: ["Stay", "Leave"],
        datasets: [
          { label: "Teacher Labels (API)", data: [origLabels.Stay, origLabels.Leave], backgroundColor: COLORS.victim, borderRadius: 3 },
          { label: "Substitute Predictions", data: [subLabels.Stay, subLabels.Leave], backgroundColor: COLORS.surrogate, borderRadius: 3 },
        ],
      },
      options: {
        ...CHART_OPTS,
        scales: {
          y: { beginAtZero: true, grid: { color: "#2a3548" } },
          x: { grid: { display: false } },
        },
      },
    });

    const progression = data.agreement.query_progression || [];
    // Downsample for large datasets to keep chart performant
    const step = progression.length > 100 ? Math.ceil(progression.length / 100) : 1;
    const sampled = progression.filter((_, i) => i % step === 0 || i === progression.length - 1);

    createChart("progressionChart", {
      type: "line",
      data: {
        labels: sampled.map((p) => p.queries),
        datasets: [{
          label: "Agreement Rate (%)",
          data: sampled.map((p) => p.agreement_rate),
          borderColor: COLORS.cyan,
          backgroundColor: "rgba(34,211,238,0.08)",
          fill: true,
          tension: 0,
          pointRadius: 0,
          borderWidth: 2,
        }],
      },
      options: {
        ...CHART_OPTS,
        plugins: { legend: { display: false } },
        scales: {
          x: { title: { display: true, text: "Number of Queries" }, grid: { color: "#2a3548" } },
          y: { min: 0, max: 100, title: { display: true, text: "Agreement Rate (%)" }, grid: { color: "#2a3548" } },
        },
      },
    });

    const rate = data.agreement.agreement_rate || 0;
    createChart("agreementGauge", {
      type: "doughnut",
      data: {
        labels: ["Agreement", "Disagreement"],
        datasets: [{ data: [rate, 100 - rate], backgroundColor: [COLORS.cyan, "#2a3548"], borderWidth: 0 }],
      },
      options: {
        ...CHART_OPTS,
        circumference: 270,
        rotation: 225,
        cutout: "75%",
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
      },
    });

    renderFIChart("fiOriginalChart", data.feature_importance.original, COLORS.victim, "fiOriginalFallback");
    renderFIChart("fiSubstituteChart", data.feature_importance.substitute, COLORS.surrogate, "fiSubstituteFallback");
  }

  function normalizeFeatureImportanceData(data) {
    if (!data.feature_importance) {
      data.feature_importance = { original: [], substitute: [] };
    }

    if (
      data.feature_importance.substitute &&
      !data.feature_importance.substitute.length &&
      Array.isArray(data.substitute_features) &&
      Array.isArray(data.substitute_importances)
    ) {
      const reconstructed = [];
      const count = Math.min(data.substitute_features.length, data.substitute_importances.length);
      for (let i = 0; i < count; i += 1) {
        reconstructed.push({ feature: data.substitute_features[i], importance: Number(data.substitute_importances[i]) });
      }
      data.feature_importance.substitute = reconstructed;
    }

    return data;
  }

  function isFeatureImportanceValid(features) {
    if (!features || !features.length) return false;
    return features.some((item) => Number.isFinite(item.importance) && item.importance > 0);
  }

  function renderFIChart(canvasId, features, color, fallbackId) {
    const fallback = document.getElementById(fallbackId);
    if (!features || !features.length || !isFeatureImportanceValid(features)) {
      if (fallback) {
        fallback.style.display = "block";
      }
      return;
    }

    if (fallback) {
      fallback.style.display = "none";
    }

    const labels = features.map((f) => f.feature).reverse();
    const values = features.map((f) => f.importance).reverse();

    createChart(canvasId, {
      type: "bar",
      data: { labels, datasets: [{ label: "Importance", data: values, backgroundColor: color, borderRadius: 3 }] },
      options: {
        ...CHART_OPTS,
        indexAxis: "y",
        plugins: { legend: { display: false } },
        scales: {
          x: { beginAtZero: true, grid: { color: "#2a3548" } },
          y: { grid: { display: false } },
        },
      },
    });
  }

  async function loadDashboard() {
    const overlay = document.getElementById("loading-overlay");
    const main = document.getElementById("dashboard-main");

    try {
      const data = await fetchAnalysisData();
      console.debug("Analysis payload:", data);
      normalizeFeatureImportanceData(data);
      populateOverview(data);
      populateMetricCards(data);
      populateAgreement(data);
      populateAttackAnalytics(data);
      populateConclusions(data);
      populateConfusionMatrices(data);

      overlay.classList.add("hidden");
      main.style.display = "block";

      // Render charts once after layout is visible
      requestAnimationFrame(() => renderAllCharts(data));
    } catch (error) {
      console.error("Failed to load analysis data", error);
      overlay.innerHTML = `
        <div class="text-center px-4">
          <p class="text-danger fw-bold">Failed to Load Analysis Data</p>
          <p class="text-secondary">${error.message}</p>
          <p class="text-muted small">Run: python attack/run_attack_pipeline.py --count 500</p>
          <a href="/" class="btn btn-outline-info btn-sm mt-3">Back to Prediction</a>
        </div>`;
    }
  }

  // Refresh button support
  document.addEventListener("DOMContentLoaded", () => {
    loadDashboard();
    const refreshBtn = document.getElementById("refresh-data");
    if (refreshBtn) {
      refreshBtn.addEventListener("click", () => {
        initialized = false;
        Object.keys(charts).forEach((k) => { charts[k].destroy(); delete charts[k]; });
        document.getElementById("loading-overlay").classList.remove("hidden");
        document.getElementById("dashboard-main").style.display = "none";
        loadDashboard();
      });
    }
  });
})();

