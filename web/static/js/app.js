/**
 * Employee Attrition Predictor - Frontend Application
 * ====================================================
 * Handles form validation, API communication, dashboard updates,
 * and result display for the HR analytics web interface.
 */

(function () {
  "use strict";

  // -------------------------------------------------------------------------
  // Configuration
  // -------------------------------------------------------------------------

  const API = {
    predict: "/api/predict",
    stats: "/api/stats",
    options: "/api/options",
    health: "/api/health",
  };

  /** Validation rules matching backend constraints */
  const VALIDATION_RULES = {
    Age: { min: 18, max: 70, label: "Age", type: "number" },
    Monthly_Income: { min: 1000, max: 25000, label: "Monthly Income", type: "number" },
    Years_at_Company: { min: 0, max: 40, label: "Years at Company", type: "number" },
    Job_Satisfaction: { min: 1, max: 4, label: "Job Satisfaction", type: "select" },
    Work_Life_Balance: { min: 1, max: 4, label: "Work-Life Balance", type: "select" },
    Number_of_Companies_Worked: { min: 0, max: 10, label: "Number of Companies Worked", type: "number" },
    Gender: { label: "Gender", type: "select" },
    Department: { label: "Department", type: "select" },
    Job_Role: { label: "Job Role", type: "select" },
    Overtime: { label: "Overtime", type: "select" },
  };

  // -------------------------------------------------------------------------
  // DOM References
  // -------------------------------------------------------------------------

  const form = document.getElementById("attrition-form");
  const predictBtn = document.getElementById("predict-btn");
  const resetBtn = document.getElementById("reset-form");
  const refreshStatsBtn = document.getElementById("refresh-stats");

  const resultPlaceholder = document.getElementById("result-placeholder");
  const resultLoading = document.getElementById("result-loading");
  const resultCard = document.getElementById("result-card");

  const statTotal = document.getElementById("stat-total");
  const statAvgRisk = document.getElementById("stat-avg-risk");
  const statModel = document.getElementById("stat-model");
  const recentTableBody = document.getElementById("recent-table-body");

  // -------------------------------------------------------------------------
  // Utility Functions
  // -------------------------------------------------------------------------

  /**
   * Map form field names to HTML element IDs.
   */
  function fieldToId(fieldName) {
    const map = {
      Age: "age",
      Gender: "gender",
      Department: "department",
      Job_Role: "job-role",
      Monthly_Income: "income",
      Years_at_Company: "years",
      Job_Satisfaction: "satisfaction",
      Work_Life_Balance: "balance",
      Overtime: "overtime",
      Number_of_Companies_Worked: "companies",
    };
    return map[fieldName] || fieldName.toLowerCase();
  }

  /**
   * Format ISO timestamp for display in the table.
   */
  function formatTimestamp(isoString) {
    try {
      const date = new Date(isoString);
      return date.toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return isoString;
    }
  }

  /**
   * Show one of the three result panel states.
   */
  function showResultState(state) {
    resultPlaceholder.classList.toggle("hidden", state !== "placeholder");
    resultLoading.classList.toggle("hidden", state !== "loading");
    resultCard.classList.toggle("hidden", state !== "result");
  }

  /**
   * Set button loading state.
   */
  function setButtonLoading(isLoading) {
    predictBtn.disabled = isLoading;
    predictBtn.classList.toggle("btn--loading", isLoading);
    predictBtn.setAttribute("aria-busy", isLoading ? "true" : "false");
  }

  // -------------------------------------------------------------------------
  // Form Validation
  // -------------------------------------------------------------------------

  /**
   * Validate a single form field and update UI error state.
   * @returns {boolean} True if valid
   */
  function validateField(fieldName) {
    const id = fieldToId(fieldName);
    const input = document.getElementById(id);
    const errorEl = document.getElementById(`${id}-error`);
    const rules = VALIDATION_RULES[fieldName];

    if (!input || !rules) return true;

    const value = input.value.trim();
    let error = "";

    if (!value) {
      error = `${rules.label} is required.`;
    } else if (rules.type === "number") {
      const num = Number(value);
      if (!Number.isInteger(num)) {
        error = `${rules.label} must be a whole number.`;
      } else if (num < rules.min || num > rules.max) {
        error = `${rules.label} must be between ${rules.min.toLocaleString()} and ${rules.max.toLocaleString()}.`;
      }
    }

    input.classList.toggle("is-invalid", Boolean(error));
    input.setAttribute("aria-invalid", error ? "true" : "false");
    if (errorEl) errorEl.textContent = error;

    return !error;
  }

  /**
   * Validate all form fields before submission.
   * @returns {boolean} True if entire form is valid
   */
  function validateForm() {
    let isValid = true;
    Object.keys(VALIDATION_RULES).forEach((field) => {
      if (!validateField(field)) isValid = false;
    });
    return isValid;
  }

  /**
   * Clear all validation error states.
   */
  function clearValidationErrors() {
    Object.keys(VALIDATION_RULES).forEach((field) => {
      const id = fieldToId(field);
      const input = document.getElementById(id);
      const errorEl = document.getElementById(`${id}-error`);
      if (input) {
        input.classList.remove("is-invalid");
        input.setAttribute("aria-invalid", "false");
      }
      if (errorEl) errorEl.textContent = "";
    });
  }

  /**
   * Collect form data as JSON payload for the API.
   */
  function getFormData() {
    const formData = new FormData(form);
    const payload = {};

    for (const [key, value] of formData.entries()) {
      const numericFields = [
        "Age",
        "Monthly_Income",
        "Years_at_Company",
        "Job_Satisfaction",
        "Work_Life_Balance",
        "Number_of_Companies_Worked",
      ];
      payload[key] = numericFields.includes(key) ? parseInt(value, 10) : value;
    }

    return payload;
  }

  // -------------------------------------------------------------------------
  // API Communication
  // -------------------------------------------------------------------------

  /**
   * Fetch dropdown options from the backend.
   */
  async function loadFormOptions() {
    try {
      const response = await fetch(API.options);
      if (!response.ok) throw new Error("Failed to load options");

      const options = await response.json();

      populateSelect("gender", options.Gender);
      populateSelect("department", options.Department);
      populateSelect("job-role", options.Job_Role);
      populateSelect("overtime", options.Overtime);
    } catch (error) {
      console.error("Options load error:", error);
    }
  }

  /**
   * Populate a select element with options.
   */
  function populateSelect(elementId, values) {
    const select = document.getElementById(elementId);
    if (!select || !values) return;

    const placeholder = select.querySelector("option[value='']");
    select.innerHTML = "";
    if (placeholder) select.appendChild(placeholder);

    values.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      select.appendChild(option);
    });
  }

  /**
   * Send prediction request to backend API.
   */
  async function submitPrediction(payload) {
    const response = await fetch(API.predict, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction request failed.");
    }

    return data;
  }

  /**
   * Fetch dashboard statistics from backend.
   */
  async function loadDashboardStats() {
    try {
      const response = await fetch(API.stats);
      if (!response.ok) throw new Error("Failed to load stats");

      const data = await response.json();
      updateDashboard(data);
    } catch (error) {
      console.error("Stats load error:", error);
    }
  }

  /**
   * Check API health on startup.
   */
  async function checkHealth() {
    try {
      const response = await fetch(API.health);
      const data = await response.json();
      statModel.textContent = data.model_loaded ? "Active" : "Unavailable";
      statModel.style.color = data.model_loaded ? "" : "var(--color-danger)";
    } catch {
      statModel.textContent = "Offline";
      statModel.style.color = "var(--color-danger)";
    }
  }

  // -------------------------------------------------------------------------
  // UI Updates
  // -------------------------------------------------------------------------

  /**
   * Render prediction result in the result card.
   */
  function displayResult(data) {
    const isLeave = data.attrition_status === "Likely to Leave";
    const badge = document.getElementById("result-badge");
    const status = document.getElementById("result-status");
    const confidence = document.getElementById("result-confidence");
    const confidenceFill = document.getElementById("confidence-fill");
    const confidenceBar = document.getElementById("confidence-bar");
    const leaveProb = document.getElementById("result-leave");
    const stayProb = document.getElementById("result-stay");
    const modelName = document.getElementById("result-model");

    badge.textContent = data.attrition_status;
    badge.className = `result-card__badge ${isLeave ? "result-card__badge--leave" : "result-card__badge--stay"}`;

    status.textContent = isLeave ? "Employee Likely to Leave" : "Employee Likely to Stay";

    const confValue = data.prediction_confidence;
    confidence.textContent = `${confValue}%`;
    confidenceFill.style.width = `${confValue}%`;
    confidenceFill.className = `confidence-meter__fill ${isLeave ? "confidence-meter__fill--leave" : "confidence-meter__fill--stay"}`;
    confidenceBar.setAttribute("aria-valuenow", confValue);

    leaveProb.textContent = `${data.attrition_probability}%`;
    stayProb.textContent = `${data.stay_probability}%`;
    modelName.textContent = data.model_name;

    showResultState("result");
  }

  /**
   * Update dashboard stat cards and recent predictions table.
   */
  function updateDashboard(data) {
    statTotal.textContent = data.total_predictions.toLocaleString();
    statAvgRisk.textContent = `${data.average_attrition_risk}%`;

    if (!data.recent_predictions || data.recent_predictions.length === 0) {
      recentTableBody.innerHTML = `
        <tr class="table-empty">
          <td colspan="7">No predictions yet. Submit the form below to get started.</td>
        </tr>`;
      return;
    }

    recentTableBody.innerHTML = data.recent_predictions
      .map((row) => {
        const isLeave = row.status === "Likely to Leave";
        return `
          <tr>
            <td><code>${row.id}</code></td>
            <td>${formatTimestamp(row.timestamp)}</td>
            <td>${escapeHtml(row.department)}</td>
            <td>${escapeHtml(row.job_role)}</td>
            <td><span class="badge ${isLeave ? "badge--leave" : "badge--stay"}">${escapeHtml(row.status)}</span></td>
            <td>${row.risk}%</td>
            <td>${row.confidence}%</td>
          </tr>`;
      })
      .join("");
  }

  /**
   * Escape HTML to prevent XSS when rendering user-influenced data.
   */
  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // -------------------------------------------------------------------------
  // Event Handlers
  // -------------------------------------------------------------------------

  /**
   * Handle form submission.
   */
  async function handleSubmit(event) {
    event.preventDefault();
    clearValidationErrors();

    if (!validateForm()) {
      const firstInvalid = form.querySelector(".is-invalid");
      if (firstInvalid) firstInvalid.focus();
      return;
    }

    const payload = getFormData();

    showResultState("loading");
    setButtonLoading(true);

    try {
      const result = await submitPrediction(payload);
      displayResult(result);
      await loadDashboardStats();
    } catch (error) {
      showResultState("placeholder");
      alert(`Prediction Error: ${error.message}`);
    } finally {
      setButtonLoading(false);
    }
  }

  /**
   * Reset form to initial state.
   */
  function handleReset() {
    form.reset();
    clearValidationErrors();
    showResultState("placeholder");
  }

  /**
   * Attach real-time validation on blur.
   */
  function attachFieldValidation() {
    Object.keys(VALIDATION_RULES).forEach((field) => {
      const input = document.getElementById(fieldToId(field));
      if (input) {
        input.addEventListener("blur", () => validateField(field));
        input.addEventListener("input", () => {
          if (input.classList.contains("is-invalid")) {
            validateField(field);
          }
        });
      }
    });
  }

  // -------------------------------------------------------------------------
  // Initialization
  // -------------------------------------------------------------------------

  function init() {
    form.addEventListener("submit", handleSubmit);
    resetBtn.addEventListener("click", handleReset);
    refreshStatsBtn.addEventListener("click", loadDashboardStats);

    attachFieldValidation();
    loadFormOptions();
    loadDashboardStats();
    checkHealth();
    showResultState("placeholder");
  }

  // Start application when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
