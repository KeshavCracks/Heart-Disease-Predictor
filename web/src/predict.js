import MODEL from "./model.json";

// Client-side replica of the fitted scikit-learn Logistic Regression pipeline.
// Verified against sklearn predict_proba (0.026918 / 0.944255 for presets A/B).
export function predict(values) {
  const x = [];
  MODEL.numeric.forEach((n, i) => {
    x.push((parseFloat(values[n]) - MODEL.num_mean[i]) / MODEL.num_scale[i]);
  });
  // Missing-indicator columns (scaled) are constant for complete inputs.
  MODEL.ind_mean.forEach((m, j) => x.push((0 - m) / MODEL.ind_scale[j]));
  MODEL.categorical.forEach((c, ci) => {
    for (const cat of MODEL.categories[ci]) x.push(cat === String(values[c]) ? 1 : 0);
  });
  let z = MODEL.intercept;
  for (let i = 0; i < x.length; i++) z += MODEL.coef[i] * x[i];
  const score = 1 / (1 + Math.exp(-z));
  return { score, positive: score >= MODEL.threshold };
}

export const FIELDS = {
  numeric: [
    { key: "age", label: "Age in years", min: 28, max: 77, step: 1 },
    { key: "trestbps", label: "Resting blood pressure (mm Hg)", min: 80, max: 200, step: 1 },
    { key: "chol", label: "Cholesterol (mg/dL)", min: 85, max: 603, step: 1 },
    { key: "thalch", label: "Maximum heart rate in exercise test", min: 60, max: 202, step: 1 },
    { key: "oldpeak", label: "ST-segment change (oldpeak)", min: -2.6, max: 6.2, step: 0.1 },
  ],
  categorical: [
    { key: "sex", label: "Sex as recorded in the dataset", options: { 0: "Female", 1: "Male" } },
    { key: "cp", label: "Chest-pain category", options: { 1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "No chest pain reported" } },
    { key: "fbs", label: "Fasting blood sugar above 120?", options: { 0: "No", 1: "Yes" } },
    { key: "restecg", label: "Resting ECG category", options: { 0: "Normal", 1: "ST-T wave change", 2: "Possible LV hypertrophy" } },
    { key: "exang", label: "Chest pain during exercise?", options: { 0: "No", 1: "Yes" } },
    { key: "slope", label: "Exercise-test ST slope", options: { 1: "Rising", 2: "Flat", 3: "Falling" } },
  ],
};

export const PRESETS = {
  A: { age: 45, trestbps: 120, chol: 200, thalch: 170, oldpeak: 0.2, sex: "0", cp: "2", fbs: "0", restecg: "0", exang: "0", slope: "1" },
  B: { age: 60, trestbps: 150, chol: 280, thalch: 120, oldpeak: 2.5, sex: "1", cp: "4", fbs: "1", restecg: "1", exang: "1", slope: "3" },
};

export const GLOSSARY = {
  age: "How old the person was, in years.",
  sex: "Recorded as male or female in the old study.",
  cp: "The kind of chest discomfort that was noted, in four buckets.",
  trestbps: "Blood pressure while resting, in mm Hg.",
  chol: "The amount of cholesterol in the blood, in mg/dL.",
  fbs: "Whether blood sugar after not eating was above 120 mg/dL.",
  restecg: "A resting heart tracing (ECG), grouped into three types.",
  thalch: "The fastest the heart beat during an exercise test.",
  exang: "Whether chest pain appeared during exercise.",
  oldpeak: "How far part of the exercise ECG tracing dipped.",
  slope: "The shape of that ECG segment: rising, flat, or falling.",
};
