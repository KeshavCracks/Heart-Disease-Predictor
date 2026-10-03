import { useEffect, useRef, useState } from "react";
import { motion, animate, useInView } from "framer-motion";
import { predict, FIELDS, PRESETS, GLOSSARY } from "./predict.js";

const EASE = [0.22, 1, 0.36, 1];

function Reveal({ children, delay = 0, y = 24 }) {
  return (
    <motion.div initial={{ opacity: 0, y }} whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-70px" }} transition={{ duration: 0.55, delay, ease: EASE }}>
      {children}
    </motion.div>
  );
}

function CountUp({ to, suffix = "" }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true });
  useEffect(() => {
    if (!inView) return;
    const c = animate(0, to, { duration: 1.3, ease: "easeOut", onUpdate: (v) => { if (ref.current) ref.current.textContent = Math.round(v) + suffix; } });
    return () => c.stop();
  }, [inView, to, suffix]);
  return <b ref={ref}>0{suffix}</b>;
}

function AnimatedScore({ value }) {
  const ref = useRef(null);
  const prev = useRef(0);
  useEffect(() => {
    const c = animate(prev.current, value, { duration: 0.6, ease: "easeOut", onUpdate: (v) => { if (ref.current) ref.current.textContent = v.toFixed(3); } });
    prev.current = value;
    return () => c.stop();
  }, [value]);
  return <p className="score" ref={ref}>0.000</p>;
}

function Words({ text, accent = false, base = 0 }) {
  return (
    <span className={accent ? "accent" : undefined}>
      {text.split(" ").map((w, i) => (
        <motion.span key={i} style={{ display: "inline-block", marginRight: "0.28em" }}
          initial={{ opacity: 0, y: 26, filter: "blur(6px)" }} animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          transition={{ duration: 0.6, delay: base + i * 0.06, ease: EASE }}>
          {w}
        </motion.span>
      ))}
    </span>
  );
}

const CANDIDATES = [
  ["Logistic Regression, 11 fields — selected", "0.816 ± 0.053", true],
  ["Logistic Regression, 13 fields", "0.815 ± 0.056", false],
  ["XGBoost, 11 fields", "0.812 ± 0.062", false],
  ["Random Forest, 11 fields", "0.811 ± 0.059", false],
  ["Random Forest, 13 fields", "0.809 ± 0.060", false],
  ["XGBoost, 13 fields", "0.802 ± 0.057", false],
];
const HELDOUT = [
  ["Accuracy", "0.826", "How often the guess matched the study's label, overall."],
  ["Sensitivity / recall", "0.882", "Of the positives in the study, how often the model caught them."],
  ["Specificity", "0.756", "Of the negatives, how often the model also said negative."],
  ["Precision", "0.818", "When the model said positive, how often the study agreed."],
  ["F1", "0.849", "One number balancing catching positives with being right."],
  ["ROC-AUC", "0.905", "How well the score separates positives from negatives."],
];
const COHORTS = [["Cleveland", 303], ["Hungary", 294], ["VA Long Beach", 200], ["Switzerland", 123]];

export default function App() {
  const [preset, setPreset] = useState("A");
  const [values, setValues] = useState({ ...PRESETS.A });
  const { score, positive } = predict(values);

  const set = (k, v) => { setValues((s) => ({ ...s, [k]: v })); setPreset("custom"); };
  const choosePreset = (p) => { setPreset(p); if (p === "A" || p === "B") setValues({ ...PRESETS[p] }); };
  const presetLabel = preset === "custom" ? "Custom fictional values" : `Fictional example ${preset}`;

  return (
    <>
      <div className="wrap">
        <header className="masthead">
          <span className="brand"><span className="mark"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M3 12h4l2.5-6 4 12 2.5-6H21" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" /></svg></span>Heart disease model demo</span>
          <span className="masthead-meta"><span>UCI Heart Disease dataset</span><span>Independent educational project</span></span>
        </header>

        <section className="hero">
          <motion.div className="hero-aurora" animate={{ opacity: [0.7, 1, 0.7] }} transition={{ duration: 7, repeat: Infinity, ease: "easeInOut" }} />
          <div className="hero-dots" />
          <div className="hero-inner">
            <div>
              <span className="pill"><i />Educational demo · fictional values only</span>
              <h1 className="title"><Words text="Explore a historical model," base={0.1} /><br /><Words text="not a personal risk score" accent base={0.4} /></h1>
              <p className="lede">Try fictional values against a model trained on 920 historical records from four source cohorts. Its output is a dataset label, not a health assessment.</p>
              <p className="hero-note"><strong>Use fictional values only.</strong> Everything runs in your browser and nothing is saved or sent. Selected model: Logistic Regression. The score is not a calibrated probability for a person.</p>
            </div>
            <div className="stats">
              <div className="stat"><CountUp to={920} /><span>historical records</span></div>
              <div className="stat"><CountUp to={4} /><span>source cohorts</span></div>
              <div className="stat"><CountUp to={11} /><span>input fields in the selected model</span></div>
            </div>
          </div>
        </section>

        <Reveal>
          <p className="overline">Try it</p>
          <h2 className="h2">Try a fictional example</h2>
          <p className="p">Inputs are limited to values observed in the source data — dataset limits, not medical ranges. The output updates live as you change values.</p>
        </Reveal>

        <div className="tool">
          <Reveal delay={0.05}>
            <div className="field" style={{ marginBottom: "1.2rem" }}>
              <label htmlFor="preset">Choose a starting example</label>
              <select id="preset" value={preset} onChange={(e) => choosePreset(e.target.value)}>
                <option value="A">Fictional example A</option>
                <option value="B">Fictional example B</option>
                <option value="custom">Enter custom fictional values</option>
              </select>
            </div>
            <h3 className="h3" style={{ marginTop: 0 }}>Measurements</h3>
            <div className="grid2">
              {FIELDS.numeric.map((f) => (
                <div className="field" key={f.key}>
                  <label htmlFor={f.key}>{f.label}</label>
                  <input id={f.key} type="number" min={f.min} max={f.max} step={f.step} value={values[f.key]}
                    onChange={(e) => set(f.key, e.target.value)} />
                </div>
              ))}
            </div>
            <h3 className="h3">Categories</h3>
            <div className="grid2">
              {FIELDS.categorical.map((f) => (
                <div className="field" key={f.key}>
                  <label htmlFor={f.key}>{f.label}</label>
                  <select id={f.key} value={values[f.key]} onChange={(e) => set(f.key, e.target.value)}>
                    {Object.entries(f.options).map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                  </select>
                </div>
              ))}
            </div>
          </Reveal>

          <Reveal delay={0.12}>
            <div className="output" role="status" aria-live="polite">
              <p className="lbl">Model score, from 0 to 1</p>
              <AnimatedScore value={score} />
              <div className="bar"><motion.i initial={{ width: 0 }} animate={{ width: `${Math.round(score * 100)}%` }} transition={{ duration: 0.6, ease: EASE }} /></div>
              <div><motion.span key={String(positive)} initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className={`chip ${positive ? "pos" : "neg"}`}>{positive ? "Positive source-data label" : "Negative source-data label"}</motion.span></div>
              <p className="det">Uncalibrated model output — not a percentage chance or an individual risk estimate.</p>
              <p className="det">Class at the 0.50 software cutoff. Positive means the original table recorded <code>num &gt; 0</code>; this is not a diagnosis.</p>
              <p className="meta">{presetLabel}</p>
            </div>
          </Reveal>
        </div>

        <Reveal><div className="section">
          <p className="overline">Interpretation</p>
          <h2 className="h2">How to read the output</h2>
          <p className="p">Think of the score as a similarity dial: near 1 means your row looks like the positive examples in the old research table; near 0 means it looks like the negative ones. It is not a person's chance of having a condition.</p>
          <p className="p">The model runs entirely in your browser by comparing your numbers with patterns from the historical data, then says which side of the 0.50 line the row falls on.</p>
        </div></Reveal>

        <Reveal><div className="section">
          <p className="overline">Evaluation</p>
          <h2 className="h2">How the candidates compared</h2>
          <p className="p">Six combinations were tested. Selection used a one-standard-error rule on grouped cross-validation by source cohort; the patient-level test split did not choose the model.</p>
          <div className="tbl"><table>
            <caption style={{ textAlign: "left", padding: ".8rem 1rem 0", color: "var(--ink-3)", fontSize: ".85rem" }}>Grouped cross-validation ROC-AUC across source-cohort folds (mean ± SD).</caption>
            <thead><tr><th>Candidate</th><th className="num">Grouped-CV ROC-AUC</th></tr></thead>
            <tbody>{CANDIDATES.map(([n, v, sel]) => <tr key={n} className={sel ? "sel" : undefined}><td>{n}</td><td className="num">{v}</td></tr>)}</tbody>
          </table></div>
          <h3 className="h3">Selected model on the held-out examples (184 records)</h3>
          <div className="tbl"><table>
            <thead><tr><th>Measure</th><th className="num">Value</th><th>What it means, in plain words</th></tr></thead>
            <tbody>{HELDOUT.map(([m, v, d]) => <tr key={m}><td>{m}</td><td className="num">{v}</td><td>{d}</td></tr>)}</tbody>
          </table></div>
        </div></Reveal>

        <Reveal><div className="section">
          <p className="overline">Data</p>
          <h2 className="h2">What is in the source data?</h2>
          <p className="p">The demo combines four historical UCI cohorts. The original <code>num</code> field is mapped to a binary label: zero stays negative, anything above becomes positive. Cohort names are used for grouped validation only, never as inputs.</p>
          <div className="tbl"><table>
            <thead><tr><th>Source cohort</th><th className="num">Records</th></tr></thead>
            <tbody>{COHORTS.map(([n, c]) => <tr key={n}><td>{n}</td><td className="num">{c}</td></tr>)}</tbody>
          </table></div>
        </div></Reveal>

        <Reveal><div className="section">
          <p className="overline">In everyday words</p>
          <h2 className="h2">What the words mean</h2>
          <p className="p">No medical background needed — here is what each input means in plain language.</p>
          <dl className="gloss">
            {Object.entries(GLOSSARY).map(([k, v]) => {
              const f = [...FIELDS.numeric, ...FIELDS.categorical].find((x) => x.key === k);
              return <div key={k}><dt>{f ? f.label : k}</dt><dd>{v}</dd></div>;
            })}
          </dl>
        </div></Reveal>

        <Reveal><div className="section">
          <p className="overline">Limits</p>
          <h2 className="h2">What the results do not establish</h2>
          <ul style={{ color: "var(--ink-2)", maxWidth: "74ch", lineHeight: 1.7 }}>
            <li>The dataset is historical, modest in size, and not representative of every current or local population.</li>
            <li>Grouped validation uses four source cohorts; it cannot replace independent external validation.</li>
            <li>No prospective, calibration, fairness, or clinical review has been completed.</li>
            <li>The 0.50 cutoff is a software demo setting, not a clinically chosen threshold.</li>
            <li>Do not use the output for diagnosis, screening, treatment, or reassurance.</li>
          </ul>
          <p className="p">Data source: UCI Machine Learning Repository, <a href="https://archive.ics.uci.edu/dataset/45/heart+disease" rel="noreferrer">Heart Disease dataset</a>.</p>
        </div></Reveal>

        <footer className="footer">
          <span>Independent educational project.</span><span>Not affiliated with Vercel.</span><span>Not for clinical use.</span>
        </footer>
      </div>
    </>
  );
}
