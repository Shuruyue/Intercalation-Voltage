// The page's JavaScript port against Python: voltage and distance of every case in results/web_check.json.
// The model is read from the data block of the published page, so this tests what the browser runs.
// Usage: node tests/web_model.test.js
"use strict";
const fs = require("fs");
const path = require("path");
const ROOT = path.join(__dirname, "..");
const core = require(path.join(ROOT, "web", "src", "core.js"));
const page = fs.readFileSync(path.join(ROOT, "web", "intercalation-voltage.html"), "utf8");
const m = /window\.VOLTAGE_DATA = (\{.*?\});<\/script>/s.exec(page);
if (!m) { console.log("no data block in the page"); process.exit(1); }
const M = JSON.parse(m[1]).model;   // the page writes "</" as the JSON escape "<\/", which JSON.parse reads back
const cases = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "web_check.json"), "utf8"));
const TOL_V = 1e-9, TOL_D = 1e-6;
let worstV = 0, worstD = 0, fail = 0;
for (const c of cases) {
  const r = core.run(c.discharged, c.charged, c.ion, M);
  const dv = Math.abs(r.voltage - c.pred), dd = Math.abs(r.distance - c.dist);
  worstV = Math.max(worstV, dv); worstD = Math.max(worstD, dd);
  if (!(dv <= TOL_V && dd <= TOL_D)) {
    fail++;
    if (fail <= 5) console.log("MISMATCH", c.discharged, c.charged, c.ion, r.voltage, c.pred, r.distance, c.dist);
  }
}
// formulas the parser must reject or read the same way
const same = [["Li3V2(PO4)3", "Li3V2P3O12"], ["Li2[CoO2]3", "Li2Co3O6"], ["LiFe0.5Mn0.5PO4", "LiMn0.5Fe0.5PO4"]];
for (const [a, b] of same) {
  const x = core.features(a, "", null, M).x, y = core.features(b, "", null, M).x;
  if (x.some((v, i) => Math.abs(v - y[i]) > 1e-12)) { fail++; console.log("PARSE", a, b); }
}
for (const bad of ["", "CoO2", "Li(CoO2", "LiXx2", "<img src=x>"]) {
  let threw = false;
  try { core.run(bad, "", null, M); } catch (e) { threw = true; }
  if (!threw) { fail++; console.log("ACCEPTED", bad); }
}
console.log(`cases=${cases.length}  max|dV|=${worstV.toExponential(2)} V  max|d dist|=${worstD.toExponential(2)}  fail=${fail}`);
process.exit(fail ? 1 : 0);
