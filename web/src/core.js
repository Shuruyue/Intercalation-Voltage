// Core of the voltage page: formula parsing, Magpie features, the boosted trees and the distance check.
// Runs in the browser (window.VoltageCore) and in Node for the tests (module.exports). Mirrors
// intercalation_voltage/features.py and web_model.py; tests/web_model.test.js checks it against Python.
(function (root) {
  "use strict";
  var IONS = ["Li", "Na", "K", "Mg", "Ca", "Zn", "Al", "Y", "Rb", "Cs"];

  function parseFormula(s) {
    s = String(s).replace(/\s+/g, "");
    if (!s) throw new Error("empty formula");
    var i = 0;
    function num() {
      var m = /^\d+(\.\d+)?/.exec(s.slice(i));
      if (!m) return 1;
      i += m[0].length;
      return parseFloat(m[0]);
    }
    function group() {
      var out = {};
      while (i < s.length) {
        var c = s[i];
        if (c === "(" || c === "[") {
          i++;
          var inner = group();
          if (s[i] !== ")" && s[i] !== "]") throw new Error("unbalanced brackets");
          i++;
          var n = num();
          for (var k in inner) out[k] = (out[k] || 0) + inner[k] * n;
        } else if (c === ")" || c === "]") {
          return out;
        } else {
          var m = /^[A-Z][a-z]?/.exec(s.slice(i));
          if (!m) throw new Error("cannot read the formula from “" + s.slice(i, i + 12) + "”");
          i += m[0].length;
          var n2 = num();
          out[m[0]] = (out[m[0]] || 0) + n2;
        }
      }
      return out;
    }
    var comp = group();
    if (i !== s.length) throw new Error("unbalanced brackets");
    for (var k in comp) if (comp[k] <= 0) delete comp[k];
    if (Object.keys(comp).length === 0) throw new Error("empty composition");
    return comp;
  }

  function isclose(a, b) { return Math.abs(a - b) <= 1e-8 + 1e-5 * Math.abs(b); }   // numpy.isclose defaults

  // matminer ElementProperty "magpie": minimum, maximum, range, mean, avg_dev, mode for each element property
  function magpie(comp, M) {
    var els = Object.keys(comp), p, j;
    for (j = 0; j < els.length; j++) if (!M.elements[els[j]]) throw new Error("no element data for " + els[j]);
    var w = els.map(function (e) { return comp[e]; });
    var W = w.reduce(function (a, b) { return a + b; }, 0), wmax = Math.max.apply(null, w), out = [];
    for (p = 0; p < M.props.length; p++) {
      var d = els.map(function (e) { return M.elements[e][p]; });
      var mn = Math.min.apply(null, d), mx = Math.max.apply(null, d), mean = 0, dev = 0, mode = Infinity;
      for (j = 0; j < d.length; j++) mean += d[j] * w[j];
      mean /= W;
      for (j = 0; j < d.length; j++) dev += Math.abs(d[j] - mean) * w[j];
      dev /= W;
      for (j = 0; j < d.length; j++) if (isclose(w[j], wmax) && d[j] < mode) mode = d[j];
      out.push(mn, mx, mx - mn, mean, dev, mode);
    }
    return out;
  }

  function detectIon(comp) {
    for (var i = 0; i < IONS.length; i++) if (comp[IONS[i]]) return IONS[i];
    return null;
  }

  function features(dischargedStr, chargedStr, ion, M) {
    var cd = parseFormula(dischargedStr);
    ion = ion || detectIon(cd);
    if (!ion) throw new Error("no working ion (Li, Na, K, Mg, Ca, Zn, Al, Y, Rb, Cs) in the discharged formula");
    var cc;
    if (chargedStr && String(chargedStr).trim()) cc = parseFormula(chargedStr);
    else { cc = Object.assign({}, cd); delete cc[ion]; }
    if (Object.keys(cc).length === 0) throw new Error("the charged composition is empty");
    function sumHost(c) { return Object.keys(c).filter(function (k) { return k !== ion; }).reduce(function (a, k) { return a + c[k]; }, 0); }
    var hd = sumHost(cd), hc = sumHost(cc);
    var nAtomsD = Object.keys(cd).reduce(function (a, k) { return a + cd[k]; }, 0);
    var nIonC = (cc[ion] || 0) * (hc > 0 ? hd / hc : 0);
    var extra = [((cd[ion] || 0) - nIonC) / nAtomsD, (cd[ion] || 0) / nAtomsD];
    if (!(extra[0] > 0)) throw new Error("the discharged formula must hold more " + ion + " per host than the charged one");
    return { x: magpie(cd, M).concat(magpie(cc, M), extra), ion: ion, charged: cc, discharged: cd };
  }

  // scikit-learn compares float32 features with float64 thresholds
  function predict(x, M) {
    var s = M.init;
    for (var i = 0; i < M.trees.length; i++) {
      var t = M.trees[i], n = 0;
      while (t.l[n] !== -1) n = Math.fround(x[t.f[n]]) <= t.t[n] ? t.l[n] : t.r[n];
      s += M.lr * t.v[n];
    }
    return s;
  }

  function decodeTrain(M) {
    if (M._train) return M._train;
    var bytes;
    if (typeof atob === "function") {
      var bin = atob(M.train_b64); bytes = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    } else { bytes = new Uint8Array(Buffer.from(M.train_b64, "base64")); }
    M._train = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    return M._train;
  }

  // mean Euclidean distance to the k nearest training reactions in standardized space, and their indices
  function neighbours(x, M) {
    var T = decodeTrain(M), K = M.n_keep, n = M.n_train, k = M.k;
    var z = M.keep.map(function (j, i) { return (x[j] - M.mean[i]) / M.scale[i]; });
    var best = [];
    for (var r = 0; r < n; r++) {
      var d2 = 0, o = r * K;
      for (var c = 0; c < K; c++) { var q = z[c] - T[o + c]; d2 += q * q; }
      if (best.length < k || d2 < best[best.length - 1][0]) {
        best.push([d2, r]);
        best.sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
        if (best.length > k) best.pop();
      }
    }
    var dist = best.reduce(function (a, b) { return a + Math.sqrt(b[0]); }, 0) / best.length;
    return { distance: dist, index: best.map(function (b) { return b[1]; }), each: best.map(function (b) { return Math.sqrt(b[0]); }) };
  }

  function run(discharged, charged, ion, M) {
    var f = features(discharged, charged, ion, M);
    var nb = neighbours(f.x, M);
    return { voltage: predict(f.x, M), distance: nb.distance, inside: nb.distance <= M.threshold, neighbours: nb,
             ion: f.ion, charged: f.charged, discharged: f.discharged };
  }

  function formatComp(c) {
    return Object.keys(c).map(function (k) { var v = c[k]; return k + (Math.abs(v - 1) < 1e-9 ? "" : String(+v.toFixed(3))); }).join("");
  }

  var api = { parseFormula: parseFormula, magpie: magpie, features: features, predict: predict, neighbours: neighbours,
              run: run, detectIon: detectIon, formatComp: formatComp, IONS: IONS };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.VoltageCore = api;
})(typeof window !== "undefined" ? window : this);
