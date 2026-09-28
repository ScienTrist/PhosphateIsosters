(() => {
  const screens = {
    start: document.getElementById("screen-start"),
    adminSetup: document.getElementById("screen-admin-setup"),
    review: document.getElementById("screen-review"),
    done: document.getElementById("screen-done"),
  };
  function showScreen(name) {
    Object.values(screens).forEach((s) => s.classList.add("hidden"));
    screens[name].classList.remove("hidden");
  }

  // Same set src/constants.py's METALS uses elsewhere in the pipeline. A
  // metal ion is a lone, unbonded atom -- "stick" style draws nothing
  // sensible for those, so they always render as a small sphere instead,
  // and their PDB atom name equals their resn (e.g. atom "MG" for resn
  // "MG"), not a residue-style backbone name like "CA" -- matters for label
  // anchoring below.
  const METALS = new Set(["MG", "ZN", "MN", "CA", "FE", "CU", "CO", "NI"]);

  // 3Dmol.js's own PDB bond-perception (vendor/3Dmol-min.js) splits every
  // atom into "standard polymer" (matched against a hardcoded amino/nucleic
  // acid residue-name list) vs "hetero" (hetflag, e.g. the ligand) BEFORE
  // computing any bonds, then assigns bonds separately within each group --
  // polymer atoms via a sequential-residue-neighbor check, hetero atoms via
  // its own distance-based assignBonds(). Neither pass ever compares an atom
  // from one group against the other, so a REAL covalent bond between a
  // ligand atom and a protein residue (e.g. a phospho-Ser/Thr adduct) never
  // gets an edge in either atom's .bonds array -- confirmed directly on
  // ref_5GZW_AMP_401: AMP's P sits 1.59 A from SER64's OG (a real covalent
  // distance), but nothing connects them because P is hetero and OG isn't.
  // Patched here by re-running the SAME covalent-radius-sum+0.25A distance
  // test 3Dmol uses internally (bondLength table below matches its vendored
  // values), but specifically across that hetero/polymer boundary. Metals
  // are excluded on purpose -- this is the mirror-image of the metal/CONECT
  // false-covalent-bond bug already fixed elsewhere in this pipeline (see
  // [[feedback-metal-conect-records]]); re-triggering it here for Zn/Mg/etc.
  // coordination contacts would trade one display bug for another.
  const BOND_RADII = { H: .37, C: .77, N: .75, O: .73, F: .71, P: 1.06, S: 1.02, CL: .99, BR: 1.14, I: 1.33, B: .82 };
  function patchCovalentLigandBonds(v, modelIndex, resn, resi, chain) {
    const model = v.getModel(modelIndex);
    const ligAtoms = model.selectedAtoms({ resn, resi, chain });
    if (!ligAtoms.length) return;
    const protAtoms = model.selectedAtoms({
      hetflag: false, byres: true, within: { distance: 3, sel: { resn, resi, chain } },
    });
    for (const a of ligAtoms) {
      const ra = BOND_RADII[(a.elem || "").toUpperCase()];
      if (ra == null) continue;
      for (const b of protAtoms) {
        if (METALS.has(b.resn)) continue;
        const rb = BOND_RADII[(b.elem || "").toUpperCase()];
        if (rb == null) continue;
        const cutoff = ra + rb + 0.25;
        const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < 0.5 || d2 > cutoff * cutoff) continue;
        if (a.bonds.indexOf(b.index) === -1) { a.bonds.push(b.index); a.bondOrder.push(1); }
        if (b.bonds.indexOf(a.index) === -1) { b.bonds.push(a.index); b.bondOrder.push(1); }
      }
    }
  }

  let config = { count_options: [100, 150, 200], seconds_per_structure: 60, pool_size: 0 };
  let state = {
    reviewerName: null,
    pairIds: [],
    ratedSet: new Set(),
    currentPairId: null,
    selectedScore: null,
    wrongPhosphate: false,
    viewer: null,
    measureAtoms: [],
    measureShape: null,
    measureLabel: null,
    adminMode: false,
    adminToken: null,
    adminList: [],
    adminIndex: -1,
    adminReviewerName: null,
  };

  // ---------- Start screen ----------

  async function loadConfig() {
    const res = await fetch("/api/config");
    config = await res.json();
    // Every reviewer is assigned exclusively from the shared core (see
    // app.py's /api/start), so shared_core_size -- not pool_size -- is the
    // real ceiling on how many structures anyone can pick.
    const cap = config.shared_core_size;
    const options = config.count_options.filter((n) => n <= cap);
    if (options.length === 0) options.push(cap);
    const container = document.getElementById("count-buttons");
    container.innerHTML = "";
    options.forEach((n) => {
      const btn = document.createElement("button");
      btn.textContent = n;
      btn.type = "button";
      btn.addEventListener("click", () => selectCount(n));
      container.appendChild(btn);
    });
    document.getElementById("count-custom").max = String(cap);
    selectCount(options[0]);
  }

  let selectedCount = null;
  function selectCount(n) {
    selectedCount = n;
    document.getElementById("count-custom").value = "";
    [...document.getElementById("count-buttons").children].forEach((b) => {
      b.classList.toggle("selected", Number(b.textContent) === n);
    });
    updateTimeEstimate();
  }

  document.getElementById("count-custom").addEventListener("input", (e) => {
    const v = parseInt(e.target.value, 10);
    if (v > 0) {
      selectedCount = Math.min(v, config.shared_core_size);
      [...document.getElementById("count-buttons").children].forEach((b) =>
        b.classList.remove("selected")
      );
      updateTimeEstimate();
    }
  });

  function updateTimeEstimate() {
    if (!selectedCount) return;
    const totalSeconds = selectedCount * config.seconds_per_structure;
    const minutes = Math.round(totalSeconds / 60);
    document.getElementById("time-estimate").textContent =
      `About ${minutes} min total for ${selectedCount} structures -- the same ${config.shared_core_size} every reviewer sees.`;
  }

  document.getElementById("start-btn").addEventListener("click", async () => {
    const reviewer_name = document.getElementById("reviewer-name").value.trim();
    const access_code = document.getElementById("access-code").value;
    const errEl = document.getElementById("start-error");
    errEl.textContent = "";

    if (!reviewer_name) { errEl.textContent = "Enter your name."; return; }
    if (!selectedCount || selectedCount < 10) { errEl.textContent = "Pick how many structures to review."; return; }

    try {
      const res = await fetch("/api/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reviewer_name, access_code, requested_count: selectedCount }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        errEl.textContent = body.detail || "Could not start session.";
        return;
      }
      const data = await res.json();
      state.reviewerName = data.reviewer_name;
      state.pairIds = data.pair_ids;
      state.ratedSet = new Set(data.rated_pair_ids);
      state.adminMode = false;
      togglePanelsForMode();
      showScreen("review");
      initViewer();
      updateProgress(data.done, data.total);
      await loadNextPair();
    } catch (e) {
      errEl.textContent = "Network error contacting the server.";
    }
  });

  // ---------- Admin: browse by score ----------

  document.getElementById("admin-link").addEventListener("click", (e) => {
    e.preventDefault();
    showScreen("adminSetup");
  });

  document.getElementById("admin-back-link").addEventListener("click", (e) => {
    e.preventDefault();
    state.adminMode = false;
    showScreen("start");
  });

  function renderAdminResults() {
    const container = document.getElementById("admin-results");
    container.innerHTML = "";
    if (!state.adminList.length) {
      container.textContent = "No pairs found in that range.";
      return;
    }
    state.adminList.forEach((p, i) => {
      const row = document.createElement("div");
      row.className = "admin-result-row";
      const score = p.score_no_vdw ?? p.score_with_vdw;
      row.innerHTML =
        `<span>${p.ref_pdb}/${p.ref_resname} vs ${p.hit_pdb}/${p.hit_resname}</span>` +
        `<span class="admin-result-score">${Number(score).toFixed(3)}</span>`;
      row.addEventListener("click", () => {
        state.adminReviewerName = null;
        enterAdminPair(i);
      });
      container.appendChild(row);
    });
  }

  document.getElementById("admin-load-btn").addEventListener("click", async () => {
    const errEl = document.getElementById("admin-error");
    errEl.textContent = "";
    const token = document.getElementById("admin-token").value;
    if (!token) { errEl.textContent = "Enter the admin token."; return; }
    const params = new URLSearchParams({
      token,
      metric: document.getElementById("admin-metric").value,
      min_score: document.getElementById("admin-min-score").value,
      max_score: document.getElementById("admin-max-score").value,
      min_ref_bits: document.getElementById("admin-min-ref-bits").value,
      sort: document.getElementById("admin-sort").value,
      limit: document.getElementById("admin-limit").value,
    });
    const maxRefBits = document.getElementById("admin-max-ref-bits").value;
    if (maxRefBits !== "") params.set("max_ref_bits", maxRefBits);
    try {
      const res = await fetch(`/api/admin/pairs?${params.toString()}`);
      const body = await res.json();
      if (!res.ok) { errEl.textContent = body.detail || `Error: HTTP ${res.status}`; return; }
      state.adminToken = token;
      state.adminList = body;
      renderAdminResults();
    } catch (e) {
      errEl.textContent = "Network error contacting the server.";
    }
  });

  // ---------- Admin: browse as a reviewer ----------

  document.getElementById("admin-load-reviewers-btn").addEventListener("click", async () => {
    const errEl = document.getElementById("admin-reviewer-error");
    errEl.textContent = "";
    const token = document.getElementById("admin-token").value;
    if (!token) { errEl.textContent = "Enter the admin token above."; return; }
    try {
      const res = await fetch(`/api/admin/reviewers?token=${encodeURIComponent(token)}`);
      const body = await res.json();
      if (!res.ok) { errEl.textContent = body.detail || `Error: HTTP ${res.status}`; return; }
      state.adminToken = token;
      const sel = document.getElementById("admin-reviewer-select");
      sel.innerHTML = "";
      body.forEach((r) => {
        const opt = document.createElement("option");
        opt.value = r.reviewer_name;
        opt.textContent = `${r.reviewer_name} (${r.rated_count}/${r.total_assigned} rated)`;
        sel.appendChild(opt);
      });
      document.getElementById("admin-browse-reviewer-btn").disabled = body.length === 0;
      if (body.length === 0) errEl.textContent = "No reviewers have started a session yet.";
    } catch (e) {
      errEl.textContent = "Network error contacting the server.";
    }
  });

  document.getElementById("admin-browse-reviewer-btn").addEventListener("click", async () => {
    const errEl = document.getElementById("admin-reviewer-error");
    errEl.textContent = "";
    const reviewerName = document.getElementById("admin-reviewer-select").value;
    if (!reviewerName) return;
    try {
      const res = await fetch(
        `/api/admin/reviewer/${encodeURIComponent(reviewerName)}/pairs?token=${encodeURIComponent(state.adminToken)}`
      );
      const body = await res.json();
      if (!res.ok) { errEl.textContent = body.detail || `Error: HTTP ${res.status}`; return; }
      if (!body.length) { errEl.textContent = `${reviewerName} has no (still-valid) assigned structures.`; return; }
      state.adminList = body;
      state.adminReviewerName = reviewerName;
      await enterAdminPair(0);
    } catch (e) {
      errEl.textContent = "Network error contacting the server.";
    }
  });

  function togglePanelsForMode() {
    document.getElementById("rating-panel").classList.toggle("hidden", state.adminMode);
    document.getElementById("comment-label").classList.toggle("hidden", state.adminMode);
    document.getElementById("submit-btn").classList.toggle("hidden", state.adminMode);
    document.getElementById("rating-error").classList.toggle("hidden", state.adminMode);
    document.getElementById("admin-panel").classList.toggle("hidden", !state.adminMode);
  }

  function updateAdminPanel() {
    const p = state.adminList[state.adminIndex];
    let text =
      `pair_id: ${p.pair_id}\n` +
      `prolif_plif_score_no_vdw:   ${p.score_no_vdw}\n` +
      `prolif_plif_score_with_vdw: ${p.score_with_vdw}\n` +
      `ref_n_bits_novdw (min_ref_int): ${p.ref_n_bits_novdw}\n` +
      `pocket_rmsd: ${p.pocket_rmsd == null ? "n/a" : p.pocket_rmsd.toFixed(3)}`;
    if (state.adminReviewerName) {
      const given = p.reviewer_score != null
        ? `score ${p.reviewer_score}/5`
        : p.reviewer_wrong_reference_phosphate
          ? "flagged: wrong reference phosphate"
          : "not yet rated";
      text += `\n\n--- ${state.adminReviewerName}'s rating ---\n${given}`;
      if (p.reviewer_comment) text += `\ncomment: ${p.reviewer_comment}`;
    }
    document.getElementById("admin-panel-score").textContent = text;
    document.getElementById("admin-panel-position").textContent =
      `${state.adminIndex + 1} / ${state.adminList.length}`;
    document.getElementById("admin-prev-btn").disabled = state.adminIndex <= 0;
    document.getElementById("admin-next-btn").disabled = state.adminIndex >= state.adminList.length - 1;
  }

  async function enterAdminPair(index) {
    state.adminMode = true;
    state.adminIndex = index;
    showScreen("review");
    initViewer();
    togglePanelsForMode();
    updateAdminPanel();
    await loadPairIntoViewer(state.adminList[index].pair_id);
  }

  document.getElementById("admin-prev-btn").addEventListener("click", async () => {
    if (state.adminIndex <= 0) return;
    state.adminIndex -= 1;
    updateAdminPanel();
    await loadPairIntoViewer(state.adminList[state.adminIndex].pair_id);
  });

  document.getElementById("admin-next-btn").addEventListener("click", async () => {
    if (state.adminIndex >= state.adminList.length - 1) return;
    state.adminIndex += 1;
    updateAdminPanel();
    await loadPairIntoViewer(state.adminList[state.adminIndex].pair_id);
  });

  document.getElementById("admin-back-to-list-link").addEventListener("click", (e) => {
    e.preventDefault();
    showScreen("adminSetup");
  });

  // ---------- Review screen ----------

  function showViewerError(message) {
    const el = document.getElementById("viewer-error");
    el.textContent = message;
    el.classList.remove("hidden");
  }

  function hideViewerError() {
    document.getElementById("viewer-error").classList.add("hidden");
  }

  function hasWebGL() {
    try {
      const canvas = document.createElement("canvas");
      return !!(canvas.getContext("webgl") || canvas.getContext("experimental-webgl"));
    } catch (e) {
      return false;
    }
  }

  // Some environments (seen: Firefox under RDP with an "AllowWebgl2:false"
  // policy) make getContext("webgl2", ...) *throw* instead of returning null
  // per spec -- that breaks 3Dmol.js's own
  // getContext("webgl2")||getContext("experimental-webgl")||getContext("webgl")
  // fallback chain, since a thrown exception aborts the whole `||` expression
  // instead of falling through to the WebGL1 attempts that actually work
  // here. Intercept webgl2 requests ourselves and return null immediately
  // (the spec-compliant "not supported" response) so 3Dmol's own fallback
  // reaches WebGL1.
  function patchOutThrowingWebGL2() {
    [window.HTMLCanvasElement, window.OffscreenCanvas].forEach((proto) => {
      if (!proto || proto.prototype.__phipPatched) return;
      const orig = proto.prototype.getContext;
      proto.prototype.getContext = function (type, ...args) {
        if (type === "webgl2") return null;
        try {
          return orig.call(this, type, ...args);
        } catch (e) {
          return null;
        }
      };
      proto.prototype.__phipPatched = true;
    });

    // 3Dmol.js's initGL() only takes its safe (webgl2 || experimental-webgl
    // || webgl) fallback path when `!OffscreenCanvas` is true, or when a
    // grid layout (rows/cols/row/col) is fully configured. We're not using a
    // grid, so without this, it takes the *other* branch instead -- pure
    // OffscreenCanvas + webgl2, no fallback to webgl1 at all -- which is
    // exactly what leaves this._gl unset here even after the getContext
    // patch above.
    //
    // IMPORTANT: assign undefined, don't `delete` -- deleting the global
    // removes the *identifier* entirely, and 3Dmol's minified code
    // references the bare (unqualified) `OffscreenCanvas` name, so
    // evaluating `!OffscreenCanvas` against a deleted identifier throws a
    // ReferenceError (caught by 3Dmol's own try/catch, silently aborting
    // initGL before this._gl is ever assigned) rather than safely
    // evaluating to true the way `!undefined` does.
    try {
      window.OffscreenCanvas = undefined;
    } catch (e) {
      try {
        Object.defineProperty(window, "OffscreenCanvas", { value: undefined, configurable: true });
      } catch (e2) { /* nothing more we can do */ }
    }
  }

  function initViewer() {
    if (state.viewer) return;
    if (!hasWebGL()) {
      showViewerError(
        "This browser/session can't get a WebGL context, so the 3D viewer can't render " +
        "(common over Remote Desktop without GPU passthrough). Try: a different browser, " +
        "opening this on the machine's physical display instead of over RDP, or checking " +
        "chrome://gpu (or about:support on Firefox) for 'WebGL: unavailable'."
      );
      return;
    }
    patchOutThrowingWebGL2();
    try {
      state.viewer = $3Dmol.createViewer(document.getElementById("viewer"), {
        backgroundColor: "0x0f1115",
      });
      if (!state.viewer || !state.viewer.render) {
        throw new Error("createViewer returned no usable viewer");
      }
    } catch (e) {
      showViewerError(
        `3Dmol.js failed to create a viewer even after forcing WebGL1: ${e.message || e}`
      );
    }
  }

  const ELEMENT_NAMES = {
    C: "Carbon", N: "Nitrogen", O: "Oxygen", S: "Sulfur", P: "Phosphorus",
    H: "Hydrogen", MG: "Magnesium", ZN: "Zinc", MN: "Manganese", CA: "Calcium",
    FE: "Iron", CU: "Copper", CO: "Cobalt", NI: "Nickel", CL: "Chlorine", NA: "Sodium",
    K: "Potassium", F: "Fluorine", BR: "Bromine", I: "Iodine", B: "Boron", SE: "Selenium",
  };

  const MEASURE_COLOR = "0xffcc00";

  function clearMeasurement() {
    const v = state.viewer;
    if (v) {
      if (state.measureShape) v.removeShape(state.measureShape);
      if (state.measureLabel) v.removeLabel(state.measureLabel);
      v.render();
    }
    state.measureAtoms = [];
    state.measureShape = null;
    state.measureLabel = null;
    document.getElementById("measure-body").textContent = "Click two atoms to measure the distance between them.";
  }

  function describeAtom(atom) {
    const side = atom.model === 0 ? "ref" : "hit";
    return `${atom.atom} (${atom.resn}${atom.resi}.${atom.chain}, ${side})`;
  }

  // Draws the line+label for the current 2-atom measurement. Callable both
  // right after a click AND from applyStyles() -- that function's blanket
  // removeAllShapes()/removeAllLabels() has no way to spare just this one
  // shape, so an in-progress measurement needs to be redrawn from
  // state.measureAtoms any time applyStyles() runs (see the call near its
  // end), or it'd silently vanish on the next checkbox click while the text
  // readout still claimed it was there.
  function drawMeasurement() {
    const v = state.viewer;
    const a = state.measureAtoms[0];
    const b = state.measureAtoms[1];
    const dist = Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
    const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 };
    state.measureShape = v.addLine({
      start: { x: a.x, y: a.y, z: a.z },
      end: { x: b.x, y: b.y, z: b.z },
      color: MEASURE_COLOR, dashed: false, linewidth: 3,
    });
    state.measureLabel = v.addLabel(dist.toFixed(2) + " Angstrom", {
      position: mid, fontSize: 13, fontColor: "black",
      backgroundColor: MEASURE_COLOR, backgroundOpacity: 0.9,
      borderThickness: 0, inFront: true,
    });
    document.getElementById("measure-body").textContent =
      describeAtom(a) + "\n  distance: " + dist.toFixed(2) + " Angstrom\n" + describeAtom(b);
  }

  // Rolling 2-atom window: click a third atom and it replaces the older of
  // the previous two, rather than requiring an explicit reset each time --
  // matches how people actually want to probe several distances in a row.
  function handleMeasureClick(atom) {
    const v = state.viewer;
    state.measureAtoms.push(atom);
    if (state.measureAtoms.length > 2) state.measureAtoms.shift();

    if (state.measureShape) { v.removeShape(state.measureShape); state.measureShape = null; }
    if (state.measureLabel) { v.removeLabel(state.measureLabel); state.measureLabel = null; }

    if (state.measureAtoms.length < 2) {
      document.getElementById("measure-body").textContent =
        "First atom picked: " + describeAtom(atom) + "\nClick a second atom to measure.";
      v.render();
      return;
    }

    drawMeasurement();
    v.render();
  }

  function setupAtomClickInspector(v) {
    v.setClickable({}, true, (atom) => {
      const out = document.getElementById("atom-info-body");
      if (!atom) { out.textContent = "Click any atom in the viewer to inspect it."; return; }
      const side = atom.model === 0 ? "reference" : "hit";
      const elemName = ELEMENT_NAMES[(atom.elem || "").toUpperCase()] || atom.elem || "unknown";
      out.textContent =
        `atom name: ${atom.atom}\n` +
        `element:   ${atom.elem} (${elemName})\n` +
        `residue:   ${atom.resn} ${atom.resi} chain ${atom.chain}\n` +
        `side:      ${side} structure`;
      handleMeasureClick(atom);
    });
  }

  document.getElementById("measure-clear-btn").addEventListener("click", clearMeasurement);

  function updateProgress(done, total) {
    document.getElementById("progress-text").textContent = `${done} / ${total} rated`;
    const pct = total > 0 ? Math.min(100, (done / total) * 100) : 0;
    document.getElementById("progress-fill").style.width = `${pct}%`;
  }

  const scoreContainer = document.getElementById("score-buttons");
  const wrongPhosphateBtn = document.getElementById("wrong-phosphate-btn");
  for (let i = 1; i <= 5; i++) {
    const btn = document.createElement("button");
    btn.textContent = i;
    btn.type = "button";
    btn.addEventListener("click", () => {
      state.selectedScore = i;
      state.wrongPhosphate = false;
      [...scoreContainer.children].forEach((b) => b.classList.toggle("selected", Number(b.textContent) === i));
      wrongPhosphateBtn.classList.remove("selected");
      document.getElementById("submit-btn").disabled = false;
    });
    scoreContainer.appendChild(btn);
  }

  // Its own vote, mutually exclusive with the 1-5 score buttons above --
  // not a 6th score value. Flags that the pipeline picked the wrong
  // phosphate group on the reference ligand to compare against, which is a
  // data problem, not a judgment about mimic quality (see the user guide's
  // "Wrong reference phosphate" examples).
  wrongPhosphateBtn.addEventListener("click", () => {
    state.selectedScore = null;
    state.wrongPhosphate = true;
    [...scoreContainer.children].forEach((b) => b.classList.remove("selected"));
    wrongPhosphateBtn.classList.add("selected");
    document.getElementById("submit-btn").disabled = false;
  });

  ["tog-ref-ligand", "tog-hit-ligand", "tog-ref-cartoon", "tog-hit-cartoon", "tog-waters",
   "tog-metals", "tog-ref-residues", "tog-hit-residues",
   "tog-ref-interactions", "tog-hit-interactions", "tog-hydrogens",
   "tog-ref-nearby", "tog-hit-nearby"].forEach((id) => {
    document.getElementById(id).addEventListener("change", applyStyles);
  });

  let currentMeta = null;

  async function fetchOk(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url} -> HTTP ${r.status}`);
    return r;
  }

  // Shared by the reviewer flow (loadNextPair) and admin browsing
  // (enterAdminPair/prev/next) -- fetches a pair's metadata + both
  // structures and renders them into the (single, shared) viewer. Returns
  // false on fetch failure so each caller can decide how to react (skip to
  // the next assignment vs. surface an error in the admin panel).
  async function loadPairIntoViewer(pairId) {
    state.currentPairId = pairId;

    let metaRes, refPdb, hitPdb;
    try {
      [metaRes, refPdb, hitPdb] = await Promise.all([
        fetchOk(`/api/pair/${encodeURIComponent(pairId)}`).then((r) => r.json()),
        fetchOk(`/api/structure/ref/${encodeURIComponent(pairId)}`).then((r) => r.text()),
        fetchOk(`/api/structure/hit/${encodeURIComponent(pairId)}`).then((r) => r.text()),
      ]);
    } catch (e) {
      console.warn(`Failed to load ${pairId}: ${e.message || e}`);
      return false;
    }

    currentMeta = metaRes;
    // pocket_rmsd is a structural-alignment QC signal (how well these two
    // pockets actually superimpose), not a mimic-quality judgment -- unlike
    // prolif_plif_score/ref_n_bits_novdw, showing it doesn't anchor a
    // reviewer's 1-5 rating, so it's shown here rather than admin-only.
    const rmsdText = metaRes.pocket_rmsd == null
      ? ""
      : ` · pocket RMSD: ${metaRes.pocket_rmsd.toFixed(2)} Å`;
    document.getElementById("pair-label").textContent =
      `${metaRes.ref_pdb}/${metaRes.ref_resname} vs ${metaRes.hit_pdb}/${metaRes.hit_resname}${rmsdText}`;

    if (!state.viewer) return true; // WebGL unavailable -- rating/browsing still works, just no 3D view

    try {
      const v = state.viewer;
      v.clear();
      // v.clear() already wiped any measurement line/label from the previous
      // structure -- reset the refs directly rather than calling
      // removeShape/removeLabel on now-stale handles.
      state.measureAtoms = [];
      state.measureShape = null;
      state.measureLabel = null;
      document.getElementById("measure-body").textContent = "Click two atoms to measure the distance between them.";
      // keepH: true -- 3Dmol.js drops hydrogen atoms during PDB parsing by
      // default. Several real interaction endpoints (an Arg guanidinium's
      // HH12/HH22, a Lys NZ's HZ2/HZ3, ...) are hydrogens; without this,
      // selecting them client-side silently finds nothing and the
      // interaction-line fallback anchors at CA instead, mixing correct
      // atom-level lines with "root of the residue" ones on the same
      // structure.
      v.addModel(refPdb, "pdb", { keepH: true });
      v.addModel(hitPdb, "pdb", { keepH: true });
      patchCovalentLigandBonds(v, 0, metaRes.ref_resname, metaRes.ref_resnum, metaRes.ref_chain);
      patchCovalentLigandBonds(v, 1, metaRes.hit_resname, metaRes.hit_resnum, metaRes.hit_chain);
      // setClickable only flags atoms present in the model at the moment
      // it's called -- has to be redone after every addModel, not once at
      // viewer-creation time (there'd be nothing to flag yet then).
      setupAtomClickInspector(v);
      document.getElementById("atom-info-body").textContent = "Click any atom in the viewer to inspect it.";
      applyStyles();

      const refSel = {
        model: 0, resn: metaRes.ref_resname, resi: metaRes.ref_resnum, chain: metaRes.ref_chain,
      };
      v.zoomTo(refSel);
      v.zoom(0.55);
      v.render();
      hideViewerError();
    } catch (e) {
      showViewerError(`3Dmol.js failed to render this structure: ${e.message || e}`);
    }
    return true;
  }

  async function loadNextPair() {
    const nextId = state.pairIds.find((id) => !state.ratedSet.has(id));
    if (!nextId) {
      showScreen("done");
      return;
    }
    state.selectedScore = null;
    state.wrongPhosphate = false;
    document.getElementById("submit-btn").disabled = true;
    document.getElementById("comment").value = "";
    [...scoreContainer.children].forEach((b) => b.classList.remove("selected"));
    wrongPhosphateBtn.classList.remove("selected");
    document.getElementById("rating-error").textContent = "";

    const ok = await loadPairIntoViewer(nextId);
    if (!ok) {
      // Server-side filtering should prevent this (a dead assignment
      // pointing at a pair no longer in the pool), but if it ever happens
      // anyway, skip it locally rather than stranding the reviewer on a
      // permanent error screen.
      state.ratedSet.add(nextId);
      return loadNextPair();
    }
  }

  function applyStyles() {
    const v = state.viewer;
    if (!v || !currentMeta) return;

    v.removeAllLabels();
    v.removeAllShapes();

    const showRefLigand = document.getElementById("tog-ref-ligand").checked;
    const showHitLigand = document.getElementById("tog-hit-ligand").checked;
    const showRefCartoon = document.getElementById("tog-ref-cartoon").checked;
    const showHitCartoon = document.getElementById("tog-hit-cartoon").checked;
    const showWaters = document.getElementById("tog-waters").checked;

    v.setStyle({}, {});

    // Palette: two families, one per "side" -- everything reference-related
    // (cartoon, ligand, phosphate emphasis, binding residues) is GREEN;
    // everything hit-related (cartoon, ligand, binding residues) is CYAN.
    // Sub-elements within a family stay the *same hue*, varied only by
    // shade/context (cartoon = dark and translucent, ligand = full
    // "<color>Carbon" scheme at normal thickness, binding residues = same
    // scheme but thinner + labeled) rather than switching to an unrelated
    // color -- so at a glance "green stuff = reference side, cyan stuff =
    // hit side" holds regardless of which specific element you're looking
    // at. REF_GREEN/HIT_CYAN below are the exact hex 3Dmol's built-in
    // greenCarbon/cyanCarbon schemes render carbon as, so the cartoon and
    // labels are literally the same color the ligand/residue sticks use,
    // not just a similar-looking approximation.
    //
    // The only *flat*, family-independent colors left are the semantic tags:
    // matched / same-group-unmatched on the hit ligand, and phosphorus-of-
    // interest on the reference ligand -- deliberately picked from OUTSIDE
    // both families AND outside the standard CPK palette (O=red, N=blue,
    // S=yellow, P=orange) so a tag is never confusable with an atom just
    // naturally being that color, or with which side it's on: magenta for
    // "matched", white for "unmatched" or "this is the phosphorus this pair
    // was scored against" -- clearly distinct from each other, from
    // red/blue/yellow/orange, and from green/cyan. White doubles up (hit
    // "unmatched" vs ref "phosphorus of interest") without ambiguity since
    // the two never appear on the same ligand.
    const REF_GREEN = "0x00ff00";
    const HIT_CYAN = "0x00ffff";
    const REF_DARK = "0x1e6b1e";
    const HIT_DARK = "0x117864";
    const TAG_MATCHED = "0xff00ff";
    const TAG_UNMATCHED_GROUP = "0xffffff";
    const TAG_REF_PHOSPHORUS = "0xffffff";

    if (showRefCartoon) {
      v.setStyle({ model: 0, hetflag: false }, { cartoon: { color: REF_DARK, opacity: 0.55 } });
    }
    if (showHitCartoon) {
      v.setStyle({ model: 1, hetflag: false }, { cartoon: { color: HIT_DARK, opacity: 0.55 } });
    }
    if (showWaters) {
      v.setStyle({ model: 0, resn: "HOH" }, { sphere: { scale: 0.18, color: REF_DARK } });
      v.setStyle({ model: 1, resn: "HOH" }, { sphere: { scale: 0.18, color: HIT_DARK } });
    }

    // "All residues within 6A" -- unlike ref/hit binding residues below (only
    // the specific residues ProLIF flagged as making a real non-VdW
    // interaction), this is pure geometry: every residue with at least one
    // atom within 6A of the ligand, sidechains included, regardless of
    // whether ProLIF detected anything there. Drawn BEFORE the ligand/
    // binding-residue/metal blocks below on purpose -- thin and unlabeled so
    // the more specific, labeled layers painting the same atoms afterward
    // visually win, rather than this generic layer covering them up.
    const showRefNearby = document.getElementById("tog-ref-nearby").checked;
    const showHitNearby = document.getElementById("tog-hit-nearby").checked;
    if (showRefNearby) {
      const refLigSel = { model: 0, resn: currentMeta.ref_resname, resi: currentMeta.ref_resnum, chain: currentMeta.ref_chain };
      v.setStyle(
        { model: 0, hetflag: false, byres: true, within: { distance: 6, sel: refLigSel } },
        { stick: { colorscheme: "greenCarbon", radius: 0.1 } },
      );
    }
    if (showHitNearby) {
      const hitLigSel = { model: 1, resn: currentMeta.hit_resname, resi: currentMeta.hit_resnum, chain: currentMeta.hit_chain };
      v.setStyle(
        { model: 1, hetflag: false, byres: true, within: { distance: 6, sel: hitLigSel } },
        { stick: { colorscheme: "cyanCarbon", radius: 0.1 } },
      );
    }

    if (showRefLigand) {
      const refSel = { model: 0, resn: currentMeta.ref_resname, resi: currentMeta.ref_resnum, chain: currentMeta.ref_chain };
      v.setStyle(refSel, { stick: { colorscheme: "greenCarbon", radius: 0.18 } });
      if (currentMeta.ref_phosphate_atom_names.length) {
        // No separate color needed -- true P (orange) / O (red) already pop
        // against the green-carbon rest of the ligand; just thicker sticks.
        v.setStyle({ ...refSel, atom: currentMeta.ref_phosphate_atom_names },
          { stick: { colorscheme: "greenCarbon", radius: 0.26 } });

        // Multi-phosphate ligands (ATP, ADP, NAD, ...) have more than one
        // phosphorus -- flag the SPECIFIC one this pair was scored against
        // the same way the hit side flags its matched/unmatched atoms: a
        // translucent halo layered on top, true element color still visible
        // underneath. Found by element rather than atom name since naming
        // conventions vary (P / PA / PB / PG / P1 / ...).
        const phosAtoms = v.selectedAtoms({ ...refSel, atom: currentMeta.ref_phosphate_atom_names, elem: "P" });
        if (phosAtoms.length) {
          const phosNames = [...new Set(phosAtoms.map((a) => a.atom))];
          v.setStyle({ ...refSel, atom: phosNames }, {
            stick: { colorscheme: "greenCarbon", radius: 0.26 },
            sphere: { scale: 0.45, color: TAG_REF_PHOSPHORUS, opacity: 0.4 },
          });
        }
      }
    }

    if (showHitLigand) {
      const hitSel = { model: 1, resn: currentMeta.hit_resname, resi: currentMeta.hit_resnum, chain: currentMeta.hit_chain };
      v.setStyle(hitSel, { stick: { colorscheme: "cyanCarbon", radius: 0.18 } });
      // Tag atoms keep their real element color (via the same colorscheme
      // as the rest of the ligand) -- the "matched"/"unmatched" flag is a
      // translucent halo sphere layered around each tagged atom instead of
      // replacing its color outright, so you can see both what an atom
      // chemically is AND that it's flagged, at the same time.
      if (currentMeta.purple_atom_names.length) {
        v.setStyle({ ...hitSel, atom: currentMeta.purple_atom_names }, {
          stick: { colorscheme: "cyanCarbon", radius: 0.2 },
          sphere: { scale: 0.4, color: TAG_UNMATCHED_GROUP, opacity: 0.35 },
        });
      }
      if (currentMeta.red_atom_names.length) {
        v.setStyle({ ...hitSel, atom: currentMeta.red_atom_names }, {
          stick: { colorscheme: "cyanCarbon", radius: 0.22 },
          sphere: { scale: 0.45, color: TAG_MATCHED, opacity: 0.4 },
        });
      }
    }

    const showRefResidues = document.getElementById("tog-ref-residues").checked;
    const showHitResidues = document.getElementById("tog-hit-residues").checked;

    function residueAnchorSel(model, resn, resi, chain) {
      // Metals have no "CA" atom to anchor on (their one atom's name IS
      // the resn) -- anchor on the whole (single-atom) selection instead.
      return METALS.has(resn)
        ? { model, resn, resi, chain }
        : { model, resn, resi, chain, atom: ["CA"] };
    }

    function labelResidue(resn, resi, chain, model, fontColor) {
      v.addLabel(`${resn}${resi}`, {
        fontSize: 11, fontColor, backgroundColor: "black", backgroundOpacity: 0.6,
        borderThickness: 0, inFront: true,
      }, residueAnchorSel(model, resn, resi, chain));
    }

    function styleResidue(resn, resi, chain, model, colorscheme, flatColor) {
      const sel = { model, resn, resi, chain };
      if (METALS.has(resn)) {
        v.setStyle(sel, { sphere: { scale: 0.35, color: flatColor } });
      } else {
        v.setStyle(sel, { stick: { colorscheme, radius: 0.13 } });
      }
    }

    // Metals are handled exclusively by the "Metal ions" toggle below, even
    // when they're also in a binding-residues list -- addLabel() always adds
    // a NEW label rather than replacing one at the same spot, so styling a
    // metal here too (as this used to) drew a duplicate overlapping label
    // for it, and unchecking "Metal ions" alone couldn't remove that copy.
    // Confirmed on ref_4ZG7_NKN_909/hit_7G2Q_XF3_902: ZN903/ZN904 (ref) and
    // ZN906 (hit) are all real binding residues AND metals, so they were
    // each drawn/labeled twice.
    if (showRefResidues) {
      currentMeta.ref_binding_residues.forEach(([resn, resi, chain]) => {
        if (METALS.has(resn)) return;
        styleResidue(resn, resi, chain, 0, "greenCarbon", REF_GREEN);
        labelResidue(resn, resi, chain, 0, REF_GREEN);
      });
    }
    if (showHitResidues) {
      currentMeta.hit_binding_residues.forEach(([resn, resi, chain]) => {
        if (METALS.has(resn)) return;
        styleResidue(resn, resi, chain, 1, "cyanCarbon", HIT_CYAN);
        labelResidue(resn, resi, chain, 1, HIT_CYAN);
      });
    }

    // All metal ions in the pocket, not just the ones flagged as a binding
    // residue above (a pocket can have a metal that's structurally present
    // but didn't happen to register a non-VdW ProLIF interaction -- still
    // worth seeing). This is the single, exclusive place metals get styled
    // and labeled, so this toggle alone always controls all of them.
    const showMetals = document.getElementById("tog-metals").checked;
    if (showMetals) {
      const metalList = [...METALS];
      v.setStyle({ model: 0, resn: metalList }, { sphere: { scale: 0.35, color: REF_GREEN } });
      v.setStyle({ model: 1, resn: metalList }, { sphere: { scale: 0.35, color: HIT_CYAN } });
      [0, 1].forEach((model) => {
        v.selectedAtoms({ model, resn: metalList }).forEach((a) => {
          labelResidue(a.resn, a.resi, a.chain, model, model === 0 ? REF_GREEN : HIT_CYAN);
        });
      });
    }

    // Dashed lines from the exact interacting ligand atom to the exact
    // interacting residue ATOM (protAtomName -- e.g. an Arg's NH1, a Ser's
    // OG), not the residue's CA "root": a salt bridge or H-bond has a real,
    // specific atom on both ends, and anchoring at CA made every line point
    // at the backbone regardless of which side of the residue actually
    // reaches the ligand. Falls back to the CA/metal anchor only if
    // protAtomName is missing (e.g. an older pool build before this field
    // existed). Same green/cyan side colors as everything else, so "which
    // side" never depends on reading text.
    function drawInteractionLine(atomName, resn, resi, chain, protAtomName, ligModel, ligResn, ligResi, ligChain, color) {
      const ligAtoms = v.selectedAtoms({
        model: ligModel, resn: ligResn, resi: ligResi, chain: ligChain, atom: [atomName],
      });
      const resSel = protAtomName
        ? { model: ligModel, resn, resi, chain, atom: [protAtomName] }
        : residueAnchorSel(ligModel, resn, resi, chain);
      let resAtoms = v.selectedAtoms(resSel);
      if (!resAtoms.length && protAtomName) {
        // Named protein atom not found in this structure (e.g. altloc/atom
        // naming mismatch) -- fall back rather than silently dropping the
        // line entirely.
        resAtoms = v.selectedAtoms(residueAnchorSel(ligModel, resn, resi, chain));
      }
      if (!ligAtoms.length || !resAtoms.length) return;
      const a = ligAtoms[0], b = resAtoms[0];
      v.addLine({
        start: { x: a.x, y: a.y, z: a.z },
        end: { x: b.x, y: b.y, z: b.z },
        color, dashed: true, dashLength: 0.4, gapLength: 0.3, linewidth: 2,
      });
    }

    const showRefInteractions = document.getElementById("tog-ref-interactions").checked;
    const showHitInteractions = document.getElementById("tog-hit-interactions").checked;
    if (showRefInteractions) {
      (currentMeta.ref_atom_links || []).forEach(([atomName, resn, resi, chain, protAtomName]) => {
        drawInteractionLine(
          atomName, resn, resi, chain, protAtomName, 0,
          currentMeta.ref_resname, currentMeta.ref_resnum, currentMeta.ref_chain, REF_GREEN,
        );
      });
    }
    if (showHitInteractions) {
      (currentMeta.hit_atom_links || []).forEach(([atomName, resn, resi, chain, protAtomName]) => {
        drawInteractionLine(
          atomName, resn, resi, chain, protAtomName, 1,
          currentMeta.hit_resname, currentMeta.hit_resnum, currentMeta.hit_chain, HIT_CYAN,
        );
      });
    }

    // keepH:true (see loadPairIntoViewer) keeps hydrogens in the model so
    // the interaction lines above can anchor on a real interacting H (an
    // Arg's HH12, a Lys's HZ2/HZ3) -- but every stick style set above
    // (ligand, binding residues, "nearby" residues) selects by resn/resi/
    // chain only, with no element filter, so those hydrogens inherit a
    // visible stick too unless something clears it. This override runs last,
    // after every style block above, and unconditionally wins per-atom
    // (3Dmol's setStyle replaces, doesn't merge) -- an empty style hides
    // them without touching the underlying atoms selectedAtoms()/the
    // interaction lines still read from.
    const showHydrogens = document.getElementById("tog-hydrogens").checked;
    if (!showHydrogens) {
      v.setStyle({ elem: "H" }, {});
    }

    // A measurement in progress has its own line/label, wiped along with
    // everything else by this function's earlier removeAllShapes()/
    // removeAllLabels() -- redraw it here so toggling an unrelated checkbox
    // doesn't silently make it vanish while the text readout still shows it.
    if (state.measureAtoms.length === 2) {
      drawMeasurement();
    }

    v.render();
  }

  document.getElementById("submit-btn").addEventListener("click", async () => {
    if ((!state.selectedScore && !state.wrongPhosphate) || !state.currentPairId) return;
    const btn = document.getElementById("submit-btn");
    btn.disabled = true;
    const errEl = document.getElementById("rating-error");
    errEl.textContent = "";
    try {
      const res = await fetch("/api/rating", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          reviewer_name: state.reviewerName,
          pair_id: state.currentPairId,
          score: state.wrongPhosphate ? null : state.selectedScore,
          wrong_reference_phosphate: state.wrongPhosphate,
          comment: document.getElementById("comment").value || null,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        errEl.textContent = body.detail || "Could not submit rating.";
        btn.disabled = false;
        return;
      }
      const data = await res.json();
      state.ratedSet.add(state.currentPairId);
      updateProgress(data.done, data.total);
      await loadNextPair();
    } catch (e) {
      errEl.textContent = "Network error submitting rating.";
      btn.disabled = false;
    }
  });

  document.getElementById("debug-fetch-btn").addEventListener("click", async () => {
    const out = document.getElementById("debug-output");
    const token = document.getElementById("debug-token").value;
    if (!state.currentPairId) { out.textContent = "No pair loaded yet."; return; }
    if (!token) { out.textContent = "Enter the admin token first."; return; }
    out.textContent = "Loading...";
    try {
      const res = await fetch(`/api/debug/pair/${encodeURIComponent(state.currentPairId)}?token=${encodeURIComponent(token)}`);
      const body = await res.json();
      if (!res.ok) { out.textContent = `Error: ${body.detail || res.status}`; return; }
      out.textContent =
        `pair_id: ${state.currentPairId}\n` +
        `prolif_plif_score_no_vdw:   ${body.prolif_plif_score_no_vdw}\n` +
        `prolif_plif_score_with_vdw: ${body.prolif_plif_score_with_vdw}\n` +
        `ref_n_bits_novdw (min_ref_int): ${body.ref_n_bits_novdw}\n` +
        `pocket_rmsd: ${body.pocket_rmsd}`;
    } catch (e) {
      out.textContent = "Network error.";
    }
  });

  loadConfig();
})();
