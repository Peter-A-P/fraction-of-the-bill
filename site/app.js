/* The page, drawn from results.json. No framework and no build step: the page is a static
   file a stranger can read, and its script should be too.

   Nothing here is a number of its own. The figures come from results.json, which
   `smallprint site` writes from the same runs as the repository's tables; the break-even
   arithmetic is breakeven.js, which is checked against the Python it copies. */

"use strict";

(function () {
  const SVG = "http://www.w3.org/2000/svg";
  const B = window.Breakeven;

  // ------------------------------------------------------------------ formatting

  const pct = (x) => (x * 100).toFixed(1) + "%";

  function usd(x) {
    if (x >= 100) return "US$" + Math.round(x).toLocaleString("en-US");
    if (x >= 1) return "US$" + x.toFixed(2);
    if (x >= 0.1) return "US$" + x.toFixed(2);
    return "US$" + x.toFixed(3);
  }

  function money(x) {
    return "US$" + Math.round(x).toLocaleString("en-US");
  }

  // A volume as people say it: to the nearest thousand once it is in thousands.
  function volume(v) {
    if (v === null || v === undefined) return "never";
    if (v >= 1e6) return (v / 1e6).toFixed(v >= 1e7 ? 0 : 2).replace(/\.?0+$/, "") + " million";
    if (v >= 1e4) return (Math.round(v / 1e3) * 1e3).toLocaleString("en-US");
    if (v >= 1e3) return (Math.round(v / 100) * 100).toLocaleString("en-US");
    return Math.round(v).toLocaleString("en-US");
  }

  const gpuName = (gpu) => gpu.replace(/^RTX(?=[A-Z0-9])/, "RTX ");
  const size = (s) => s.toUpperCase();

  function prompt(p) {
    if (p === "few-shot") return "two-shot";
    return p;
  }

  // "openai/gpt-5.6-luna zero_shot" -> "gpt-5.6-luna, zero-shot"
  function frontierName(f) {
    return f.model.split("/").pop() + ", " + prompt(f.prompt);
  }

  function servedName(s) {
    return size(s.size) + " " + s.format + " on " + gpuName(s.gpu);
  }

  function el(tag, attrs, parent, text) {
    const node = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
    if (text !== undefined) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }

  function html(tag, attrs, parent, text) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
    if (text !== undefined) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }

  const byId = (id) => document.getElementById(id);
  const setText = (id, text) => { const n = byId(id); if (n) n.textContent = text; };

  // ------------------------------------------------------------------ the data

  function inputsFor(data, served, apiPer1000, rate, fixed) {
    return {
      usdPerHour: rate,
      requestsPerSecond: served.requests_per_second,
      apiUsdPerCall: apiPer1000 / 1000,
      fixedUsdPerMonth: fixed,
      hoursPerMonth: data.hours_per_month,
      secondsPerHour: data.seconds_per_hour,
    };
  }

  function curvePoint(data, servedIndex, against, utilisation) {
    const c = data.curves.find((c) => c.served === servedIndex && c.against === against);
    if (!c) return null;
    return c.points.find((p) => Math.abs(p.utilisation - utilisation) < 1e-9) || null;
  }

  // The headline model: the smallest size served in bf16, on its cheapest card.
  function heroIndex(data) {
    let best = -1;
    data.served.forEach((s, i) => {
      if (s.format !== "bf16") return;
      const b = data.served[best];
      if (best < 0 || s.size < b.size || (s.size === b.size && s.usd_per_1000 < b.usd_per_1000)) best = i;
    });
    return best;
  }

  // ------------------------------------------------------------------ the hero

  function hero(data) {
    const anchor = data.frontier.find((f) => f.key === data.anchor);
    const ceiling = data.frontier.find((f) => f.key === data.ceiling);
    const i = heroIndex(data);
    const s = data.served[i];
    const u = data.quoted_utilisation;
    setText("hero-accuracy", pct(s.accuracy.point));
    setText("hero-accuracy-note",
      "on " + data.filings.toLocaleString("en-US") + " filings published after it was trained; " +
      frontierName(ceiling) + ", the dearest frontier result, got " + pct(ceiling.accuracy.point));
    setText("hero-cost", usd(s.usd_per_1000));
    setText("hero-cost-note",
      "on one " + gpuName(s.gpu) + " at " + usd(s.usd_per_hour) + " an hour, busy " + pct(u).replace(".0", "") +
      " of the time; " + frontierName(anchor) + " is " + usd(anchor.usd_per_1000));
    const even = curvePoint(data, i, data.anchor, u);
    const evenCeiling = curvePoint(data, i, data.ceiling, u);
    setText("hero-even", even ? volume(even.volume_per_month) : "-");
    setText("hero-even-note", "against " + frontierName(anchor) + ", the cheapest API that does the job");
    setText("hero-even-ceiling", evenCeiling ? volume(evenCeiling.volume_per_month) : "-");
    setText("hero-even-ceiling-note", "against " + frontierName(ceiling) + ", the frontier result it matches");
    setText("task-filings", data.filings.toLocaleString("en-US"));
    document.querySelectorAll(".conc").forEach((n) => { n.textContent = String(data.concurrency); });
    document.querySelectorAll(".util").forEach((n) => {
      n.textContent = u === 0.5 ? "half" : pct(u).replace(".0", "") + " of";
    });
    setText("foot-written", "Results file written " + data.written + " from the runs");
  }

  // ------------------------------------------------------------------ quality against cost

  function front(points) {
    // The marks no other mark beats on both axes: no dearer, at least as accurate, one strictly.
    return points.filter((p) => !points.some((q) => q !== p &&
      q.cost <= p.cost && q.acc >= p.acc && (q.cost < p.cost || q.acc > p.acc)));
  }

  function pareto(data) {
    const points = [
      ...data.frontier.map((f) => ({
        hosted: false, cost: f.usd_per_1000, acc: f.accuracy.point, low: f.accuracy.low, high: f.accuracy.high,
        label: f.model.split("/").pop() + " " + prompt(f.prompt), key: f.key,
        detail: frontierName(f) + ": " + pct(f.accuracy.point) + " of fields (" + pct(f.accuracy.low) + " to " +
          pct(f.accuracy.high) + "), " + usd(f.usd_per_1000) + " per 1,000, as billed" + (f.recosted ? ", recomputed at the cache-write rate" : ""),
      })),
      ...data.served.filter((s) => s.accuracy).map((s) => ({
        hosted: true, cost: s.usd_per_1000, acc: s.accuracy.point, low: s.accuracy.low, high: s.accuracy.high,
        label: size(s.size) + " " + s.format + ", " + gpuName(s.gpu), key: s.name + s.format + s.gpu,
        detail: servedName(s) + ": " + pct(s.accuracy.point) + " of fields (" + pct(s.accuracy.low) + " to " +
          pct(s.accuracy.high) + "), " + usd(s.usd_per_1000) + " per 1,000 at " + usd(s.usd_per_hour) +
          " an hour and " + s.requests_per_second.toFixed(2) + " requests a second",
      })),
    ];
    const W = 760, H = 440, L = 64, R = 24, T = 24, Bm = 56;
    const lo = Math.floor(Math.log10(Math.min(...points.map((p) => p.cost))));
    const hi = Math.ceil(Math.log10(Math.max(...points.map((p) => p.cost))));
    const yLo = Math.floor(Math.min(...points.map((p) => p.low)) * 200) / 200;
    const yHi = Math.ceil(Math.max(...points.map((p) => p.high)) * 200) / 200;
    const x = (c) => L + (Math.log10(c) - lo) / (hi - lo) * (W - L - R);
    const y = (a) => T + (yHi - a) / (yHi - yLo) * (H - T - Bm);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": "Share of fields right against cost per 1,000 filings, every frontier API and every served fine-tune" });
    for (let d = lo; d <= hi; d++) {
      el("line", { x1: x(10 ** d), x2: x(10 ** d), y1: T, y2: H - Bm, class: "gridline" }, svg);
      el("text", { x: x(10 ** d), y: H - Bm + 18, "text-anchor": "middle", class: "tick" }, svg,
        "US$" + (d < 0 ? (10 ** d).toFixed(-d) : (10 ** d).toLocaleString("en-US")));
    }
    for (let a = yLo; a <= yHi + 1e-9; a += 0.005) {
      el("line", { x1: L, x2: W - R, y1: y(a), y2: y(a), class: "gridline" }, svg);
      el("text", { x: L - 8, y: y(a) + 4, "text-anchor": "end", class: "tick" }, svg, pct(a));
    }
    el("text", { x: (L + W - R) / 2, y: H - 14, "text-anchor": "middle", class: "axis-title" }, svg, "Cost per 1,000 filings, log scale");
    el("text", { x: 14, y: (T + H - Bm) / 2, "text-anchor": "middle", class: "axis-title",
      transform: `rotate(-90 14 ${(T + H - Bm) / 2})` }, svg, "Fields right");
    const labelled = new Set(front(points).map((p) => p.key).concat([data.anchor, data.ceiling]));
    const readout = byId("pareto-readout");
    const marks = [];
    const labels = [];
    points.forEach((p) => {
      const g = el("g", { tabindex: "0" }, svg);
      el("title", {}, g, p.detail);
      el("line", { x1: x(p.cost), x2: x(p.cost), y1: y(p.low), y2: y(p.high), class: "whisker" }, g);
      const mark = p.hosted
        ? el("path", { d: `M${x(p.cost)} ${y(p.acc) - 7}l7 7l-7 7l-7 -7z`, class: "hosted-mark" }, g)
        : el("circle", { cx: x(p.cost), cy: y(p.acc), r: 5.5, class: "frontier-mark" }, g);
      marks.push(mark);
      const show = () => { marks.forEach((m) => m.classList.remove("on")); mark.classList.add("on"); readout.textContent = p.detail; };
      g.addEventListener("mouseenter", show);
      g.addEventListener("focus", show);
      g.addEventListener("click", show);
      if (labelled.has(p.key)) labels.push({ p, g });
    });
    // Labels sit beside their mark, nudged apart where two marks are close enough to collide.
    // Each label tries right of its mark, then left, then above and below, and takes the first
    // spot that covers no mark, no whisker and no label already placed; failing that, the first
    // that covers no mark and no label, crossing a whisker.
    const marksBox = points.map((p) => ({ x0: x(p.cost) - 9, x1: x(p.cost) + 9, y0: y(p.acc) - 9, y1: y(p.acc) + 9 }));
    const whiskers = points.map((p) => ({ x0: x(p.cost) - 3, x1: x(p.cost) + 3, y0: y(p.high), y1: y(p.low) }));
    const boxes = marksBox;
    const over = (b, list) => list.some((o) => b.x0 < o.x1 && o.x0 < b.x1 && b.y0 < o.y1 && o.y0 < b.y1);
    const outside = (b) => b.x0 < L + 4 || b.x1 > W - R || b.y0 < T || b.y1 > H - Bm;
    labels.sort((a, b) => y(a.p.acc) - y(b.p.acc)).forEach(({ p, g }) => {
      const cx = x(p.cost), cy = y(p.acc), w = p.label.length * 6.6, h = 14;
      const tries = [];
      for (const dy of [0, -16, 16, -32, 32, -48, 48]) {
        tries.push({ x0: cx + 12, y0: cy - h / 2 + dy, anchor: "start" });
        tries.push({ x0: cx - 12 - w, y0: cy - h / 2 + dy, anchor: "end" });
      }
      const boxed = tries.map((t) => ({ ...t, x1: t.x0 + w, y1: t.y0 + h })).filter((b) => !outside(b) && !over(b, boxes));
      const spot = boxed.find((b) => !over(b, whiskers)) || boxed[0] ||
        { x0: cx + 12, y0: cy - h / 2, x1: cx + 12 + w, y1: cy + h / 2, anchor: "start" };
      boxes.push(spot);
      el("text", { x: spot.anchor === "start" ? spot.x0 : spot.x1, y: spot.y0 + h - 3, "text-anchor": spot.anchor,
        class: "mark-label" + (p.hosted ? " strong" : "") }, g, p.label);
    });
    byId("pareto").appendChild(svg);
    setText("pareto-caption",
      "Fine-tuned cost is the rent over the throughput at " + data.concurrency + " requests in flight, the card busy " +
      pct(data.quoted_utilisation).replace(".0", "") + " of the time. Labelled: the marks nothing beats on both cost and accuracy, " +
      "and the cheapest and the most accurate API. Every mark carries its figures on hover, focus or tap.");

    const anchor = data.frontier.find((f) => f.key === data.anchor);
    const ceiling = data.frontier.find((f) => f.key === data.ceiling);
    const s = data.served[heroIndex(data)];
    const verdict = byId("frontier-verdict");
    html("p", {}, verdict,
      "The cheapest API, " + frontierName(anchor) + ", gets " + pct(anchor.accuracy.point) + " of fields right for " +
      usd(anchor.usd_per_1000) + " per 1,000; the dearest result, " + frontierName(ceiling) + ", " +
      pct(ceiling.accuracy.point) + " for " + usd(ceiling.usd_per_1000) + ". The " + size(s.size) +
      " fine-tune gets " + pct(s.accuracy.point) + " for " + usd(s.usd_per_1000) + " on one " + gpuName(s.gpu) +
      ": the dearest model's accuracy at about " + fraction(s.usd_per_1000 / anchor.usd_per_1000) +
      " of the cheapest one's price.");
  }

  // "a twelfth", "a fifth": how a reader says a ratio below one half.
  function fraction(r) {
    const n = Math.max(2, Math.round(1 / r));
    const names = { 2: "half", 3: "third", 4: "quarter", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth",
      9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth", 15: "fifteenth", 16: "sixteenth", 20: "twentieth" };
    return names[n] ? "a " + names[n] : "1/" + n;
  }

  // ------------------------------------------------------------------ the calculator

  function calculator(data) {
    const served = byId("calc-served");
    const rate = byId("calc-rate");
    const api = byId("calc-api");
    const fixed = byId("calc-fixed");
    const util = byId("calc-util");
    const utilOut = byId("calc-util-out");
    const vol = byId("calc-volume");
    const presets = byId("calc-api-presets");

    data.served.forEach((s, i) => {
      html("option", { value: String(i) }, served,
        servedName(s) + " (" + s.requests_per_second.toFixed(2) + " req/s)");
    });
    served.value = String(heroIndex(data));

    const anchor = data.frontier.find((f) => f.key === data.anchor);
    api.value = anchor.usd_per_1000.toFixed(2);
    const chips = data.frontier.map((f) => {
      const b = html("button", { type: "button", class: "chip", "aria-pressed": "false" }, presets,
        frontierName(f) + ", " + usd(f.usd_per_1000));
      b.addEventListener("click", () => { api.value = f.usd_per_1000.toFixed(2); update(); });
      return { b, f };
    });

    function setRate() {
      const s = data.served[Number(served.value)];
      rate.value = s.usd_per_hour.toFixed(2);
      setText("calc-rate-hint", gpuName(s.gpu) + " at " + s.provider.charAt(0).toUpperCase() + s.provider.slice(1) + ", " + s.price_checked);
    }
    setRate();

    served.addEventListener("change", () => { setRate(); update(); });
    [rate, api, fixed, util, vol].forEach((n) => n.addEventListener("input", update));

    function number(n, fallback) {
      const v = parseFloat(n.value);
      return Number.isFinite(v) ? v : fallback;
    }

    function update() {
      const s = data.served[Number(served.value)];
      const u = Number(util.value) / 100;
      utilOut.textContent = util.value + "%";
      const r = number(rate, s.usd_per_hour);
      const a = number(api, anchor.usd_per_1000);
      const f = Math.max(0, number(fixed, 0));
      const v = Math.max(0, number(vol, 0));
      chips.forEach(({ b, f: fr }) => b.setAttribute("aria-pressed", String(Math.abs(fr.usd_per_1000 - a) < 0.005)));
      const out = byId("out-even");
      if (!(r > 0 && a > 0)) {
        out.textContent = "-";
        setText("out-even-note", "the rent and the API price must both be more than zero");
        return;
      }
      const inputs = inputsFor(data, s, a, r, f);
      const p = B.breakEven(inputs, u);
      out.classList.toggle("never", p.volumePerMonth === null);
      if (p.volumePerMonth === null) {
        out.textContent = "Never";
        setText("out-even-note", "at this load one card costs " + usd(p.selfHostedUsdPerCall * 1000) +
          " per 1,000, more than the API, so no volume pays; run it busier or rent cheaper");
      } else {
        out.textContent = volume(p.volumePerMonth);
        setText("out-even-note", "calls a month and the card is the cheaper" +
          (p.gpus > 1 ? ", on " + p.gpus + " cards" : "") + "; one card serves up to " +
          volume(B.capacity(inputs, u)) + " a month at this load");
      }
      setText("out-own", usd(p.selfHostedUsdPerCall * 1000));
      setText("out-own-note", "per 1,000 in rent on the card, busy " + util.value + "% of the time" +
        (f > 0 ? ", before the fixed costs" : ""));
      const monthApi = v * a / 1000;
      const monthOwn = B.selfHostedMonthly(inputs, v, u);
      setText("out-month-api", money(monthApi));
      setText("out-month-api-note", "a month on the API at " + volume(v) + " calls");
      setText("out-month-own", money(monthOwn));
      const cards = Math.max(1, Math.ceil(v / B.capacity(inputs, u)));
      setText("out-month-own-note", "a month on " + (cards === 1 ? "one card" : cards + " cards") +
        (monthOwn < monthApi ? ", saving " + money(monthApi - monthOwn) : monthOwn > monthApi ? ", " + money(monthOwn - monthApi) + " more than the API" : ""));
      monthChart(inputs, u, p, v);
    }
    update();
    let width = window.innerWidth;
    window.addEventListener("resize", () => { if (window.innerWidth !== width) { width = window.innerWidth; update(); } });
  }

  function niceMax(x) {
    const p = 10 ** Math.floor(Math.log10(x));
    for (const m of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * p >= x) return m * p;
    return 10 * p;
  }

  function monthChart(inputs, u, p, v) {
    const host = byId("month");
    host.textContent = "";
    const cap = B.capacity(inputs, u);
    const reach = Math.max(p.volumePerMonth !== null ? p.volumePerMonth * 2 : cap * 2, v * 1.25, 1000);
    const xMax = niceMax(reach);
    const yMax = niceMax(Math.max(xMax * inputs.apiUsdPerCall, B.selfHostedMonthly(inputs, xMax, u)) * 1.05);
    // Drawn at the width it is shown at, so its type stays legible on a phone.
    const W = Math.max(320, Math.min(760, host.clientWidth || 760)), H = W < 500 ? 300 : 340;
    const L = 76, R = 24, T = 20, Bm = 50;
    const ticks = W < 500 ? 4 : 5;
    const x = (vv) => L + vv / xMax * (W - L - R);
    const y = (c) => T + (1 - c / yMax) * (H - T - Bm);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": "Monthly cost on the API and on rented cards against monthly volume" });
    for (let i = 0; i <= ticks; i++) {
      const vv = xMax * i / ticks, c = yMax * i / ticks;
      el("line", { x1: L, x2: W - R, y1: y(c), y2: y(c), class: "gridline" }, svg);
      el("text", { x: L - 8, y: y(c) + 4, "text-anchor": "end", class: "tick" }, svg, money(c));
      el("text", { x: x(vv), y: H - Bm + 18, "text-anchor": "middle", class: "tick" }, svg, volume(vv));
    }
    el("text", { x: (L + W - R) / 2, y: H - 10, "text-anchor": "middle", class: "axis-title" }, svg, "Calls a month");
    el("line", { x1: x(0), y1: y(0), x2: x(xMax), y2: y(xMax * inputs.apiUsdPerCall), class: "api-line" }, svg);
    // The staircase: flat while k cards cover the volume, one card's month higher after.
    let d = "", k = 1;
    let start = 0;
    while (start < xMax && k < 500) {
      const end = Math.min(k * cap, xMax);
      const c = k * inputs.usdPerHour * inputs.hoursPerMonth + inputs.fixedUsdPerMonth;
      d += (d ? "L" : "M") + x(start) + " " + y(Math.min(c, yMax)) + "L" + x(end) + " " + y(Math.min(c, yMax));
      start = end;
      k += 1;
    }
    el("path", { d, class: "own-line" }, svg);
    if (p.volumePerMonth !== null && p.volumePerMonth <= xMax) {
      const ex = x(p.volumePerMonth);
      el("line", { x1: ex, x2: ex, y1: T, y2: H - Bm, class: "even-mark" }, svg);
      el("text", { x: ex + 6, y: T + 12, class: "even-label" }, svg, "break-even, " + volume(p.volumePerMonth));
    }
    if (v > 0 && v <= xMax) {
      const vx = x(v);
      el("line", { x1: vx, x2: vx, y1: T + 18, y2: H - Bm, class: "you-mark" }, svg);
      el("text", { x: vx + 6, y: T + 30, class: "you-label" }, svg, "you");
    }
    host.appendChild(svg);
  }

  // ------------------------------------------------------------------ the table

  function table(data) {
    const chips = byId("table-against");
    const utils = data.curves.length ? data.curves[0].points.map((p) => p.utilisation) : [];
    let against = data.anchor;
    const choices = [data.anchor, data.ceiling].filter((k, i, a) => a.indexOf(k) === i);
    const buttons = choices.map((k) => {
      const f = data.frontier.find((ff) => ff.key === k);
      const b = html("button", { type: "button", class: "chip" }, chips,
        "Against " + frontierName(f) + ", " + usd(f.usd_per_1000) + " per 1,000");
      b.addEventListener("click", () => { against = k; draw(); });
      return { b, k };
    });

    function draw() {
      buttons.forEach(({ b, k }) => b.setAttribute("aria-pressed", String(k === against)));
      const t = byId("even-table");
      t.textContent = "";
      const head = html("tr", {}, html("thead", {}, t));
      ["Model", "Format", "Card, US$ an hour", "Req/s", "Per 1,000 at " + pct(data.quoted_utilisation).replace(".0", "")]
        .forEach((h, i) => html("th", { class: i > 2 ? "num" : "", scope: "col" }, head, h));
      utils.forEach((u) => html("th", { class: "num", scope: "col" }, head, pct(u).replace(".0", "") + " busy"));
      const body = html("tbody", {}, t);
      data.served.forEach((s, i) => {
        const tr = html("tr", {}, body);
        html("th", { scope: "row" }, tr, size(s.size));
        html("td", {}, tr, s.format);
        html("td", {}, tr, gpuName(s.gpu) + ", " + s.usd_per_hour.toFixed(2) + " (" + s.price_checked + ")");
        html("td", { class: "num" }, tr, s.requests_per_second.toFixed(2));
        html("td", { class: "num" }, tr, usd(s.usd_per_1000));
        utils.forEach((u) => {
          const p = curvePoint(data, i, against, u);
          const vv = p ? p.volume_per_month : null;
          html("td", { class: "num" + (vv === null ? " never" : "") }, tr, volume(vv));
        });
      });
      setText("table-note",
        "While one card carries the volume, the break-even is a month's rent over the API's price a call, " +
        "so it is the same for every model on the same card until the load is too light for one card to beat " +
        "the API at all. Throughput measured at " + data.concurrency + " requests in flight; a month is " +
        data.hours_per_month + " hours.");
    }
    draw();
  }

  // ------------------------------------------------------------------ the gate

  function gate(data) {
    const t = byId("gate-table");
    const head = html("tr", {}, html("thead", {}, t));
    ["Fine-tune", "Against", "Decision", "Fields where a three-point loss could not be ruled out"]
      .forEach((h) => html("th", { scope: "col" }, head, h));
    const body = html("tbody", {}, t);
    const frontierFor = (b) => data.frontier.find((f) => f.model.endsWith("-" + b) &&
      (f.key === data.anchor || f.key === data.ceiling));
    const sizeOf = (c) => c.split("-")[0];
    data.gates.forEach((g) => {
      const tr = html("tr", {}, body);
      const f = frontierFor(g.baseline);
      html("th", { scope: "row" }, tr, size(sizeOf(g.candidate)) + " bf16");
      html("td", {}, tr, f ? frontierName(f) : g.baseline);
      html("td", { class: "verdict-" + g.verdict }, tr, g.verdict === "pass" ? "Passes all " + g.suites : "Blocked");
      html("td", { class: "fields" }, tr, g.blocked.length ? g.blocked.join(", ").replace(/_/g, " ") : "none");
    });
    const passes = data.gates.filter((g) => g.verdict === "pass");
    const ceilingPassed = passes.some((g) => { const f = frontierFor(g.baseline); return f && f.key === data.ceiling; });
    const ceiling = data.frontier.find((f) => f.key === data.ceiling);
    const v = byId("gate-verdict");
    html("p", {}, v,
      (passes.length
        ? passes.map((g) => "The " + size(sizeOf(g.candidate)) + " passes against " +
            (frontierFor(g.baseline) ? frontierName(frontierFor(g.baseline)) : g.baseline)).join("; ") +
          " on every field. "
        : "No fine-tune passes on every field. ") +
      "Where one is blocked, it is on a line or two where the loss could be more than three points, " +
      "even when the model is ahead on the average of all fifteen" +
      (ceilingPassed ? "." : ", and none passes against " + frontierName(ceiling) + ".") +
      " If those lines are the ones that matter to you, that is the trade, stated.");
  }

  // ------------------------------------------------------------------ start

  fetch("results.json", { cache: "no-cache" })
    .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then((data) => { hero(data); pareto(data); calculator(data); table(data); gate(data); })
    .catch(() => {
      setText("hero-note", "The results file did not load, so the figures on this page are missing. " +
        "Every number is also in the repository's README.");
    });
})();
