/* The break-even, again, in the browser: smallprint/bench/breakeven.py line for line, so the
   calculator can take the reader's own prices. tests/test_site.py runs this file under Node
   against the Python function over a grid of inputs, including the "never" case and volumes
   that need more than one card, so the two cannot drift apart.

   GPUs needed    k(V) = max(1, ceil(V / capacity)),  capacity = rps * 3600 * H * u
   self-hosted    S(V) = k(V) * rate * H + fixed
   API            A(V) = V * api cost per call

   The break-even is the smallest V with A(V) >= S(V); null where one card at that
   utilisation already costs more per call than the API does. */

"use strict";

(function (root) {
  function usdPerCall(usdPerHour, requestsPerSecond, utilisation, secondsPerHour) {
    if (!(utilisation > 0 && utilisation <= 1)) throw new RangeError("utilisation must be in (0, 1]");
    if (!(usdPerHour > 0 && requestsPerSecond > 0)) throw new RangeError("the rate and the throughput must both be positive");
    return usdPerHour / (requestsPerSecond * secondsPerHour * utilisation);
  }

  function capacity(inputs, utilisation) {
    return inputs.requestsPerSecond * inputs.secondsPerHour * inputs.hoursPerMonth * utilisation;
  }

  function selfHostedMonthly(inputs, volume, utilisation) {
    if (volume < 0) throw new RangeError("volume cannot be negative");
    const gpus = Math.max(1, Math.ceil(volume / capacity(inputs, utilisation)));
    return gpus * inputs.usdPerHour * inputs.hoursPerMonth + inputs.fixedUsdPerMonth;
  }

  function breakEven(inputs, utilisation) {
    const perCall = usdPerCall(inputs.usdPerHour, inputs.requestsPerSecond, utilisation, inputs.secondsPerHour);
    const api = inputs.apiUsdPerCall;
    if (!(api > 0)) throw new RangeError("the API cost per call must be positive");
    const point = { utilisation, selfHostedUsdPerCall: perCall, apiUsdPerCall: api, volumePerMonth: null, gpus: null, monthlyCost: null };
    const cap = capacity(inputs, utilisation);
    const gpuMonth = inputs.usdPerHour * inputs.hoursPerMonth;
    for (let gpus = 1; ; gpus++) {
      const needed = (gpus * gpuMonth + inputs.fixedUsdPerMonth) / api;
      const bandLow = gpus === 1 ? 0 : (gpus - 1) * cap;
      if (needed <= gpus * cap) {
        const volume = Math.max(needed, bandLow);
        return Object.assign(point, { volumePerMonth: volume, gpus, monthlyCost: selfHostedMonthly(inputs, volume, utilisation) });
      }
      if (perCall >= api) return point;
    }
  }

  const api = { usdPerCall, capacity, selfHostedMonthly, breakEven };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Breakeven = api;
})(this);
