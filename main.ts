import * as fs from 'node:fs';
import * as path from 'node:path';

// Refresh-cohort model for GTA VI trophy earn rate forecasting.
// Exophase refreshes cause discrete jumps in tracked players.
// Each refresh brings a new cohort with its own earn rates s_i.
// We estimate s_i from historical refreshes (recency-weighted),
// extrapolate N(target) via log-log growth, project earned counts,
// and produce logit-normal quantiles with backtested spread.

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
const LAUNCH_EPOCH = new Date('2026-11-19T00:00:00Z').getTime();
const MS_PER_HOUR = 3600_000;

function logit(p: number): number {
  const clamped = Math.max(0.01, Math.min(99.99, p));
  return Math.log(clamped / (100 - clamped));
}

function invLogit(l: number): number {
  const p = 100 / (1 + Math.exp(-l));
  return Math.max(0.01, Math.min(99.99, p));
}

function hoursSinceLaunch(ts: string): number {
  return (new Date(ts).getTime() - LAUNCH_EPOCH) / MS_PER_HOUR;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

function readInputs(): any[] {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || '/input';
  const files = fs.readdirSync(inputDir).filter(f => f.endsWith('.in')).sort();
  if (files.length === 0) throw new Error(`No .in files found in ${inputDir}`);
  return files.map(f => JSON.parse(fs.readFileSync(path.join(inputDir, f), 'utf8')));
}

function writeOutput(id: string, output: any): void {
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || '/output';
  fs.mkdirSync(outputDir, { recursive: true });
  fs.writeFileSync(path.join(outputDir, `${id}.out`), JSON.stringify(output));
}

function collapseStates(history: any[]): any[] {
  const distinct: any[] = [];
  for (const h of history) {
    if (distinct.length === 0) {
      distinct.push(h);
    } else {
      const last = distinct[distinct.length - 1];
      if (last.players_tracked !== h.players_tracked || 
          JSON.stringify(last.rates) !== JSON.stringify(h.rates)) {
        distinct.push(h);
      }
    }
  }
  return distinct;
}

function findRefreshes(states: any[]): number[] {
  const refreshes: number[] = [];
  for (let i = 1; i < states.length; i++) {
    if (states[i].players_tracked > states[i - 1].players_tracked * 1.001) {
      refreshes.push(i);
    }
  }
  return refreshes;
}

function getLastNonNullRate(trophyIdx: number, states: any[]): {rate: number, N: number} {
  for (let i = states.length - 1; i >= 0; i--) {
    const r = states[i].rates[trophyIdx];
    if (r !== null && r !== undefined) {
      return { rate: r, N: states[i].players_tracked };
    }
  }
  return { rate: 0, N: states[states.length - 1].players_tracked };
}

function computeEarned(rate: number | null, N: number): number {
  if (rate === null || rate === undefined) return 0;
  return (rate / 100) * N;
}

function forecastCase(input: any): void {
  const { id, target, trophies, history, notes } = input;
  const T = trophies.length;

  // Collapse duplicate states
  const states = collapseStates(history);
  const S = states.length;

  // Last state
  const lastState = states[S - 1];
  const N_last = lastState.players_tracked;
  
  // Last rates with null handling
  const lastRates: number[] = [];
  const lastEarned: number[] = [];
  for (let i = 0; i < T; i++) {
    const { rate, N } = getLastNonNullRate(i, states);
    lastRates[i] = rate;
    lastEarned[i] = computeEarned(rate, N);
  }

  // Find refreshes
  const refreshIndices = findRefreshes(states);
  const R = refreshIndices.length;

  let s_i: number[];
  let N_target: number;
  let isFlatHistory = false;
  let s_per_refresh: number[][] = [];

  if (R === 0) {
    // FLAT-HISTORY FALLBACK
    isFlatHistory = true;
    N_target = N_last * 6.06;
    s_i = new Array(T).fill(0);
    for (let i = 0; i < T; i++) {
      if (lastRates[i] >= 99.99) {
        s_i[i] = 100; // pinned at 100%
      }
    }
  } else {
    // Compute cohort earn rates per refresh
    s_per_refresh = [];
    for (const idx of refreshIndices) {
      const before = states[idx - 1];
      const after = states[idx];
      const N_before = before.players_tracked;
      const N_after = after.players_tracked;
      const deltaN = N_after - N_before;
      const s_k: number[] = [];
      for (let i = 0; i < T; i++) {
        const E_before = computeEarned(before.rates[i], N_before);
        const E_after = computeEarned(after.rates[i], N_after);
        let s = (E_after - E_before) / deltaN * 100; // as percent
        s = clamp(s, 0, 100);
        s_k.push(s);
      }
      s_per_refresh.push(s_k);
    }

    // Recency-weighted mean: weight 2^k where k=0 is oldest, k=R-1 newest
    s_i = new Array(T).fill(0);
    for (let i = 0; i < T; i++) {
      let num = 0, den = 0;
      for (let k = 0; k < R; k++) {
        const w = Math.pow(2, k);
        num += w * s_per_refresh[k][i];
        den += w;
      }
      s_i[i] = num / den;
    }

    // Extrapolate N(target) via log-log regression
    if (S >= 2) {
      const xs: number[] = [];
      const ys: number[] = [];
      for (const st of states) {
        const h = hoursSinceLaunch(st.at);
        if (h > 0) {
          xs.push(Math.log(1 + h));
          ys.push(Math.log(st.players_tracked));
        }
      }
      if (xs.length >= 2) {
        // Least squares: y = a + b*x
        const n = xs.length;
        const sumX = xs.reduce((a, b) => a + b, 0);
        const sumY = ys.reduce((a, b) => a + b, 0);
        const sumXY = xs.reduce((a, x, i) => a + x * ys[i], 0);
        const sumXX = xs.reduce((a, x) => a + x * x, 0);
        const b = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const a = (sumY - b * sumX) / n;
        const h_target = hoursSinceLaunch(target);
        const x_target = Math.log(1 + Math.max(0, h_target));
        const y_target = a + b * x_target;
        N_target = Math.exp(y_target);
        N_target = clamp(N_target, N_last, 50 * N_last);
      } else {
        N_target = N_last;
      }
    } else {
      N_target = N_last;
    }
  }

  // Compute E_i(target) and median m_i
  const m_i: number[] = [];
  for (let i = 0; i < T; i++) {
    const deltaN = Math.max(0, N_target - N_last);
    const E_target = lastEarned[i] + (s_i[i] / 100) * deltaN;
    let m = 100 * E_target / N_target;
    if (lastRates[i] >= 99.99) m = 99.99; // pinned
    m = clamp(m, 0.01, 99.99);
    m_i[i] = m;
  }

  // Compute sigma per trophy via backtest
  const sigma: number[] = [];
  
  if (isFlatHistory) {
    for (let i = 0; i < T; i++) {
      if (lastRates[i] >= 99.99) sigma[i] = 0.05;
      else if (lastRates[i] <= 0.05) sigma[i] = 0.9;
      else sigma[i] = 0.35;
    }
  } else {
    // Backtest each refresh (except first, which has no prior)
    const backtestErrors: number[][] = new Array(T).fill(null).map(() => []);
    
    for (let r = 1; r < R; r++) {
      // Use refreshes 0..r-1 to estimate s_i, predict refresh r
      const s_est: number[] = [];
      for (let i = 0; i < T; i++) {
        let num = 0, den = 0;
        for (let k = 0; k < r; k++) {
          const w = Math.pow(2, k);
          num += w * s_per_refresh[k][i];
          den += w;
        }
        s_est[i] = num / den;
      }
      
      const before = states[refreshIndices[r] - 1];
      const after = states[refreshIndices[r]];
      const N_before = before.players_tracked;
      const N_after = after.players_tracked;
      const deltaN = N_after - N_before;
      
      for (let i = 0; i < T; i++) {
        const E_before = computeEarned(before.rates[i], N_before);
        const E_pred = E_before + (s_est[i] / 100) * deltaN;
        const m_pred = 100 * E_pred / N_after;
        const m_clamped = clamp(m_pred, 0.01, 99.99);
        const actual = after.rates[i];
        if (actual !== null && actual !== undefined) {
          const err = logit(m_clamped) - logit(clamp(actual, 0.01, 99.99));
          backtestErrors[i].push(err);
        }
      }
    }

    const lnRatio = Math.min(3, Math.log(N_target / N_last));
    
    for (let i = 0; i < T; i++) {
      if (lastRates[i] >= 99.99) {
        sigma[i] = 0.05;
      } else {
        let sigma_bt = 0.6;
        if (backtestErrors[i].length > 0) {
          const sq = backtestErrors[i].reduce((a, e) => a + e * e, 0);
          sigma_bt = Math.sqrt(sq / backtestErrors[i].length);
        }
        let s = clamp(1.15 * sigma_bt + 0.1 * lnRatio, 0.55, 1.6);
        if (lastRates[i] <= 0.05) s = Math.max(s, 0.9);
        sigma[i] = s;
      }
    }
  }

  // Compute quantiles
  const resultTrophies = [];
  for (let i = 0; i < T; i++) {
    let q: number[];
    if (lastRates[i] >= 99.99) {
      q = [99.5, 99.9, 99.99, 99.99, 99.99];
    } else {
      const base = logit(m_i[i]);
      q = Z.map(z => invLogit(base + z * sigma[i]));
      // Enforce non-decreasing
      for (let j = 1; j < 5; j++) {
        if (q[j] < q[j - 1]) q[j] = q[j - 1];
      }
      q = q.map(round4);
    }
    resultTrophies.push({ id: trophies[i].id, q });
  }

  writeOutput(id, { trophies: resultTrophies });
}

function main(): void {
  const cases = readInputs();
  for (const c of cases) forecastCase(c);
}

main();
