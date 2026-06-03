import { OptionChainRow } from './bseApi';

export interface PcrData {
  pcr_oi: number;
  pcr_vol: number;
  sentiment: string;
  description: string;
  color: string;
  total_call_oi: number;
  total_put_oi: number;
  total_call_vol: number;
  total_put_vol: number;
}

export interface SupportResistance {
  strike: number;
  oi: number;
  ltp: number;
}

export interface Totals {
  total_ce_oi: number;
  total_pe_oi: number;
  total_oi: number;
  total_ce_vol: number;
  total_pe_vol: number;
  total_vol: number;
  total_ce_premium: number;
  total_pe_premium: number;
  total_premium: number;
  atm_strike: number;
  atm_ce_ltp: number;
  atm_pe_ltp: number;
  straddle_price: number;
  breakeven_up: number;
  breakeven_down: number;
  oi_skew: number;
  premium_skew: number;
  pcr_oi: number;
  pcr_vol: number;
  max_ce_strike: number | null;
  max_pe_strike: number | null;
}

export interface BiasFactor {
  name: string;
  vote: number;
  reason: string;
  bull_val: string;
  bear_val: string;
}

export interface PriceBias {
  factors: BiasFactor[];
  score: number;
  max_score: number;
  pct: number;
  verdict: string;
  color: string;
  action: string;
  emoji: string;
  bull_count: number;
  bear_count: number;
  neut_count: number;
}

export interface EnrichedRow extends OptionChainRow {
  Moneyness: string;
  'CE Concentration': string;
  'PE Concentration': string;
  'CE Vol Surge': string;
  'PE Vol Surge': string;
  'Strike PCR': number;
  Straddle: number;
  'CE Notional': number;
  'PE Notional': number;
  'Strike Bias': string;
  'IV Signal': string;
  'Dist from Spot': number;
  'Dist %': number;
}

export function fmt(value: number): string {
  return `₹${value.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export function fmtCr(value: number): string {
  if (value >= 1e7) return `₹${(value / 1e7).toFixed(2)} Cr`;
  if (value >= 1e5) return `₹${(value / 1e5).toFixed(2)} L`;
  return `₹${value.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
}

export function calculatePcrAnalysis(df: OptionChainRow[]): PcrData {
  const totalCeOi = df.reduce((s, r) => s + r['CE OI'], 0);
  const totalPeOi = df.reduce((s, r) => s + r['PE OI'], 0);
  const totalCeVol = df.reduce((s, r) => s + r['CE Volume'], 0);
  const totalPeVol = df.reduce((s, r) => s + r['PE Volume'], 0);
  const pcrOi = totalCeOi > 0 ? Math.round((totalPeOi / totalCeOi) * 100) / 100 : 0;
  const pcrVol = totalCeVol > 0 ? Math.round((totalPeVol / totalCeVol) * 100) / 100 : 0;
  let sentiment: string, description: string, color: string;
  if (pcrOi > 1.2) {
    sentiment = '📈 Bullish'; description = 'Strong Put Writing – Support Building'; color = 'green';
  } else if (pcrOi < 0.8) {
    sentiment = '📉 Bearish'; description = 'Strong Call Writing – Resistance Building'; color = 'red';
  } else {
    sentiment = '⚖️ Neutral'; description = 'Balanced Market Conditions'; color = 'orange';
  }
  return { pcr_oi: pcrOi, pcr_vol: pcrVol, sentiment, description, color, total_call_oi: totalCeOi, total_put_oi: totalPeOi, total_call_vol: totalCeVol, total_put_vol: totalPeVol };
}

export function findSupportResistance(df: OptionChainRow[], spotPrice: number, numLevels = 5) {
  const supports = df.filter(r => r['Strike Price'] <= spotPrice)
    .sort((a, b) => b['PE OI'] - a['PE OI'])
    .slice(0, numLevels)
    .map(r => ({ strike: r['Strike Price'], oi: r['PE OI'], ltp: r['PE LTP'] }));
  const resistances = df.filter(r => r['Strike Price'] >= spotPrice)
    .sort((a, b) => b['CE OI'] - a['CE OI'])
    .slice(0, numLevels)
    .map(r => ({ strike: r['Strike Price'], oi: r['CE OI'], ltp: r['CE LTP'] }));
  const nearestSupport = supports.length > 0 ? Math.max(...supports.map(s => s.strike)) : null;
  const nearestResistance = resistances.length > 0 ? Math.min(...resistances.map(r => r.strike)) : null;
  return { supports, resistances, nearestSupport, nearestResistance };
}

export function computeTotals(df: OptionChainRow[], spotPrice: number): Totals {
  const totalCeOi = df.reduce((s, r) => s + r['CE OI'], 0);
  const totalPeOi = df.reduce((s, r) => s + r['PE OI'], 0);
  const totalOi = totalCeOi + totalPeOi;
  const totalCeVol = df.reduce((s, r) => s + r['CE Volume'], 0);
  const totalPeVol = df.reduce((s, r) => s + r['PE Volume'], 0);
  const totalVol = totalCeVol + totalPeVol;
  const totalCePremium = df.reduce((s, r) => s + r['CE LTP'] * r['CE OI'], 0);
  const totalPePremium = df.reduce((s, r) => s + r['PE LTP'] * r['PE OI'], 0);
  const totalPremium = totalCePremium + totalPePremium;
  const atmIdx = df.reduce((best, r, i) =>
    Math.abs(r['Strike Price'] - spotPrice) < Math.abs(df[best]['Strike Price'] - spotPrice) ? i : best, 0);
  const atmStrike = df[atmIdx]['Strike Price'];
  const atmCeLtp = df[atmIdx]['CE LTP'];
  const atmPeLtp = df[atmIdx]['PE LTP'];
  const straddlePrice = atmCeLtp + atmPeLtp;
  const breakevenUp = atmStrike + straddlePrice;
  const breakevenDown = atmStrike - straddlePrice;
  const oiSkew = totalOi > 0 ? ((totalPeOi - totalCeOi) / totalOi) * 100 : 0;
  const premiumSkew = totalPremium > 0 ? ((totalPePremium - totalCePremium) / totalPremium) * 100 : 0;
  const pcrOi = totalCeOi > 0 ? totalPeOi / totalCeOi : 0;
  const pcrVol = totalCeVol > 0 ? totalPeVol / totalCeVol : 0;
  const maxCeRow = [...df].sort((a, b) => b['CE OI'] - a['CE OI'])[0];
  const maxPeRow = [...df].sort((a, b) => b['PE OI'] - a['PE OI'])[0];
  const maxCeStrike = totalCeOi > 0 ? maxCeRow['Strike Price'] : null;
  const maxPeStrike = totalPeOi > 0 ? maxPeRow['Strike Price'] : null;
  return {
    total_ce_oi: totalCeOi, total_pe_oi: totalPeOi, total_oi: totalOi,
    total_ce_vol: totalCeVol, total_pe_vol: totalPeVol, total_vol: totalVol,
    total_ce_premium: totalCePremium, total_pe_premium: totalPePremium, total_premium: totalPremium,
    atm_strike: atmStrike, atm_ce_ltp: atmCeLtp, atm_pe_ltp: atmPeLtp,
    straddle_price: straddlePrice, breakeven_up: breakevenUp, breakeven_down: breakevenDown,
    oi_skew: oiSkew, premium_skew: premiumSkew, pcr_oi: pcrOi, pcr_vol: pcrVol,
    max_ce_strike: maxCeStrike, max_pe_strike: maxPeStrike,
  };
}

export function computePriceBias(
  spotPrice: number,
  totals: Totals,
  nearestSupport: number | null,
  nearestResistance: number | null,
  maxPain: number | null,
  dayHigh: number | null,
  dayLow: number | null,
  pcrOi: number,
): PriceBias {
  const factors: BiasFactor[] = [];

  if (nearestSupport !== null && nearestResistance !== null) {
    const distSup = spotPrice - nearestSupport;
    const distRes = nearestResistance - spotPrice;
    const pr = distSup + distRes > 0 ? distSup / (distSup + distRes) : 0.5;
    if (pr < 0.35) {
      factors.push({ name: 'Support/Resistance Proximity', vote: +1, reason: `Spot ₹${spotPrice.toLocaleString()} is only ${distSup.toFixed(0)} pts above support ${nearestSupport.toLocaleString()} → bounce zone`, bull_val: `${nearestSupport.toLocaleString()}`, bear_val: `${nearestResistance.toLocaleString()}` });
    } else if (pr > 0.65) {
      factors.push({ name: 'Support/Resistance Proximity', vote: -1, reason: `Spot ₹${spotPrice.toLocaleString()} is only ${distRes.toFixed(0)} pts below resistance ${nearestResistance.toLocaleString()} → rejection zone`, bull_val: `${nearestSupport.toLocaleString()}`, bear_val: `${nearestResistance.toLocaleString()}` });
    } else {
      factors.push({ name: 'Support/Resistance Proximity', vote: 0, reason: `Spot is mid-range between support (${nearestSupport.toLocaleString()}) and resistance (${nearestResistance.toLocaleString()})`, bull_val: `${nearestSupport.toLocaleString()}`, bear_val: `${nearestResistance.toLocaleString()}` });
    }
  } else {
    factors.push({ name: 'Support/Resistance Proximity', vote: 0, reason: 'Not enough data', bull_val: '—', bear_val: '—' });
  }

  if (maxPain !== null) {
    const mpDiff = spotPrice - maxPain;
    if (mpDiff > 0) {
      factors.push({ name: 'Max Pain Gravity', vote: -1, reason: `Spot (${spotPrice.toLocaleString()}) is ${mpDiff.toFixed(0)} pts ABOVE max pain (${maxPain.toLocaleString()}) → gravity pull DOWN`, bull_val: '—', bear_val: `${maxPain.toLocaleString()} (pull down)` });
    } else if (mpDiff < 0) {
      factors.push({ name: 'Max Pain Gravity', vote: +1, reason: `Spot (${spotPrice.toLocaleString()}) is ${Math.abs(mpDiff).toFixed(0)} pts BELOW max pain (${maxPain.toLocaleString()}) → gravity pull UP`, bull_val: `${maxPain.toLocaleString()} (pull up)`, bear_val: '—' });
    } else {
      factors.push({ name: 'Max Pain Gravity', vote: 0, reason: `Spot is AT max pain (${maxPain.toLocaleString()}) → pinning`, bull_val: '—', bear_val: '—' });
    }
  } else {
    factors.push({ name: 'Max Pain Gravity', vote: 0, reason: 'Max pain not calculated', bull_val: '—', bear_val: '—' });
  }

  const beUp = totals.breakeven_up;
  const beDown = totals.breakeven_down;
  if (spotPrice > beUp) {
    factors.push({ name: 'Straddle Breakeven Zone', vote: +1, reason: `Spot (${spotPrice.toLocaleString()}) has broken ABOVE upper breakeven (${beUp.toLocaleString()}) → trending up strongly`, bull_val: `>${beUp.toLocaleString()}`, bear_val: `<${beDown.toLocaleString()}` });
  } else if (spotPrice < beDown) {
    factors.push({ name: 'Straddle Breakeven Zone', vote: -1, reason: `Spot (${spotPrice.toLocaleString()}) has broken BELOW lower breakeven (${beDown.toLocaleString()}) → trending down strongly`, bull_val: `>${beUp.toLocaleString()}`, bear_val: `<${beDown.toLocaleString()}` });
  } else {
    const mid = (beUp + beDown) / 2;
    const vote = spotPrice > mid ? +1 : -1;
    const side = spotPrice > mid ? 'upper half' : 'lower half';
    factors.push({ name: 'Straddle Breakeven Zone', vote, reason: `Spot is inside straddle zone (${beDown.toLocaleString()}–${beUp.toLocaleString()}), in ${side} → mild directional lean`, bull_val: `>${beUp.toLocaleString()}`, bear_val: `<${beDown.toLocaleString()}` });
  }

  if (dayHigh !== null && dayLow !== null && (dayHigh - dayLow) > 0) {
    const dayRange = dayHigh - dayLow;
    const pos = (spotPrice - dayLow) / dayRange;
    if (pos >= 0.7) {
      factors.push({ name: 'Day Range Position', vote: -1, reason: `Spot is in TOP ${(pos * 100).toFixed(0)}% of day range → overextended`, bull_val: `<30% (${dayLow.toLocaleString()}–${(dayLow + dayRange * 0.3).toLocaleString()})`, bear_val: `>70% (${(dayLow + dayRange * 0.7).toLocaleString()}–${dayHigh.toLocaleString()})` });
    } else if (pos <= 0.3) {
      factors.push({ name: 'Day Range Position', vote: +1, reason: `Spot is in BOTTOM ${(pos * 100).toFixed(0)}% of day range → oversold intraday`, bull_val: `<30% (${dayLow.toLocaleString()}–${(dayLow + dayRange * 0.3).toLocaleString()})`, bear_val: `>70% (${(dayLow + dayRange * 0.7).toLocaleString()}–${dayHigh.toLocaleString()})` });
    } else {
      const vote = pos > 0.5 ? +1 : -1;
      factors.push({ name: 'Day Range Position', vote, reason: `Spot at ${(pos * 100).toFixed(0)}% of day range → mild ${pos > 0.5 ? 'upper' : 'lower'} bias`, bull_val: `<30%`, bear_val: `>70%` });
    }
  }

  if (totals.max_ce_strike !== null && totals.max_pe_strike !== null) {
    const distToRes = totals.max_ce_strike - spotPrice;
    const distToSup = spotPrice - totals.max_pe_strike;
    if (distToSup < distToRes * 0.5) {
      factors.push({ name: 'Max OI Wall Distance', vote: +1, reason: `Max PE OI wall (${totals.max_pe_strike.toLocaleString()}) very close below → strong floor`, bull_val: `PE wall ${totals.max_pe_strike.toLocaleString()}`, bear_val: `CE wall ${totals.max_ce_strike.toLocaleString()}` });
    } else if (distToRes < distToSup * 0.5) {
      factors.push({ name: 'Max OI Wall Distance', vote: -1, reason: `Max CE OI wall (${totals.max_ce_strike.toLocaleString()}) very close above → strong ceiling`, bull_val: `PE wall ${totals.max_pe_strike.toLocaleString()}`, bear_val: `CE wall ${totals.max_ce_strike.toLocaleString()}` });
    } else {
      factors.push({ name: 'Max OI Wall Distance', vote: 0, reason: `Max CE OI at ${totals.max_ce_strike.toLocaleString()}, Max PE OI at ${totals.max_pe_strike.toLocaleString()} — balanced`, bull_val: `PE wall ${totals.max_pe_strike.toLocaleString()}`, bear_val: `CE wall ${totals.max_ce_strike.toLocaleString()}` });
    }
  }

  if (pcrOi >= 1.2) {
    factors.push({ name: 'PCR Confirmation', vote: +1, reason: `PCR ${pcrOi.toFixed(2)} ≥ 1.2 → strong put writing, bullish`, bull_val: '≥1.2', bear_val: '≤0.8' });
  } else if (pcrOi <= 0.8) {
    factors.push({ name: 'PCR Confirmation', vote: -1, reason: `PCR ${pcrOi.toFixed(2)} ≤ 0.8 → strong call writing, bearish`, bull_val: '≥1.2', bear_val: '≤0.8' });
  } else {
    factors.push({ name: 'PCR Confirmation', vote: 0, reason: `PCR ${pcrOi.toFixed(2)} is neutral (0.8–1.2)`, bull_val: '≥1.2', bear_val: '≤0.8' });
  }

  const score = factors.reduce((s, f) => s + f.vote, 0);
  const maxScore = factors.length;
  const bullCount = factors.filter(f => f.vote === +1).length;
  const bearCount = factors.filter(f => f.vote === -1).length;
  const neutCount = factors.filter(f => f.vote === 0).length;
  const pct = (score / maxScore) * 100;

  let verdict: string, color: string, action: string, emoji: string;
  if (pct >= 50) { verdict = 'STRONG BUY CALLS'; color = '#00cc44'; action = 'BUY CALLS / SELL PUTS'; emoji = '🚀'; }
  else if (pct >= 20) { verdict = 'MILD BUY CALLS'; color = '#66dd88'; action = 'Consider CALL buying'; emoji = '📈'; }
  else if (pct <= -50) { verdict = 'STRONG BUY PUTS'; color = '#cc2200'; action = 'BUY PUTS / SELL CALLS'; emoji = '🔻'; }
  else if (pct <= -20) { verdict = 'MILD BUY PUTS'; color = '#dd6666'; action = 'Consider PUT buying'; emoji = '📉'; }
  else { verdict = 'RANGE / NEUTRAL'; color = '#aaaaaa'; action = 'Straddle / Iron Condor'; emoji = '⚖️'; }

  return { factors, score, max_score: maxScore, pct, verdict, color, action, emoji, bull_count: bullCount, bear_count: bearCount, neut_count: neutCount };
}

export function computeMaxPain(df: OptionChainRow[]): { maxPain: number | null; painData: { strike: number; pain: number }[] } {
  const strikes = Array.from(new Set(df.map(r => r['Strike Price']))).sort((a, b) => a - b);
  const painData = strikes.map(strike => {
    const callPain = df.filter(r => r['Strike Price'] > strike)
      .reduce((sum, r) => sum + r['CE OI'] * (r['Strike Price'] - strike), 0);
    const putPain = df.filter(r => r['Strike Price'] < strike)
      .reduce((sum, r) => sum + r['PE OI'] * (strike - r['Strike Price']), 0);
    return { strike, pain: callPain + putPain };
  });
  const minPain = painData.reduce((min, p) => p.pain < min.pain ? p : min, painData[0]);
  return { maxPain: minPain ? minPain.strike : null, painData };
}

export function enrichChain(df: OptionChainRow[], spotPrice: number): EnrichedRow[] {
  const atmIdx = df.reduce((best, r, i) =>
    Math.abs(r['Strike Price'] - spotPrice) < Math.abs(df[best]['Strike Price'] - spotPrice) ? i : best, 0);
  const atmVal = df[atmIdx]['Strike Price'];
  const atmThreshold = atmVal * 0.003;

  const ceOiValues = df.map(r => r['CE OI']).sort((a, b) => a - b);
  const peOiValues = df.map(r => r['PE OI']).sort((a, b) => a - b);
  const ceQ80 = ceOiValues[Math.floor(ceOiValues.length * 0.8)] || 0;
  const peQ80 = peOiValues[Math.floor(peOiValues.length * 0.8)] || 0;
  const ceMedVol = df.map(r => r['CE Volume']).sort((a, b) => a - b)[Math.floor(df.length / 2)] || 0;
  const peMedVol = df.map(r => r['PE Volume']).sort((a, b) => a - b)[Math.floor(df.length / 2)] || 0;

  return df.map(row => {
    const moneyness = Math.abs(row['Strike Price'] - atmVal) <= atmThreshold ? 'ATM'
      : row['Strike Price'] < spotPrice ? 'ITM-CE / OTM-PE' : 'OTM-CE / ITM-PE';
    const strikePcr = row['CE OI'] > 0 ? Math.round((row['PE OI'] / row['CE OI']) * 100) / 100 : NaN;
    const straddle = Math.round((row['CE LTP'] + row['PE LTP']) * 100) / 100;
    const ceNotional = Math.round(row['CE LTP'] * row['CE OI']);
    const peNotional = Math.round(row['PE LTP'] * row['PE OI']);

    let strikeBias: string;
    if (isNaN(strikePcr)) strikeBias = '—';
    else if (strikePcr >= 1.5) strikeBias = '🟢 Strong Support';
    else if (strikePcr >= 1.1) strikeBias = '🟡 Mild Support';
    else if (strikePcr <= 0.6) strikeBias = '🔴 Strong Resistance';
    else if (strikePcr <= 0.9) strikeBias = '🟠 Mild Resistance';
    else strikeBias = '⚪ Neutral';

    let ivSignal: string;
    if (row['CE IV'] === 0 || row['PE IV'] === 0) ivSignal = '—';
    else {
      const ratio = row['PE IV'] / row['CE IV'];
      if (ratio > 1.3) ivSignal = '⬇️ Fear (High PE IV)';
      else if (ratio < 0.7) ivSignal = '⬆️ Greed (High CE IV)';
      else ivSignal = '➡️ Balanced';
    }

    return {
      ...row,
      Moneyness: moneyness,
      'CE Concentration': row['CE OI'] >= ceQ80 ? '🔴 HIGH' : '',
      'PE Concentration': row['PE OI'] >= peQ80 ? '🟢 HIGH' : '',
      'CE Vol Surge': row['CE Volume'] > 2 * ceMedVol && ceMedVol > 0 ? '⚡ SURGE' : '',
      'PE Vol Surge': row['PE Volume'] > 2 * peMedVol && peMedVol > 0 ? '⚡ SURGE' : '',
      'Strike PCR': strikePcr,
      Straddle: straddle,
      'CE Notional': ceNotional,
      'PE Notional': peNotional,
      'Strike Bias': strikeBias,
      'IV Signal': ivSignal,
      'Dist from Spot': Math.round((row['Strike Price'] - spotPrice) * 10) / 10,
      'Dist %': Math.round(((row['Strike Price'] - spotPrice) / spotPrice) * 10000) / 100,
    };
  });
}

export interface TradingSignals {
  call_buy: TradingSignal[];
  put_buy: TradingSignal[];
  call_sell: TradingSignal[];
  put_sell: TradingSignal[];
  market_bias: string;
  strategy: string;
}

export interface TradingSignal {
  strike: number;
  type: string;
  target: number | string;
  stop_loss: number;
  reason: string;
}

export function generateTradingSignals(
  df: OptionChainRow[],
  spotPrice: number,
  pcrData: PcrData,
  nearestSupport: number | null,
  nearestResistance: number | null,
): TradingSignals {
  const pcr = pcrData.pcr_oi;
  const signals: TradingSignals = { call_buy: [], put_buy: [], call_sell: [], put_sell: [], market_bias: '', strategy: '' };

  if (pcr > 1.3) { signals.market_bias = 'Strongly Bullish'; signals.strategy = 'Buy Calls or Sell Puts'; }
  else if (pcr > 1.0) { signals.market_bias = 'Moderately Bullish'; signals.strategy = 'Buy ATM/OTM Calls'; }
  else if (pcr < 0.7) { signals.market_bias = 'Strongly Bearish'; signals.strategy = 'Buy Puts or Sell Calls'; }
  else if (pcr < 0.9) { signals.market_bias = 'Moderately Bearish'; signals.strategy = 'Buy ATM/OTM Puts'; }
  else { signals.market_bias = 'Neutral/Rangebound'; signals.strategy = 'Iron Condor or Straddle'; }

  const atmIdx = df.reduce((best, r, i) =>
    Math.abs(r['Strike Price'] - spotPrice) < Math.abs(df[best]['Strike Price'] - spotPrice) ? i : best, 0);
  const atmStrike = df[atmIdx]['Strike Price'];

  if (pcr >= 1.0) {
    signals.call_buy.push({
      strike: atmStrike, type: 'ATM Call',
      target: nearestResistance ?? spotPrice + 500,
      stop_loss: nearestSupport ?? spotPrice - 200,
      reason: 'ATM call for bullish move',
    });
    const otmCandidates = df.filter(r => r['Strike Price'] > spotPrice).sort((a, b) => a['Strike Price'] - b['Strike Price']);
    if (otmCandidates.length > 0) {
      const otm = otmCandidates[0];
      signals.call_buy.push({
        strike: otm['Strike Price'], type: 'OTM Call',
        target: nearestResistance ?? spotPrice + 700,
        stop_loss: spotPrice - 100, reason: 'OTM call – aggressive bullish',
      });
    }
  }

  if (pcr <= 0.9) {
    signals.put_buy.push({
      strike: atmStrike, type: 'ATM Put',
      target: nearestSupport ?? spotPrice - 500,
      stop_loss: nearestResistance ?? spotPrice + 200,
      reason: 'ATM put for bearish move',
    });
    const otmCandidates = df.filter(r => r['Strike Price'] < spotPrice).sort((a, b) => b['Strike Price'] - a['Strike Price']);
    if (otmCandidates.length > 0) {
      const otm = otmCandidates[0];
      signals.put_buy.push({
        strike: otm['Strike Price'], type: 'OTM Put',
        target: nearestSupport ?? spotPrice - 700,
        stop_loss: spotPrice + 100, reason: 'OTM put – aggressive bearish',
      });
    }
  }

  if (pcr >= 1.2) {
    const strongSup = df.filter(r => r['Strike Price'] < spotPrice).sort((a, b) => b['PE OI'] - a['PE OI']);
    if (strongSup.length > 0) {
      const ss = strongSup[0]['Strike Price'];
      signals.put_sell.push({
        strike: ss, type: 'OTM Put Sell',
        target: 'Premium collection', stop_loss: ss - 200,
        reason: `Strong support at ${ss.toLocaleString()} – high PE OI`,
      });
    }
  }

  if (pcr <= 0.8) {
    const strongRes = df.filter(r => r['Strike Price'] > spotPrice).sort((a, b) => b['CE OI'] - a['CE OI']);
    if (strongRes.length > 0) {
      const sr = strongRes[0]['Strike Price'];
      signals.call_sell.push({
        strike: sr, type: 'OTM Call Sell',
        target: 'Premium collection', stop_loss: sr + 200,
        reason: `Strong resistance at ${sr.toLocaleString()} – high CE OI`,
      });
    }
  }

  return signals;
}

export function getNearbyStrikes(df: OptionChainRow[], spotPrice: number, rangePoints = 500) {
  return df
    .filter(r => r['Strike Price'] >= spotPrice - rangePoints && r['Strike Price'] <= spotPrice + rangePoints)
    .map(r => ({
      ...r,
      'OI Diff': r['PE OI'] - r['CE OI'],
      'OI Ratio': r['CE OI'] > 0 ? Math.round((r['PE OI'] / r['CE OI']) * 100) / 100 : 0,
    }))
    .sort((a, b) => a['Strike Price'] - b['Strike Price']);
}
