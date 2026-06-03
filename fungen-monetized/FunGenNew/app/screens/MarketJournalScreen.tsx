import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
  TouchableOpacity,
  RefreshControl,
  Animated,
  Dimensions,
  TextInput,
  Modal,
  FlatList,
} from 'react-native';
import LinearGradient from 'react-native-linear-gradient';
import { BarChart, LineChart, PieChart } from 'react-native-chart-kit';
import {
  fetchBseOptionChain,
  fetchExpiryDates,
  BseResponse,
  OptionChainRow,
} from '../services/bseApi';
import {
  calculatePcrAnalysis,
  findSupportResistance,
  computeTotals,
  computePriceBias,
  computeMaxPain,
  enrichChain,
  generateTradingSignals,
  getNearbyStrikes,
  fmt,
  fmtCr,
  PcrData,
  Totals,
  PriceBias,
  SupportResistance,
  EnrichedRow,
  TradingSignals,
} from '../services/optionAnalyzer';

const { width } = Dimensions.get('window');
const CHART_WIDTH = width - 64;

const sentimentColors: Record<string, string> = {
  green: '#10B981',
  red: '#EF4444',
  orange: '#F59E0B',
};

export default function MarketJournalScreen() {
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<BseResponse | null>(null);
  const [expiryDates, setExpiryDates] = useState<string[]>([]);
  const [selectedExpiry, setSelectedExpiry] = useState('');
  const [expiryModal, setExpiryModal] = useState(false);
  const [manualExpiry, setManualExpiry] = useState('');
  const [useManualExpiry, setUseManualExpiry] = useState(false);
  const [scripCd, setScripCd] = useState('1');
  const [showAdvanced, setShowAdvanced] = useState(true);
  const [selectedTab, setSelectedTab] = useState<'analysis' | 'options'>('analysis');
  const [signalTab, setSignalTab] = useState<'CB' | 'PB' | 'PS' | 'CS'>('CB');
  const [advTab, setAdvTab] = useState<'pain' | 'iv' | 'chain' | 'nearby' | 'signal'>('signal');

  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(30)).current;

  useEffect(() => {
    loadExpiryDates();
  }, []);

  useEffect(() => {
    if (expiryDates.length > 0 && !selectedExpiry) {
      setSelectedExpiry(expiryDates[0]);
    }
  }, [expiryDates]);

  useEffect(() => {
    if (selectedExpiry) {
      fetchData();
    }
  }, [selectedExpiry]);

  useEffect(() => {
    if (data) {
      Animated.parallel([
        Animated.timing(fadeAnim, { toValue: 1, duration: 600, useNativeDriver: true }),
        Animated.spring(slideAnim, { toValue: 0, tension: 25, friction: 8, useNativeDriver: true }),
      ]).start();
    }
  }, [data]);

  const loadExpiryDates = async () => {
    try {
      const dates = await fetchExpiryDates(scripCd);
      setExpiryDates(dates);
    } catch {
      setExpiryDates([]);
    }
  };

  const fetchData = async (silent = false) => {
    try {
      if (!silent) setLoading(true);
      setError(null);
      const expiry = useManualExpiry ? manualExpiry : selectedExpiry;
      if (!expiry) return;
      const result = await fetchBseOptionChain(expiry, scripCd);
      setData(result);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch data');
    } finally {
      if (!silent) setLoading(false);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchData();
    setRefreshing(false);
  };

  const analysis = useMemo(() => {
    if (!data || data.rows.length === 0) return null;
    const df = data.rows;
    const spot = data.spot_price;
    const pcrData = calculatePcrAnalysis(df);
    const { supports, resistances, nearestSupport, nearestResistance } = findSupportResistance(df, spot);
    const totals = computeTotals(df, spot);
    const { maxPain } = computeMaxPain(df);
    const priceBias = computePriceBias(spot, totals, nearestSupport, nearestResistance, maxPain, data.day_high, data.day_low, totals.pcr_oi);
    const enriched = enrichChain(df, spot);
    const signals = generateTradingSignals(df, spot, pcrData, nearestSupport, nearestResistance);
    const nearby = getNearbyStrikes(df, spot);
    return { pcrData, supports, resistances, nearestSupport, nearestResistance, totals, maxPain, priceBias, enriched, signals, nearby, spot };
  }, [data]);

  if (loading && !data) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#DC2626" />
        <Text style={styles.loadingText}>Loading BSE Option Chain...</Text>
      </View>
    );
  }

  if (error && !data) {
    return (
      <ScrollView style={styles.container} contentContainerStyle={styles.centerContainer}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}>
        <Text style={styles.errorText}>⚠️ {error}</Text>
        <Text style={styles.errorHint}>Pull down to retry</Text>
        <TouchableOpacity style={styles.retryBtn} onPress={onRefresh}>
          <Text style={styles.retryBtnText}>🔄 Retry</Text>
        </TouchableOpacity>
      </ScrollView>
    );
  }

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}>
      <Animated.View style={{ opacity: fadeAnim, transform: [{ translateY: slideAnim }] }}>

        {/* ⚙️ Config Header */}
        <View style={styles.configBar}>
          <TouchableOpacity style={styles.configItem} onPress={() => setExpiryModal(true)}>
            <Text style={styles.configLabel}>Expiry</Text>
            <Text style={styles.configValue}>{useManualExpiry ? manualExpiry : selectedExpiry || 'Select'}</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.configItem}>
            <Text style={styles.configLabel}>Scrip</Text>
            <TextInput style={styles.configInput} value={scripCd} onChangeText={setScripCd} keyboardType="numeric" />
          </TouchableOpacity>
          <TouchableOpacity style={styles.configBtn} onPress={() => fetchData()}>
            <Text style={styles.configBtnText}>🔄</Text>
          </TouchableOpacity>
        </View>

        {/* ⚠️ Disclaimer */}
        <View style={styles.disclaimer}>
          <Text style={styles.disclaimerText}>⚠️ Educational Only. Not financial advice.</Text>
        </View>

        {/* 📊 Day Range Visualizer */}
        {data?.day_high && data?.day_low && (
          <DayRangeVisualizer spotPrice={data.spot_price} dayHigh={data.day_high} dayLow={data.day_low} />
        )}

        {/* 💰 Key Metrics Row */}
        {analysis && <KeyMetricsRow data={data!} analysis={analysis} />}

        {/* 📈 Sentiment Bar */}
        {analysis && <SentimentBar pcrData={analysis.pcrData} />}

        {/* 🧭 Price-Based Direction Panel */}
        {analysis && <DirectionPanel priceBias={analysis.priceBias} totals={analysis.totals} spotPrice={analysis.spot}
          nearestSupport={analysis.nearestSupport} nearestResistance={analysis.nearestResistance} />}

        {/* 📊 Total OI / Volume / Premium Summary */}
        {analysis && <TotalsPanel totals={analysis.totals} spotPrice={analysis.spot} />}

        {/* 🍩 Donut Charts */}
        {analysis && <DonutCharts totals={analysis.totals} />}

        {/* 📊 OI & Volume Bar Chart */}
        {analysis && <OiVolumeChart rows={data!.rows} spotPrice={analysis.spot} />}

        {/* 🎯 Trading Signals */}
        {analysis && <TradingSignalsPanel signals={analysis.signals} signalTab={signalTab} setSignalTab={setSignalTab} />}

        {/* 🔴🟢 Support & Resistance */}
        {analysis && <SupportResistancePanel supports={analysis.supports} resistances={analysis.resistances}
          nearestSupport={analysis.nearestSupport} nearestResistance={analysis.nearestResistance} />}

        {/* 🎯 Advanced Analytics Tabs */}
        {showAdvanced && analysis && (
          <AdvancedPanel advTab={advTab} setAdvTab={setAdvTab}
            maxPain={analysis.maxPain} rows={data!.rows} spotPrice={analysis.spot}
            enriched={analysis.enriched} nearby={getNearbyStrikes(data!.rows, analysis.spot)}
            signals={analysis.signals} priceBias={analysis.priceBias} />
        )}

        <View style={{ height: 60 }} />
      </Animated.View>

      {/* Expiry Modal */}
      <Modal visible={expiryModal} transparent animationType="slide">
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Select Expiry</Text>
            <TouchableOpacity onPress={() => { setUseManualExpiry(!useManualExpiry); }}>
              <Text style={styles.modalToggle}>{useManualExpiry ? '📅 Use Dropdown' : '✏️ Manual Entry'}</Text>
            </TouchableOpacity>
            {useManualExpiry ? (
              <TextInput style={styles.modalInput} value={manualExpiry} onChangeText={setManualExpiry}
                placeholder="DD MMM YYYY" placeholderTextColor="#666" />
            ) : (
              <FlatList data={expiryDates} keyExtractor={i => i}
                renderItem={({ item }) => (
                  <TouchableOpacity style={[styles.expiryItem, item === selectedExpiry && styles.expiryItemActive]}
                    onPress={() => { setSelectedExpiry(item); setExpiryModal(false); }}>
                    <Text style={[styles.expiryItemText, item === selectedExpiry && styles.expiryItemTextActive]}>{item}</Text>
                  </TouchableOpacity>
                )} style={{ maxHeight: 300 }} />
            )}
            <TouchableOpacity style={styles.modalClose} onPress={() => setExpiryModal(false)}>
              <Text style={styles.modalCloseText}>Done</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </ScrollView>
  );
}

/* ─── Components ─── */

function DayRangeVisualizer({ spotPrice, dayHigh, dayLow }: { spotPrice: number; dayHigh: number; dayLow: number }) {
  const dayRange = dayHigh - dayLow;
  const rangePct = dayRange / dayLow * 100;
  const posInRange = dayRange > 0 ? ((spotPrice - dayLow) / dayRange * 100) : 50;
  return (
    <LinearGradient colors={['#667eea', '#764ba2']} style={styles.dayRangeCard}>
      <View style={styles.dayRangeRow}>
        <View style={styles.dayRangeItem}>
          <Text style={styles.dayRangeLabel}>Day Low</Text>
          <Text style={styles.dayRangeValue}>{fmt(dayLow)}</Text>
        </View>
        <View style={styles.dayRangeItem}>
          <Text style={styles.dayRangeLabel}>Spot</Text>
          <Text style={[styles.dayRangeValue, { color: '#FFD700', fontSize: 22 }]}>{fmt(spotPrice)}</Text>
          <Text style={styles.dayRangeSub}>{posInRange.toFixed(1)}% in range</Text>
        </View>
        <View style={styles.dayRangeItem}>
          <Text style={styles.dayRangeLabel}>Day High</Text>
          <Text style={styles.dayRangeValue}>{fmt(dayHigh)}</Text>
        </View>
      </View>
      <View style={styles.dayRangeBarBg}>
        <View style={[styles.dayRangeBarFill, { width: `${Math.min(posInRange, 100)}%` }]} />
      </View>
      <View style={styles.dayRangeFooter}>
        <Text style={styles.dayRangeSub}>Range: {fmt(dayRange)}</Text>
        <Text style={styles.dayRangeSub}>Movement: {rangePct.toFixed(2)}%</Text>
      </View>
    </LinearGradient>
  );
}

function KeyMetricsRow({ data, analysis }: { data: BseResponse; analysis: any }) {
  return (
    <View style={styles.metricsRow}>
      <MetricItem label="Spot" value={fmt(data.spot_price)} />
      <MetricItem label="Day High" value={data.day_high ? fmt(data.day_high) : 'N/A'} />
      <MetricItem label="Day Low" value={data.day_low ? fmt(data.day_low) : 'N/A'} />
      <MetricItem label="PCR OI" value={analysis.pcrData.pcr_oi.toFixed(2)} />
      <MetricItem label="Support" value={analysis.nearestSupport ? fmt(analysis.nearestSupport) : 'N/A'}
        delta={analysis.nearestSupport ? `-${(data.spot_price - analysis.nearestSupport).toFixed(0)}` : ''} />
      <MetricItem label="Resistance" value={analysis.nearestResistance ? fmt(analysis.nearestResistance) : 'N/A'}
        delta={analysis.nearestResistance ? `+${(analysis.nearestResistance - data.spot_price).toFixed(0)}` : ''} />
    </View>
  );
}

function MetricItem({ label, value, delta }: { label: string; value: string; delta?: string }) {
  return (
    <View style={styles.metricItem}>
      <Text style={styles.metricLabel}>{label}</Text>
      <Text style={styles.metricValue}>{value}</Text>
      {delta ? <Text style={styles.metricDelta}>{delta}</Text> : null}
    </View>
  );
}

function SentimentBar({ pcrData }: { pcrData: PcrData }) {
  const color = sentimentColors[pcrData.color] || '#888';
  return (
    <View style={[styles.sentimentBar, { borderLeftColor: color, backgroundColor: color + '15' }]}>
      <Text style={[styles.sentimentText, { color }]}>{pcrData.sentiment}</Text>
      <Text style={styles.sentimentDesc}>{pcrData.description}</Text>
      <View style={styles.sentimentBadgeRow}>
        <View style={[styles.sentimentBadge, { backgroundColor: '#DC2626' + '20' }]}>
          <Text style={[styles.sentimentBadgeText, { color: '#DC2626' }]}>CE: {pcrData.total_call_oi.toLocaleString()}</Text>
        </View>
        <View style={[styles.sentimentBadge, { backgroundColor: '#10B981' + '20' }]}>
          <Text style={[styles.sentimentBadgeText, { color: '#10B981' }]}>PE: {pcrData.total_put_oi.toLocaleString()}</Text>
        </View>
      </View>
    </View>
  );
}

function DirectionPanel({ priceBias, totals, spotPrice, nearestSupport, nearestResistance }: {
  priceBias: PriceBias; totals: Totals; spotPrice: number;
  nearestSupport: number | null; nearestResistance: number | null;
}) {
  const gaugeFill = Math.max(2, Math.min(98, 50 + priceBias.pct / 2));
  const isBull = priceBias.pct >= 0;
  return (
    <View style={[styles.directionCard, { borderColor: priceBias.color }]}>
      <Text style={styles.directionTitle}>🧭 Price-Based Directional Analysis</Text>
      <Text style={[styles.directionVerdict, { color: priceBias.color }]}>
        {priceBias.emoji} {priceBias.verdict}
      </Text>
      <Text style={[styles.directionAction, { color: '#FFD700' }]}>{priceBias.action}</Text>

      <View style={styles.gaugeContainer}>
        <View style={styles.gaugeLabels}>
          <Text style={styles.gaugeLabel}>🔴 BEAR</Text>
          <Text style={styles.gaugeLabel}>⚪ NEUTRAL</Text>
          <Text style={styles.gaugeLabel}>🟢 BULL</Text>
        </View>
        <View style={styles.gaugeBar}>
          <View style={styles.gaugeCenterLine} />
          <View style={[styles.gaugeFill, {
            width: `${Math.abs(gaugeFill - 50)}%`,
            backgroundColor: priceBias.color,
            [isBull ? 'left' : 'right']: '50%',
            ...(isBull ? {} : { position: 'absolute', right: '50%' }),
          }]} />
          <View style={styles.gaugeScoreBadge}>
            <Text style={styles.gaugeScoreText}>Score: {priceBias.score >= 0 ? '+' : ''}{priceBias.score}/{priceBias.max_score}</Text>
          </View>
        </View>
        <View style={styles.gaugeCountRow}>
          <Text style={styles.gaugeCount}>🟢 {priceBias.bull_count}</Text>
          <Text style={styles.gaugeCount}>🔴 {priceBias.bear_count}</Text>
          <Text style={styles.gaugeCount}>⚪ {priceBias.neut_count}</Text>
        </View>
      </View>

      {priceBias.factors.map((f, i) => (
        <View key={i} style={styles.factorRow}>
          <Text style={styles.factorVote}>{f.vote === 1 ? '🟢' : f.vote === -1 ? '🔴' : '⚪'}</Text>
          <View style={styles.factorContent}>
            <Text style={styles.factorName}>{f.name}</Text>
            <Text style={styles.factorReason}>{f.reason}</Text>
          </View>
        </View>
      ))}

      {/* Trade Setup Box */}
      <TradeSetupBox priceBias={priceBias} totals={totals} spotPrice={spotPrice}
        nearestSupport={nearestSupport} nearestResistance={nearestResistance} />
    </View>
  );
}

function TradeSetupBox({ priceBias, totals, spotPrice, nearestSupport, nearestResistance }: {
  priceBias: PriceBias; totals: Totals; spotPrice: number;
  nearestSupport: number | null; nearestResistance: number | null;
}) {
  const atm = totals.atm_strike;
  const beUp = totals.breakeven_up;
  const beDn = totals.breakeven_down;
  let recBuy: string, recSell: string, tgt: string, sl: string, boxColor: string, border: string;

  if (priceBias.pct >= 20) {
    recBuy = `BUY CALL @ ${atm.toLocaleString()} (ATM)`;
    recSell = nearestSupport ? `SELL PUT @ ${nearestSupport.toLocaleString()}` : 'SELL OTM PUT';
    tgt = nearestResistance ? fmt(nearestResistance) : 'next resistance';
    sl = nearestSupport ? fmt(nearestSupport - 100) : 'below support';
    boxColor = '#0f4f2f'; border = '#00cc44';
  } else if (priceBias.pct <= -20) {
    recBuy = `BUY PUT @ ${atm.toLocaleString()} (ATM)`;
    recSell = nearestResistance ? `SELL CALL @ ${nearestResistance.toLocaleString()}` : 'SELL OTM CALL';
    tgt = nearestSupport ? fmt(nearestSupport) : 'next support';
    sl = nearestResistance ? fmt(nearestResistance + 100) : 'above resistance';
    boxColor = '#4f0f0f'; border = '#cc2200';
  } else {
    recBuy = `BUY STRADDLE @ ${atm.toLocaleString()}`;
    recSell = nearestResistance && nearestSupport
      ? `SELL STRANGLE CE ${nearestResistance.toLocaleString()} / PE ${nearestSupport.toLocaleString()}`
      : 'IRON CONDOR';
    tgt = `Collect premium within ${fmt(beDn)} – ${fmt(beUp)}`;
    sl = 'Exit if spot breaks outside straddle zone';
    boxColor = '#2a2a2a'; border = '#888888';
  }

  return (
    <View style={[styles.tradeSetupBox, { backgroundColor: boxColor, borderLeftColor: border }]}>
      <Text style={[styles.tradeSetupTitle, { color: border }]}>🎯 Recommended Trade Setup</Text>
      <View style={styles.tradeSetupGrid}>
        <View style={styles.tradeSetupCol}>
          <Text style={styles.tradeSetupLabel}>PRIMARY</Text>
          <Text style={[styles.tradeSetupVal, { color: '#FFD700' }]}>{recBuy}</Text>
        </View>
        <View style={styles.tradeSetupCol}>
          <Text style={styles.tradeSetupLabel}>HEDGE</Text>
          <Text style={[styles.tradeSetupVal, { color: '#ccc' }]}>{recSell}</Text>
        </View>
        <View style={styles.tradeSetupCol}>
          <Text style={styles.tradeSetupLabel}>TARGET</Text>
          <Text style={[styles.tradeSetupVal, { color: '#66ff99' }]}>{tgt}</Text>
        </View>
        <View style={styles.tradeSetupCol}>
          <Text style={styles.tradeSetupLabel}>STOP LOSS</Text>
          <Text style={[styles.tradeSetupVal, { color: '#ff6666' }]}>{sl}</Text>
        </View>
      </View>
    </View>
  );
}

function TotalsPanel({ totals, spotPrice }: { totals: Totals; spotPrice: number }) {
  return (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>📦 OI · Volume · Premium Summary</Text>
      <View style={styles.totalsRow}>
        <View style={styles.totalsItem}>
          <Text style={styles.totalsLabel}>Total OI</Text>
          <Text style={styles.totalsValue}>{totals.total_oi.toLocaleString()}</Text>
          <View style={styles.totalsSubRow}>
            <Text style={[styles.totalsSub, { color: '#ff6347' }]}>CE: {totals.total_ce_oi.toLocaleString()}</Text>
            <Text style={[styles.totalsSub, { color: '#3cb371' }]}>PE: {totals.total_pe_oi.toLocaleString()}</Text>
          </View>
        </View>
        <View style={styles.totalsItem}>
          <Text style={styles.totalsLabel}>Total Volume</Text>
          <Text style={styles.totalsValue}>{totals.total_vol.toLocaleString()}</Text>
          <View style={styles.totalsSubRow}>
            <Text style={[styles.totalsSub, { color: '#ff6347' }]}>CE: {totals.total_ce_vol.toLocaleString()}</Text>
            <Text style={[styles.totalsSub, { color: '#3cb371' }]}>PE: {totals.total_pe_vol.toLocaleString()}</Text>
          </View>
        </View>
        <View style={styles.totalsItem}>
          <Text style={styles.totalsLabel}>Total Premium</Text>
          <Text style={styles.totalsValue}>{fmtCr(totals.total_premium)}</Text>
          <View style={styles.totalsSubRow}>
            <Text style={[styles.totalsSub, { color: '#ff6347' }]}>CE: {fmtCr(totals.total_ce_premium)}</Text>
            <Text style={[styles.totalsSub, { color: '#3cb371' }]}>PE: {fmtCr(totals.total_pe_premium)}</Text>
          </View>
        </View>
      </View>
      <View style={styles.atmRow}>
        <AtmItem label="ATM Strike" value={fmt(totals.atm_strike)} sub={`CE: ${fmt(totals.atm_ce_ltp)} | PE: ${fmt(totals.atm_pe_ltp)}`} />
        <AtmItem label="Straddle" value={fmt(totals.straddle_price)} sub={`${(totals.straddle_price / totals.atm_strike * 100).toFixed(2)}% of spot`} />
        <AtmItem label="BE Up" value={fmt(totals.breakeven_up)} sub={`+${fmt(totals.breakeven_up - spotPrice)}`} />
        <AtmItem label="BE Down" value={fmt(totals.breakeven_down)} sub={`-${fmt(spotPrice - totals.breakeven_down)}`} />
      </View>
      <View style={styles.maxOiRow}>
        <View style={styles.maxOiItem}>
          <Text style={[styles.maxOiLabel, { color: '#ff6347' }]}>🔴 Max CE OI: {fmt(totals.max_ce_strike ?? 0)}</Text>
        </View>
        <View style={styles.maxOiItem}>
          <Text style={[styles.maxOiLabel, { color: '#3cb371' }]}>🟢 Max PE OI: {fmt(totals.max_pe_strike ?? 0)}</Text>
        </View>
      </View>
    </View>
  );
}

function AtmItem({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <View style={styles.atmItem}>
      <Text style={styles.totalsLabel}>{label}</Text>
      <Text style={[styles.totalsValue, { fontSize: 14 }]}>{value}</Text>
      <Text style={styles.atmSub}>{sub}</Text>
    </View>
  );
}

function DonutCharts({ totals }: { totals: Totals }) {
  const oiData = [
    { name: 'CE OI', population: totals.total_ce_oi, color: '#ff6347', legendFontColor: '#fff', legendFontSize: 11 },
    { name: 'PE OI', population: totals.total_pe_oi, color: '#3cb371', legendFontColor: '#fff', legendFontSize: 11 },
  ];
  const premiumData = [
    { name: 'CE Premium', population: totals.total_ce_premium, color: '#ffa07a', legendFontColor: '#fff', legendFontSize: 11 },
    { name: 'PE Premium', population: totals.total_pe_premium, color: '#90ee90', legendFontColor: '#fff', legendFontSize: 11 },
  ];
  const chartConfig = { color: () => '#fff', backgroundGradientFrom: '#1a1a2e', backgroundGradientTo: '#16213e', decimalCount: 0 };

  return (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>🍩 OI & Premium Distribution</Text>
      <View style={styles.donutRow}>
        <View style={styles.donutItem}>
          <PieChart data={oiData} width={CHART_WIDTH / 2} height={160} chartConfig={chartConfig} accessor="population" backgroundColor="transparent" paddingLeft="0" absolute />
        </View>
        <View style={styles.donutItem}>
          <PieChart data={premiumData} width={CHART_WIDTH / 2} height={160} chartConfig={chartConfig} accessor="population" backgroundColor="transparent" paddingLeft="0" absolute />
        </View>
      </View>
    </View>
  );
}

function OiVolumeChart({ rows, spotPrice }: { rows: OptionChainRow[]; spotPrice: number }) {
  const nearby = rows.filter(r => Math.abs(r['Strike Price'] - spotPrice) <= 500);
  const labels = nearby.map(r => (r['Strike Price'] / 1000).toFixed(1) + 'k');
  const ceOi = nearby.map(r => r['CE OI']);
  const peOi = nearby.map(r => r['PE OI']);

  const chartConfig = {
    backgroundColor: '#1a1a2e', backgroundGradientFrom: '#1a1a2e', backgroundGradientTo: '#16213e',
    decimalCount: 0, color: (opacity = 1) => `rgba(255,255,255,${opacity})`,
    labelColor: () => '#888', barPercentage: 0.6,
  };

  return (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>📊 OI Distribution (Nearby Strikes)</Text>
      <BarChart
        data={{ labels: labels.slice(0, 10), datasets: [{ data: ceOi.slice(0, 10) }] }}
        width={CHART_WIDTH} height={200} chartConfig={chartConfig} yAxisLabel="" yAxisSuffix="" style={{ borderRadius: 12 }} />
    </View>
  );
}

function TradingSignalsPanel({ signals, signalTab, setSignalTab }: {
  signals: TradingSignals; signalTab: string; setSignalTab: (t: any) => void;
}) {
  const tabs = [
    { key: 'CB' as const, label: '📈 Call Buy' },
    { key: 'PB' as const, label: '📉 Put Buy' },
    { key: 'PS' as const, label: '💰 Put Sell' },
    { key: 'CS' as const, label: '💰 Call Sell' },
  ];
  const signalMap: Record<string, { items: any[]; color: string }> = {
    CB: { items: signals.call_buy, color: '#059669' },
    PB: { items: signals.put_buy, color: '#DC2626' },
    PS: { items: signals.put_sell, color: '#065F46' },
    CS: { items: signals.call_sell, color: '#7F1D1D' },
  };
  const current = signalMap[signalTab as keyof typeof signalMap] || signalMap.CB;

  return (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>🎯 Trading Signals</Text>
      <View style={styles.signalTabs}>
        {tabs.map(t => (
          <TouchableOpacity key={t.key} style={[styles.signalTab, signalTab === t.key && styles.signalTabActive]}
            onPress={() => setSignalTab(t.key)}>
            <Text style={[styles.signalTabText, signalTab === t.key && styles.signalTabTextActive]}>{t.label}</Text>
          </TouchableOpacity>
        ))}
      </View>
      {current.items.length === 0 ? (
        <Text style={styles.signalEmpty}>No setups at current market conditions.</Text>
      ) : current.items.map((s: any, i: number) => (
        <View key={i} style={[styles.signalCard, { borderLeftColor: current.color }]}>
          <Text style={[styles.signalType, { color: current.color }]}>{s.type} – Strike: {fmt(s.strike)}</Text>
          <Text style={styles.signalDetail}>🎯 Target: {typeof s.target === 'number' ? fmt(s.target) : s.target}</Text>
          <Text style={styles.signalDetail}>🛑 Stop Loss: {fmt(s.stop_loss)}</Text>
          <Text style={styles.signalReason}>{s.reason}</Text>
        </View>
      ))}
    </View>
  );
}

function SupportResistancePanel({ supports, resistances, nearestSupport, nearestResistance }: {
  supports: SupportResistance[]; resistances: SupportResistance[];
  nearestSupport: number | null; nearestResistance: number | null;
}) {
  return (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>🔴🟢 Support & Resistance</Text>
      <View style={styles.srRow}>
        <View style={styles.srCol}>
          <Text style={[styles.srTitle, { color: '#10B981' }]}>🟢 Support Levels</Text>
          {supports.map((s, i) => (
            <View key={i} style={styles.srItem}>
              <Text style={styles.srStrike}>{fmt(s.strike)}</Text>
              <Text style={styles.srDetail}>PE OI: {s.oi.toLocaleString()}</Text>
              <Text style={styles.srDetail}>PE LTP: {fmt(s.ltp)}</Text>
            </View>
          ))}
          {!nearestSupport && <Text style={styles.srEmpty}>No support</Text>}
        </View>
        <View style={styles.srDivider} />
        <View style={styles.srCol}>
          <Text style={[styles.srTitle, { color: '#EF4444' }]}>🔴 Resistance Levels</Text>
          {resistances.map((r, i) => (
            <View key={i} style={styles.srItem}>
              <Text style={styles.srStrike}>{fmt(r.strike)}</Text>
              <Text style={styles.srDetail}>CE OI: {r.oi.toLocaleString()}</Text>
              <Text style={styles.srDetail}>CE LTP: {fmt(r.ltp)}</Text>
            </View>
          ))}
          {!nearestResistance && <Text style={styles.srEmpty}>No resistance</Text>}
        </View>
      </View>
    </View>
  );
}

function AdvancedPanel({ advTab, setAdvTab, maxPain, rows, spotPrice, enriched, nearby, signals, priceBias }: {
  advTab: string; setAdvTab: (t: any) => void; maxPain: number | null;
  rows: OptionChainRow[]; spotPrice: number; enriched: EnrichedRow[];
  nearby: any[]; signals: TradingSignals; priceBias: PriceBias;
}) {
  const tabs = [
    { key: 'signal' as const, label: '🎯 Signals' },
    { key: 'pain' as const, label: '🎯 Max Pain' },
    { key: 'iv' as const, label: '📈 IV Smile' },
    { key: 'chain' as const, label: '📋 Full Chain' },
    { key: 'nearby' as const, label: '📍 Nearby' },
  ];

  const avgCeIv = rows.reduce((s, r) => s + r['CE IV'], 0) / rows.length;
  const avgPeIv = rows.reduce((s, r) => s + r['PE IV'], 0) / rows.length;

  const ivChartConfig = {
    backgroundColor: '#1a1a2e', backgroundGradientFrom: '#1a1a2e', backgroundGradientTo: '#16213e',
    decimalCount: 1, color: (opacity = 1) => `rgba(255,255,255,${opacity})`, labelColor: () => '#888',
  };

  const nearbyFiltered = nearby.slice(0, 20);
  const ivLabels = nearbyFiltered.map(r => (r['Strike Price'] / 1000).toFixed(1));
  const ceIv = nearbyFiltered.map(r => r['CE IV']);
  const peIv = nearbyFiltered.map(r => r['PE IV']);

  return (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>🎯 Advanced Analytics</Text>
      <View style={styles.advTabs}>
        {tabs.map(t => (
          <TouchableOpacity key={t.key} style={[styles.advTab, advTab === t.key && styles.advTabActive]}
            onPress={() => setAdvTab(t.key)}>
            <Text style={[styles.advTabText, advTab === t.key && styles.advTabTextActive]}>{t.label}</Text>
          </TouchableOpacity>
        ))}
      </View>

      {advTab === 'pain' && (
        <View>
          {maxPain ? (
            <View style={styles.advSection}>
              <View style={styles.maxPainBox}>
                <Text style={styles.maxPainLabel}>🎯 Max Pain</Text>
                <Text style={styles.maxPainValue}>{fmt(maxPain)}</Text>
                <Text style={styles.maxPainDelta}>
                  Δ Spot vs Pain: {(spotPrice - maxPain) >= 0 ? '+' : ''}{(spotPrice - maxPain).toFixed(0)}
                </Text>
              </View>
            </View>
          ) : <Text style={styles.advEmpty}>Max pain not available</Text>}
        </View>
      )}

      {advTab === 'iv' && (
        <View>
          <Text style={styles.advSectionTitle}>IV Smile (Nearby Strikes)</Text>
          {ivLabels.length > 0 ? (
            <LineChart
              data={{
                labels: ivLabels.slice(0, 10),
                datasets: [
                  { data: ceIv.slice(0, 10), color: () => '#ff6347', strokeWidth: 2 },
                  { data: peIv.slice(0, 10), color: () => '#3cb371', strokeWidth: 2 },
                ],
                legend: ['CE IV', 'PE IV'],
              }}
              width={CHART_WIDTH} height={220} chartConfig={ivChartConfig}
              bezier style={{ borderRadius: 12 }} />
          ) : <Text style={styles.advEmpty}>Insufficient data</Text>}
          <View style={styles.ivAvgRow}>
            <View style={styles.ivAvgItem}><Text style={styles.ivAvgLabel}>Avg CE IV</Text><Text style={[styles.ivAvgValue, { color: '#ff6347' }]}>{avgCeIv.toFixed(2)}%</Text></View>
            <View style={styles.ivAvgItem}><Text style={styles.ivAvgLabel}>Avg PE IV</Text><Text style={[styles.ivAvgValue, { color: '#3cb371' }]}>{avgPeIv.toFixed(2)}%</Text></View>
          </View>
        </View>
      )}

      {advTab === 'chain' && (
        <View>
          <EnrichedChainTable enriched={enriched} spotPrice={spotPrice} />
        </View>
      )}

      {advTab === 'nearby' && (
        <View>
          <NearbyStrikesTable nearby={nearby} />
        </View>
      )}

      {advTab === 'signal' && (
        <View style={styles.advSection}>
          <SignalsSummary signals={signals} priceBias={priceBias} spotPrice={spotPrice} />
        </View>
      )}
    </View>
  );
}

function EnrichedChainTable({ enriched, spotPrice }: { enriched: EnrichedRow[]; spotPrice: number }) {
  const nearby = enriched.filter(r => Math.abs(r['Strike Price'] - spotPrice) <= 500).slice(0, 15);
  return (
    <ScrollView horizontal showsHorizontalScrollIndicator>
      <View>
        <View style={styles.chainHeader}>
          {['Strike', 'CE OI', 'CE LTP', 'CE Vol', 'PE OI', 'PE LTP', 'PE Vol', 'PCR', 'Bias', 'IV'].map(h => (
            <Text key={h} style={styles.chainHeaderCell}>{h}</Text>
          ))}
        </View>
        {nearby.map((r, i) => (
          <View key={i} style={[styles.chainRow, i % 2 === 0 && styles.chainRowAlt]}>
            <Text style={[styles.chainCell, styles.chainCellStrike]}>{r['Strike Price'].toLocaleString()}</Text>
            <Text style={[styles.chainCell, { color: '#ff6347' }]}>{r['CE OI'].toLocaleString()}</Text>
            <Text style={styles.chainCell}>{r['CE LTP'].toFixed(2)}</Text>
            <Text style={styles.chainCell}>{r['CE Volume'].toLocaleString()}</Text>
            <Text style={[styles.chainCell, { color: '#3cb371' }]}>{r['PE OI'].toLocaleString()}</Text>
            <Text style={styles.chainCell}>{r['PE LTP'].toFixed(2)}</Text>
            <Text style={styles.chainCell}>{r['PE Volume'].toLocaleString()}</Text>
            <Text style={styles.chainCell}>{isNaN(r['Strike PCR']) ? '—' : r['Strike PCR'].toFixed(2)}</Text>
            <Text style={styles.chainCell}>{r['Strike Bias'].slice(0, 6)}</Text>
            <Text style={styles.chainCell}>{r['CE IV'].toFixed(1)}/{r['PE IV'].toFixed(1)}</Text>
          </View>
        ))}
      </View>
    </ScrollView>
  );
}

function NearbyStrikesTable({ nearby }: { nearby: any[] }) {
  const data = nearby.slice(0, 20);
  return (
    <ScrollView horizontal showsHorizontalScrollIndicator>
      <View>
        <View style={styles.chainHeader}>
          {['Strike', 'CE OI', 'PE OI', 'OI Diff', 'OI Ratio'].map(h => (
            <Text key={h} style={styles.chainHeaderCell}>{h}</Text>
          ))}
        </View>
        {data.map((r, i) => (
          <View key={i} style={[styles.chainRow, i % 2 === 0 && styles.chainRowAlt]}>
            <Text style={[styles.chainCell, styles.chainCellStrike]}>{r['Strike Price'].toLocaleString()}</Text>
            <Text style={[styles.chainCell, { color: '#ff6347' }]}>{r['CE OI'].toLocaleString()}</Text>
            <Text style={[styles.chainCell, { color: '#3cb371' }]}>{r['PE OI'].toLocaleString()}</Text>
            <Text style={[styles.chainCell, { color: r['OI Diff'] > 0 ? '#3cb371' : '#ff6347' }]}>
              {r['OI Diff'] >= 0 ? '+' : ''}{r['OI Diff'].toLocaleString()}
            </Text>
            <Text style={styles.chainCell}>{r['OI Ratio'].toFixed(2)}</Text>
          </View>
        ))}
      </View>
    </ScrollView>
  );
}

function SignalsSummary({ signals, priceBias, spotPrice }: { signals: TradingSignals; priceBias: PriceBias; spotPrice: number }) {
  return (
    <View>
      <View style={styles.signalSummaryBox}>
        <Text style={styles.signalSummaryLabel}>Market Bias</Text>
        <Text style={[styles.signalSummaryValue, { color: priceBias.color }]}>{signals.market_bias}</Text>
      </View>
      <View style={styles.signalSummaryBox}>
        <Text style={styles.signalSummaryLabel}>Strategy</Text>
        <Text style={styles.signalSummaryValue}>{signals.strategy}</Text>
      </View>
      <View style={styles.signalSummaryCounts}>
        <Text style={styles.signalSummaryCount}>📈 Call Buys: {signals.call_buy.length}</Text>
        <Text style={styles.signalSummaryCount}>📉 Put Buys: {signals.put_buy.length}</Text>
        <Text style={styles.signalSummaryCount}>💰 Put Sells: {signals.put_sell.length}</Text>
        <Text style={styles.signalSummaryCount}>💰 Call Sells: {signals.call_sell.length}</Text>
      </View>
    </View>
  );
}

/* ─── Styles ─── */

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0f0f1a' },
  centerContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24 },
  loadingText: { marginTop: 12, fontSize: 15, color: '#94A3B8', fontWeight: '700' },
  errorText: { fontSize: 16, color: '#EF4444', fontWeight: '700', textAlign: 'center', marginBottom: 8 },
  errorHint: { fontSize: 13, color: '#64748B', fontWeight: '600', marginBottom: 12 },
  retryBtn: { backgroundColor: '#1e293b', paddingHorizontal: 24, paddingVertical: 12, borderRadius: 12 },
  retryBtnText: { fontSize: 14, fontWeight: '700', color: '#fff' },

  // Config
  configBar: { flexDirection: 'row', alignItems: 'center', padding: 12, marginHorizontal: 12, marginTop: 8, backgroundColor: '#1a1a2e', borderRadius: 12, gap: 8 },
  configItem: { flex: 1 },
  configLabel: { fontSize: 9, color: '#64748B', fontWeight: '700', letterSpacing: 0.5 },
  configValue: { fontSize: 12, color: '#fff', fontWeight: '700', marginTop: 2 },
  configBtn: { padding: 8, backgroundColor: '#2a2a4e', borderRadius: 8 },
  configBtnText: { fontSize: 16 },
  configInput: { fontSize: 12, color: '#fff', fontWeight: '700', borderBottomWidth: 1, borderBottomColor: '#333', paddingVertical: 2 },

  // Disclaimer
  disclaimer: { marginHorizontal: 12, marginVertical: 8, padding: 10, backgroundColor: '#FEF3C7', borderRadius: 8, borderWidth: 1, borderColor: '#FDE047' },
  disclaimerText: { fontSize: 11, fontWeight: '700', color: '#92400E', textAlign: 'center' },

  // Day Range
  dayRangeCard: { marginHorizontal: 12, marginVertical: 8, padding: 16, borderRadius: 16, overflow: 'hidden' },
  dayRangeRow: { flexDirection: 'row', justifyContent: 'space-around', alignItems: 'center' },
  dayRangeItem: { alignItems: 'center' },
  dayRangeLabel: { fontSize: 11, color: 'rgba(255,255,255,0.7)', fontWeight: '700' },
  dayRangeValue: { fontSize: 18, color: '#fff', fontWeight: '900' },
  dayRangeSub: { fontSize: 10, color: 'rgba(255,255,255,0.6)', marginTop: 2 },
  dayRangeBarBg: { marginTop: 12, backgroundColor: 'rgba(255,255,255,0.2)', borderRadius: 8, height: 20, overflow: 'hidden' },
  dayRangeBarFill: { height: '100%', backgroundColor: '#FFD700', borderRadius: 8 },
  dayRangeFooter: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 8 },

  // Metrics
  metricsRow: { flexDirection: 'row', flexWrap: 'wrap', marginHorizontal: 12, gap: 6 },
  metricItem: { width: (width - 48) / 3, backgroundColor: '#1a1a2e', padding: 10, borderRadius: 10, alignItems: 'center' },
  metricLabel: { fontSize: 9, color: '#64748B', fontWeight: '700', letterSpacing: 0.5 },
  metricValue: { fontSize: 14, color: '#fff', fontWeight: '900', marginTop: 2 },
  metricDelta: { fontSize: 10, color: '#94A3B8', fontWeight: '700' },

  // Sentiment
  sentimentBar: { marginHorizontal: 12, marginVertical: 8, padding: 14, borderRadius: 10, borderLeftWidth: 4 },
  sentimentText: { fontSize: 18, fontWeight: '900' },
  sentimentDesc: { fontSize: 12, color: '#ccc', marginTop: 4 },
  sentimentBadgeRow: { flexDirection: 'row', gap: 8, marginTop: 8 },
  sentimentBadge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 6 },
  sentimentBadgeText: { fontSize: 11, fontWeight: '800' },

  // Direction Panel
  directionCard: { marginHorizontal: 12, marginVertical: 8, backgroundColor: '#1a1a2e', padding: 16, borderRadius: 16, borderWidth: 2 },
  directionTitle: { fontSize: 12, color: '#aaa', fontWeight: '700', letterSpacing: 1, textAlign: 'center', marginBottom: 8 },
  directionVerdict: { fontSize: 28, fontWeight: '900', textAlign: 'center' },
  directionAction: { fontSize: 14, fontWeight: '700', textAlign: 'center', marginBottom: 12 },

  gaugeContainer: { marginVertical: 12 },
  gaugeLabels: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 4 },
  gaugeLabel: { fontSize: 9, color: '#888', fontWeight: '600' },
  gaugeBar: { backgroundColor: '#333', borderRadius: 14, height: 28, position: 'relative', overflow: 'hidden', justifyContent: 'center' },
  gaugeCenterLine: { position: 'absolute', left: '50%', top: 0, width: 2, height: '100%', backgroundColor: 'rgba(255,255,255,0.3)' },
  gaugeFill: { height: '100%', opacity: 0.85 },
  gaugeScoreBadge: { position: 'absolute', left: '50%', top: '50%', transform: [{ translateX: -60 }, { translateY: -9 }] },
  gaugeScoreText: { fontSize: 12, fontWeight: '700', color: '#fff', textShadowColor: '#000', textShadowRadius: 4 },
  gaugeCountRow: { flexDirection: 'row', justifyContent: 'center', gap: 16, marginTop: 6 },
  gaugeCount: { fontSize: 12, fontWeight: '700', color: '#aaa' },

  factorRow: { flexDirection: 'row', gap: 8, paddingVertical: 8, borderBottomWidth: 1, borderBottomColor: '#2a2a2a' },
  factorVote: { fontSize: 14, marginTop: 2 },
  factorContent: { flex: 1 },
  factorName: { fontSize: 12, fontWeight: '700', color: '#ddd' },
  factorReason: { fontSize: 10, color: '#888', marginTop: 2 },

  tradeSetupBox: { marginTop: 12, padding: 14, borderRadius: 10, borderLeftWidth: 4 },
  tradeSetupTitle: { fontSize: 14, fontWeight: '700', marginBottom: 10 },
  tradeSetupGrid: { gap: 8 },
  tradeSetupCol: {},
  tradeSetupLabel: { fontSize: 9, color: 'rgba(255,255,255,0.5)', fontWeight: '700', letterSpacing: 0.5 },
  tradeSetupVal: { fontSize: 12, fontWeight: '700', marginTop: 2 },

  // Card
  card: { marginHorizontal: 12, marginVertical: 8, backgroundColor: '#1a1a2e', padding: 16, borderRadius: 16 },
  cardTitle: { fontSize: 16, fontWeight: '900', color: '#fff', marginBottom: 12 },

  // Totals
  totalsRow: { gap: 8 },
  totalsItem: { backgroundColor: '#16213e', padding: 12, borderRadius: 10 },
  totalsLabel: { fontSize: 10, color: '#64748B', fontWeight: '700', letterSpacing: 0.5 },
  totalsValue: { fontSize: 18, color: '#fff', fontWeight: '900', marginTop: 2 },
  totalsSubRow: { flexDirection: 'row', justifyContent: 'space-around', marginTop: 6 },
  totalsSub: { fontSize: 11, fontWeight: '700' },

  atmRow: { flexDirection: 'row', gap: 6, marginTop: 8, flexWrap: 'wrap' },
  atmItem: { flex: 1, minWidth: 80, backgroundColor: '#16213e', padding: 10, borderRadius: 8, alignItems: 'center' },
  atmSub: { fontSize: 9, color: '#888', marginTop: 2 },
  maxOiRow: { flexDirection: 'row', gap: 8, marginTop: 8 },
  maxOiItem: { flex: 1, backgroundColor: '#16213e', padding: 10, borderRadius: 8 },
  maxOiLabel: { fontSize: 12, fontWeight: '700' },

  // Donuts
  donutRow: { flexDirection: 'row' },
  donutItem: { flex: 1, alignItems: 'center' },

  // Signal Tabs
  signalTabs: { flexDirection: 'row', backgroundColor: '#16213e', borderRadius: 10, padding: 3, marginBottom: 12 },
  signalTab: { flex: 1, paddingVertical: 8, alignItems: 'center', borderRadius: 8 },
  signalTabActive: { backgroundColor: '#2a2a4e' },
  signalTabText: { fontSize: 10, fontWeight: '700', color: '#64748B' },
  signalTabTextActive: { color: '#fff' },
  signalEmpty: { fontSize: 13, color: '#64748B', fontStyle: 'italic', textAlign: 'center', padding: 16 },
  signalCard: { padding: 12, backgroundColor: '#16213e', borderRadius: 10, marginBottom: 8, borderLeftWidth: 3 },
  signalType: { fontSize: 13, fontWeight: '800', marginBottom: 6 },
  signalDetail: { fontSize: 11, color: '#ccc', fontWeight: '600' },
  signalReason: { fontSize: 10, color: '#888', marginTop: 4, fontStyle: 'italic' },

  // SR
  srRow: { flexDirection: 'row' },
  srCol: { flex: 1 },
  srDivider: { width: 1, backgroundColor: '#333', marginHorizontal: 8 },
  srTitle: { fontSize: 12, fontWeight: '800', marginBottom: 8 },
  srItem: { paddingVertical: 6, borderBottomWidth: 1, borderBottomColor: '#2a2a2a' },
  srStrike: { fontSize: 16, fontWeight: '900', color: '#fff' },
  srDetail: { fontSize: 10, color: '#888', fontWeight: '600' },
  srEmpty: { fontSize: 11, color: '#64748B', fontStyle: 'italic' },

  // Advanced Tabs
  advTabs: { flexDirection: 'row', flexWrap: 'wrap', backgroundColor: '#16213e', borderRadius: 10, padding: 3, marginBottom: 12, gap: 2 },
  advTab: { paddingVertical: 6, paddingHorizontal: 10, borderRadius: 8 },
  advTabActive: { backgroundColor: '#2a2a4e' },
  advTabText: { fontSize: 10, fontWeight: '700', color: '#64748B' },
  advTabTextActive: { color: '#fff' },
  advSection: { gap: 8 },
  advSectionTitle: { fontSize: 12, color: '#ccc', fontWeight: '700', marginBottom: 8 },
  advEmpty: { fontSize: 12, color: '#64748B', fontStyle: 'italic', textAlign: 'center', padding: 12 },

  // Max Pain
  maxPainBox: { backgroundColor: '#16213e', padding: 16, borderRadius: 12, alignItems: 'center' },
  maxPainLabel: { fontSize: 11, color: '#64748B', fontWeight: '700' },
  maxPainValue: { fontSize: 28, color: '#FFD700', fontWeight: '900', marginTop: 4 },
  maxPainDelta: { fontSize: 13, color: '#ccc', fontWeight: '700', marginTop: 4 },

  // IV
  ivAvgRow: { flexDirection: 'row', gap: 8, marginTop: 8 },
  ivAvgItem: { flex: 1, backgroundColor: '#16213e', padding: 10, borderRadius: 8, alignItems: 'center' },
  ivAvgLabel: { fontSize: 10, color: '#64748B', fontWeight: '700' },
  ivAvgValue: { fontSize: 16, fontWeight: '900', marginTop: 2 },

  // Chain & Nearby Tables
  chainHeader: { flexDirection: 'row', backgroundColor: '#2a2a4e', paddingVertical: 8, paddingHorizontal: 4 },
  chainHeaderCell: { width: 60, fontSize: 9, fontWeight: '800', color: '#64748B', textAlign: 'center' },
  chainRow: { flexDirection: 'row', paddingVertical: 8, paddingHorizontal: 4, borderBottomWidth: 1, borderBottomColor: '#2a2a2a' },
  chainRowAlt: { backgroundColor: '#16162a' },
  chainCell: { width: 60, fontSize: 10, color: '#ccc', fontWeight: '600', textAlign: 'center', fontFamily: 'Menlo' },
  chainCellStrike: { fontWeight: '900', color: '#fff' },

  // Signal Summary
  signalSummaryBox: { backgroundColor: '#16213e', padding: 12, borderRadius: 8, marginBottom: 6 },
  signalSummaryLabel: { fontSize: 10, color: '#64748B', fontWeight: '700' },
  signalSummaryValue: { fontSize: 16, color: '#fff', fontWeight: '900', marginTop: 2 },
  signalSummaryCounts: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 8 },
  signalSummaryCount: { fontSize: 11, color: '#aaa', fontWeight: '700' },

  // Modal
  modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.7)', justifyContent: 'flex-end' },
  modalContent: { backgroundColor: '#1a1a2e', borderTopLeftRadius: 20, borderTopRightRadius: 20, padding: 20, maxHeight: '70%' },
  modalTitle: { fontSize: 18, fontWeight: '900', color: '#fff', marginBottom: 12 },
  modalToggle: { fontSize: 13, color: '#3B82F6', fontWeight: '700', marginBottom: 12 },
  modalInput: { backgroundColor: '#2a2a4e', borderRadius: 10, padding: 12, fontSize: 14, color: '#fff', marginBottom: 12 },
  modalClose: { backgroundColor: '#3B82F6', padding: 14, borderRadius: 12, alignItems: 'center', marginTop: 8 },
  modalCloseText: { fontSize: 16, fontWeight: '800', color: '#fff' },

  expiryItem: { paddingVertical: 12, paddingHorizontal: 16, borderBottomWidth: 1, borderBottomColor: '#2a2a2a' },
  expiryItemActive: { backgroundColor: '#2a2a4e', borderRadius: 8 },
  expiryItemText: { fontSize: 14, color: '#ccc', fontWeight: '600' },
  expiryItemTextActive: { color: '#fff', fontWeight: '800' },
});
