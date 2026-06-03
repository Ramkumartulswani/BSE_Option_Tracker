import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
  Dimensions,
  Animated,
  RefreshControl,
  TouchableOpacity,
} from 'react-native';
import LinearGradient from 'react-native-linear-gradient';

const { width } = Dimensions.get('window');

// Google Drive URLs
const MARKET_URLS = {
  NIFTY: 'https://drive.google.com/uc?export=download&id=1t9fYO6ry9igdt3DZqlBqakMArBA4CdUK',
  BANKNIFTY: 'https://drive.google.com/uc?export=download&id=1Yj0AtywQaR-RW0ofrOpw7p8Yi1S66WVa',
};

// Auto-refresh interval (in milliseconds)
const AUTO_REFRESH_INTERVAL = 5000; // 5 seconds for fast updates

/* ---------------- HELPERS ---------------- */

const getActionColors = (action: string) => {
  switch (action) {
    case 'SHORT':
      return {
        gradient: ['#DC2626', '#991B1B', '#7F1D1D'],
        accent: '#DC2626',
        bg: '#FEE2E2',
        text: '#7F1D1D',
      };
    case 'LONG':
      return {
        gradient: ['#059669', '#047857', '#065F46'],
        accent: '#059669',
        bg: '#D1FAE5',
        text: '#065F46',
      };
    default:
      return {
        gradient: ['#6B7280', '#4B5563', '#374151'],
        accent: '#6B7280',
        bg: '#F3F4F6',
        text: '#374151',
      };
  }
};

const getStrengthIndicator = (pct: number) => {
  const abs = Math.abs(pct);
  if (abs >= 60) return { bars: 5, color: '#DC2626', label: 'EXTREME' };
  if (abs >= 40) return { bars: 4, color: '#EF4444', label: 'STRONG' };
  if (abs >= 20) return { bars: 3, color: '#F59E0B', label: 'MODERATE' };
  if (abs >= 10) return { bars: 2, color: '#10B981', label: 'BUILDING' };
  return { bars: 1, color: '#94A3B8', label: 'WEAK' };
};

const formatTime = (date: Date) => {
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
};

const StrengthBars = ({ strength }: { strength: number }) => {
  const indicator = getStrengthIndicator(strength);
  return (
    <View style={styles.strengthBars}>
      {[1, 2, 3, 4, 5].map((bar) => (
        <View
          key={bar}
          style={[
            styles.strengthBar,
            {
              backgroundColor:
                bar <= indicator.bars ? indicator.color : '#E5E7EB',
              height: 4 + bar * 2,
            },
          ]}
        />
      ))}
    </View>
  );
};

const PulseIndicator = ({ color }: { color: string }) => {
  const pulseAnim = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.2,
          duration: 800,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 800,
          useNativeDriver: true,
        }),
      ])
    ).start();
  }, []);

  return (
    <Animated.View
      style={[
        styles.pulseOuter,
        {
          backgroundColor: color + '20',
          transform: [{ scale: pulseAnim }],
        },
      ]}
    >
      <View style={[styles.pulseInner, { backgroundColor: color }]} />
    </Animated.View>
  );
};

const OptionChainView = ({ data, colors }: { data: any; colors: any }) => {
  const [selectedTab, setSelectedTab] = useState<'CE' | 'PE'>('CE');
  const spotPrice = data.spot_price;

  const strikes = data.zones?.resistance?.concat(data.zones?.support || [])
    .map((z: any) => z.strike)
    .filter((v: any, i: any, a: any) => a.indexOf(v) === i) as number[]
    || [];

  const atmStrike = strikes.length > 0 
    ? strikes.reduce((prev: number, curr: number) => 
        Math.abs(curr - spotPrice) < Math.abs(prev - spotPrice) ? curr : prev)
    : spotPrice;

  const nearbyStrikes = strikes
    .sort((a: number, b: number) => a - b)
    .filter((s: number) => Math.abs(s - atmStrike) <= 500);

  return (
    <View style={styles.card}>
      <View style={styles.optionChainHeader}>
        <TouchableOpacity 
          style={[styles.optionTab, selectedTab === 'CE' && styles.optionTabActive]}
          onPress={() => setSelectedTab('CE')}
        >
          <Text style={[styles.optionTabText, selectedTab === 'CE' && styles.optionTabTextActive]}>
            CALLS (CE)
          </Text>
        </TouchableOpacity>
        <TouchableOpacity 
          style={[styles.optionTab, selectedTab === 'PE' && styles.optionTabActive]}
          onPress={() => setSelectedTab('PE')}
        >
          <Text style={[styles.optionTabText, selectedTab === 'PE' && styles.optionTabTextActive]}>
            PUTS (PE)
          </Text>
        </TouchableOpacity>
      </View>

      <View style={styles.optionChainGrid}>
        <View style={styles.optionChainCol}>
          <Text style={styles.optionChainColHeader}>STRIKE</Text>
        </View>
        <View style={styles.optionChainCol}>
          <Text style={styles.optionChainColHeader}>PREMIUM</Text>
        </View>
        <View style={styles.optionChainCol}>
          <Text style={styles.optionChainColHeader}>OI CHG</Text>
        </View>
        <View style={styles.optionChainCol}>
          <Text style={styles.optionChainColHeader}>VOLUME</Text>
        </View>
      </View>

      {nearbyStrikes.map((strike: number) => {
        const zone = [...(data.zones?.resistance || []), ...(data.zones?.support || [])]
          .find((z: any) => z.strike === strike);
        const isATM = Math.abs(strike - atmStrike) <= 50;
        
        let premium = 0;
        if (selectedTab === 'CE') {
          premium = isATM ? Math.max(1, (spotPrice - strike) * 0.15) : Math.max(0.5, (spotPrice - strike) * 0.1);
        } else {
          premium = isATM ? Math.max(1, (strike - spotPrice) * 0.15) : Math.max(0.5, (strike - spotPrice) * 0.1);
        }
        premium = Math.round(premium * 100) / 100;

        const oiChange = zone 
          ? (selectedTab === 'CE' ? zone.call_oi_change_pct : zone.put_oi_change_pct) 
          : 0;

        return (
          <View key={strike} style={[
            styles.optionChainRow,
            isATM && styles.optionChainRowATM
          ]}>
            <View style={styles.optionChainCol}>
              <Text style={[styles.optionChainStrike, isATM && styles.optionChainStrikeATM]}>
                {strike}
              </Text>
              {isATM && <Text style={styles.optionChainATMBadge}>ATM</Text>}
            </View>
            <View style={styles.optionChainCol}>
              <Text style={[
                styles.optionChainPremium,
                selectedTab === 'CE' ? styles.callPremium : styles.putPremium
              ]}>
                ₹{premium.toFixed(2)}
              </Text>
            </View>
            <View style={styles.optionChainCol}>
              <Text style={[
                styles.optionChainOiChange,
                oiChange > 0 ? styles.oiUp : styles.oiDown
              ]}>
                {oiChange > 0 ? '+' : ''}{oiChange.toFixed(1)}%
              </Text>
            </View>
            <View style={styles.optionChainCol}>
              <Text style={styles.optionChainVolume}>
                {zone ? (zone.call_oi + zone.put_oi).toLocaleString() : '-'}
              </Text>
            </View>
          </View>
        );
      })}

      <View style={styles.optionChainFooter}>
        <Text style={styles.optionChainFooterText}>
          💡 Spot: ₹{spotPrice} | ATM: {atmStrike}
        </Text>
      </View>
    </View>
  );
};

export default function MarketScreen() {
  const [marketData, setMarketData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [autoRefreshEnabled, setAutoRefreshEnabled] = useState(true);
  const [countdown, setCountdown] = useState(AUTO_REFRESH_INTERVAL / 1000);
  const [viewMode, setViewMode] = useState<'analysis' | 'options'>('analysis');

  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(30)).current;
  const refreshIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const countdownIntervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    fetchMarketData();

    return () => {
      if (refreshIntervalRef.current) clearInterval(refreshIntervalRef.current);
      if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
    };
  }, []);

  useEffect(() => {
    if (autoRefreshEnabled) {
      startAutoRefresh();
    } else {
      stopAutoRefresh();
    }

    return () => stopAutoRefresh();
  }, [autoRefreshEnabled]);

  useEffect(() => {
    if (marketData) {
      Animated.parallel([
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 600,
          useNativeDriver: true,
        }),
        Animated.spring(slideAnim, {
          toValue: 0,
          tension: 25,
          friction: 8,
          useNativeDriver: true,
        }),
      ]).start();
    }
  }, [marketData]);

  const startAutoRefresh = () => {
    stopAutoRefresh();

    refreshIntervalRef.current = setInterval(() => {
      fetchMarketData(true);
    }, AUTO_REFRESH_INTERVAL);

    setCountdown(AUTO_REFRESH_INTERVAL / 1000);
    countdownIntervalRef.current = setInterval(() => {
      setCountdown((prev) => (prev <= 1 ? AUTO_REFRESH_INTERVAL / 1000 : prev - 1));
    }, 1000);
  };

  const stopAutoRefresh = () => {
    if (refreshIntervalRef.current) {
      clearInterval(refreshIntervalRef.current);
      refreshIntervalRef.current = null;
    }
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current);
      countdownIntervalRef.current = null;
    }
  };

  const fetchMarketData = async (silent = false) => {
    try {
      if (!silent) setLoading(true);
      setError(null);

      const response = await fetch(MARKET_URLS.NIFTY + '&t=' + Date.now(), {
        headers: { 'Cache-Control': 'no-cache', 'Pragma': 'no-cache' },
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const data = await response.json();
      setMarketData(data);
      setLastUpdated(new Date());

      if (autoRefreshEnabled) {
        setCountdown(AUTO_REFRESH_INTERVAL / 1000);
      }
    } catch (err) {
      console.error('Error fetching market data:', err);
      setError('Failed to load market data');
    } finally {
      if (!silent) setLoading(false);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchMarketData();
    setRefreshing(false);
  };

  const toggleAutoRefresh = () => {
    setAutoRefreshEnabled(!autoRefreshEnabled);
  };

  if (loading && !marketData) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#DC2626" />
        <Text style={styles.loadingText}>Loading market analysis...</Text>
      </View>
    );
  }

  if (error && !marketData) {
    return (
      <ScrollView
        style={styles.container}
        contentContainerStyle={styles.centerContainer}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
      >
        <Text style={styles.errorEmoji}>⚠️</Text>
        <Text style={styles.errorText}>{error}</Text>
        <Text style={styles.errorHint}>Pull down to retry</Text>
      </ScrollView>
    );
  }

  if (!marketData) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.errorText}>No data available</Text>
      </View>
    );
  }

  const d = marketData;
  const direction = d.market_outlook?.direction || d.final_decision?.bias || 'NEUTRAL';
  const colors = getActionColors(direction);
  const netBias = d.parallel_oi_analysis.cross_strike_analysis.net_bias;

  return (
    <ScrollView
      style={styles.container}
      showsVerticalScrollIndicator={false}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    >
      <Animated.View
        style={{
          opacity: fadeAnim,
          transform: [{ translateY: slideAnim }],
        }}
      >
        {viewMode === 'options' ? (
          <OptionChainView data={d} colors={colors} />
        ) : (
        {/* ⚠️ DISCLAIMER */}
        <View style={styles.disclaimer}>
          <Text style={styles.disclaimerIcon}>⚠️</Text>
          <Text style={styles.disclaimerText}>
            <Text style={styles.disclaimerBold}>Educational Only:</Text> Not financial advice.
            Consult a certified advisor before trading.
          </Text>
          {autoRefreshEnabled && (
            <View style={styles.liveIndicator}>
              <View style={styles.liveDot} />
              <Text style={styles.liveText}>LIVE</Text>
            </View>
          )}
        </View>

        {/* 🔄 STATUS BAR */}
        <View style={styles.statusBar}>
          <View style={styles.statusLeft}>
            <View
              style={[
                styles.statusDot,
                { backgroundColor: autoRefreshEnabled ? '#10B981' : '#94A3B8' },
              ]}
            />
            <Text style={styles.statusText}>
              {lastUpdated ? `${formatTime(lastUpdated)}` : 'Loading...'}
            </Text>
          </View>

          <View style={styles.viewModeToggle}>
            <TouchableOpacity
              style={[styles.viewModeBtn, viewMode === 'analysis' && styles.viewModeBtnActive]}
              onPress={() => setViewMode('analysis')}
            >
              <Text style={[styles.viewModeText, viewMode === 'analysis' && styles.viewModeTextActive]}>
                📊 Analysis
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.viewModeBtn, viewMode === 'options' && styles.viewModeBtnActive]}
              onPress={() => setViewMode('options')}
            >
              <Text style={[styles.viewModeText, viewMode === 'options' && styles.viewModeTextActive]}>
                ⚡ Options
              </Text>
            </TouchableOpacity>
          </View>
        </View>

        {/* QUICK REFRESH TOGGLE */}
        <View style={styles.refreshBar}>
          <TouchableOpacity
            style={styles.refreshButton}
            onPress={toggleAutoRefresh}
            activeOpacity={0.7}
          >
            <Text
              style={[
                styles.refreshButtonText,
                { color: autoRefreshEnabled ? '#10B981' : '#94A3B8' },
              ]}
            >
              {autoRefreshEnabled ? `🔄 Auto ${countdown}s` : '⏸ Paused'}
            </Text>
          </TouchableOpacity>
        </View>

        {/* ≡ƒÄ» TRADING DECISION HERO */}
        <LinearGradient colors={colors.gradient} style={styles.decisionHero}>
          <View style={styles.heroDecor}>
            <View style={[styles.decorCircle, styles.decorCircle1]} />
            <View style={[styles.decorCircle, styles.decorCircle2]} />
          </View>

          <View style={styles.heroContent}>
            <View style={styles.heroTop}>
              <PulseIndicator color="#fff" />
              <Text style={styles.heroAction}>{d.final_decision.action}</Text>
            </View>

            <Text style={styles.heroStrategy}>{d.final_decision.strategy}</Text>

            <View style={styles.heroBadges}>
              <View style={styles.heroBadge}>
                <Text style={styles.heroBadgeLabel}>CONFIDENCE</Text>
                <Text style={styles.heroBadgeValue}>{d.final_decision.confidence}</Text>
              </View>
              <View style={styles.heroBadge}>
                <Text style={styles.heroBadgeLabel}>BIAS</Text>
                <Text style={styles.heroBadgeValue}>{d.final_decision.bias}</Text>
              </View>
            </View>

            <View style={styles.heroReason}>
              <Text style={styles.heroReasonText}>{d.final_decision.reason}</Text>
            </View>
          </View>
        </LinearGradient>

        {/* ≡ƒôè TRADE SETUP */}
        <View style={styles.card}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionIcon}>📊</Text>
            <Text style={styles.sectionTitle}>Trade Setup</Text>
            <View style={[styles.setupBadge, { backgroundColor: colors.bg }]}>
              <Text style={[styles.setupBadgeText, { color: colors.text }]}>
                {d.trade_setup.strategy_type.replace(/_/g, ' ')}
              </Text>
            </View>
          </View>

          <View style={styles.tradeSetupGrid}>
            {/* Entry Zone */}
            <View style={styles.tradeSetupItem}>
              <Text style={styles.tradeLabel}>ENTRY ZONE</Text>
              <View style={styles.tradeRangeContainer}>
                <View style={[styles.tradeRange, { borderColor: colors.accent }]}>
                  <Text style={styles.tradeRangeValue}>
                    {d.trade_setup.entry_range.lower}
                  </Text>
                  <Text style={styles.tradeRangeSeparator}>ΓåÆ</Text>
                  <Text style={styles.tradeRangeValue}>
                    {d.trade_setup.entry_range.upper}
                  </Text>
                </View>
              </View>
            </View>

            {/* Stop Loss */}
            {d.trade_setup.stop_loss_levels.upside && (
              <View style={styles.tradeSetupItem}>
                <Text style={styles.tradeLabel}>STOP LOSS</Text>
                <View style={styles.tradeValueBox}>
                  <Text style={[styles.tradeValue, { color: '#DC2626' }]}>
                    {d.trade_setup.stop_loss_levels.upside}
                  </Text>
                  <Text style={styles.tradeSubtext}>Upside Risk</Text>
                </View>
              </View>
            )}

            {/* Current Spot */}
            <View style={styles.tradeSetupItem}>
              <Text style={styles.tradeLabel}>SPOT PRICE</Text>
              <View style={styles.tradeValueBox}>
                <Text style={styles.tradeValue}>Γé╣ {d.spot_price}</Text>
                <Text style={styles.tradeSubtext}>Current</Text>
              </View>
            </View>
          </View>

          <View style={styles.riskRewardNote}>
            <Text style={styles.riskRewardText}>
              💡 {d.trade_setup.risk_reward_interpretation}
            </Text>
          </View>
        </View>

        {/* ≡ƒöÑ MARKET PRESSURE */}
        <View style={styles.card}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionIcon}>🔥</Text>
            <Text style={styles.sectionTitle}>Market Pressure Analysis</Text>
          </View>

          <View style={styles.pressureContainer}>
            {/* Net Bias Display */}
            <View style={[styles.netBiasCard, { backgroundColor: colors.bg }]}>
              <Text style={styles.netBiasLabel}>NET OI BIAS</Text>
              <Text style={[styles.netBiasValue, { color: colors.text }]}>
                {netBias > 0 ? '+' : ''}
                {netBias.toLocaleString()}
              </Text>
              <Text style={styles.netBiasInterpretation}>
                {d.parallel_oi_analysis.cross_strike_analysis.bias_interpretation}
              </Text>
            </View>

            {/* PCR Indicators */}
            <View style={styles.pcrGrid}>
              <View style={styles.pcrCard}>
                <Text style={styles.pcrLabel}>PCR OI</Text>
                <Text style={styles.pcrValue}>{d.key_indicators.pcr_oi}</Text>
                <Text style={styles.pcrInterpretation}>
                  {d.key_indicators.pcr_interpretation}
                </Text>
              </View>

              <View style={styles.pcrCard}>
                <Text style={styles.pcrLabel}>ATM PCR</Text>
                <Text style={styles.pcrValue}>{d.key_indicators.atm_pcr}</Text>
                <Text style={styles.pcrInterpretation}>
                  {d.key_indicators.atm_pcr_interpretation}
                </Text>
              </View>
            </View>
          </View>
        </View>

        {/* ≡ƒÄ» RESISTANCE LEVELS */}
        <View style={styles.card}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionIcon}>🔴</Text>
            <Text style={styles.sectionTitle}>Resistance Levels</Text>
          </View>

          <Text style={styles.zoneSummary}>
            {d.parallel_oi_analysis.resistance_zone.summary}
          </Text>

          {d.zones.resistance.map((zone: any, idx: number) => (
            <View key={idx} style={styles.strikeRow}>
              <View style={styles.strikeLeft}>
                <Text style={styles.strikePrice}>{zone.strike}</Text>
                <Text style={styles.strikeType}>CE</Text>
              </View>

              <View style={styles.strikeCenter}>
                <View style={styles.oiChangeRow}>
                  <Text style={styles.oiChangeLabel}>Call OI</Text>
                  <Text
                    style={[
                      styles.oiChangeValue,
                      { color: zone.call_oi_change_pct > 0 ? '#DC2626' : '#10B981' },
                    ]}
                  >
                    {zone.call_oi_change_pct > 0 ? '+' : ''}
                    {zone.call_oi_change_pct.toFixed(1)}%
                  </Text>
                </View>
                <StrengthBars strength={zone.call_oi_change_pct} />
              </View>

              <View style={[styles.strikeCode, { backgroundColor: '#FEE2E2' }]}>
                <Text style={[styles.strikeCodeText, { color: '#7F1D1D' }]}>
                  {zone.interpretation_code.replace(/_/g, ' ')}
                </Text>
              </View>
            </View>
          ))}
        </View>

        {/* ≡ƒÄ» SUPPORT LEVELS */}
        <View style={styles.card}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionIcon}>🟢</Text>
            <Text style={styles.sectionTitle}>Support Levels</Text>
          </View>

          <Text style={styles.zoneSummary}>
            {d.parallel_oi_analysis.support_zone.summary}
          </Text>

          {d.zones.support.map((zone: any, idx: number) => (
            <View key={idx} style={styles.strikeRow}>
              <View style={styles.strikeLeft}>
                <Text style={styles.strikePrice}>{zone.strike}</Text>
                <Text style={styles.strikeType}>PE</Text>
              </View>

              <View style={styles.strikeCenter}>
                <View style={styles.oiChangeRow}>
                  <Text style={styles.oiChangeLabel}>Put OI</Text>
                  <Text
                    style={[
                      styles.oiChangeValue,
                      { color: zone.put_oi_change_pct > 0 ? '#10B981' : '#DC2626' },
                    ]}
                  >
                    {zone.put_oi_change_pct > 0 ? '+' : ''}
                    {zone.put_oi_change_pct.toFixed(1)}%
                  </Text>
                </View>
                <StrengthBars strength={Math.abs(zone.put_oi_change_pct)} />
              </View>

              <View style={[styles.strikeCode, { backgroundColor: '#D1FAE5' }]}>
                <Text style={[styles.strikeCodeText, { color: '#065F46' }]}>
                  {zone.interpretation_code.replace(/_/g, ' ')}
                </Text>
              </View>
            </View>
          ))}
        </View>

        {/* ≡ƒôê OI ACTIVITY SUMMARY */}
        <LinearGradient colors={['#F8FAFC', '#F1F5F9']} style={styles.card}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionIcon}>📈</Text>
            <Text style={styles.sectionTitle}>OI Activity Summary</Text>
          </View>

          <View style={styles.oiSummaryGrid}>
            <View style={styles.oiSummaryItem}>
              <Text style={styles.oiSummaryLabel}>Total Call Change</Text>
              <Text style={[styles.oiSummaryValue, { color: '#DC2626' }]}>
                +{d.parallel_oi_analysis.cross_strike_analysis.total_call_oi_change.toLocaleString()}
              </Text>
            </View>

            <View style={styles.oiSummaryItem}>
              <Text style={styles.oiSummaryLabel}>Total Put Change</Text>
              <Text style={[styles.oiSummaryValue, { color: '#10B981' }]}>
                {d.parallel_oi_analysis.cross_strike_analysis.total_put_oi_change.toLocaleString()}
              </Text>
            </View>
          </View>
        </LinearGradient>

        {/* ≡ƒòÉ METADATA */}
        <View style={styles.metadata}>
          <Text style={styles.metadataText}>
            Analysis: {new Date(d.metadata.analysis_timestamp).toLocaleString()} • Market:{' '}
            {d.metadata.market} • Quality: {d.metadata.data_quality}
          </Text>
        </View>
        )}

        <View style={{ height: 40 }} />
      </Animated.View>
    </ScrollView>
  );
}

/* ---------------- STYLES ---------------- */

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F8FAFC',
  },

  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },

  loadingText: {
    marginTop: 12,
    fontSize: 15,
    color: '#64748B',
    fontWeight: '700',
    fontFamily: 'System',
  },

  errorEmoji: {
    fontSize: 48,
    marginBottom: 12,
  },

  errorText: {
    fontSize: 16,
    color: '#DC2626',
    fontWeight: '700',
    textAlign: 'center',
    marginBottom: 8,
  },

  errorHint: {
    fontSize: 13,
    color: '#64748B',
    fontWeight: '600',
  },

  /* DISCLAIMER */
  disclaimer: {
    backgroundColor: '#FEF3C7',
    marginHorizontal: 16,
    marginTop: 12,
    marginBottom: 8,
    padding: 12,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#FDE047',
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: 8,
  },

  disclaimerIcon: {
    fontSize: 16,
  },

  disclaimerText: {
    flex: 1,
    fontSize: 11,
    lineHeight: 15,
    color: '#92400E',
    fontWeight: '600',
  },

  disclaimerBold: {
    fontWeight: '900',
    color: '#78350F',
  },

  liveIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#DCFCE7',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    gap: 4,
  },

  liveDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#16A34A',
  },

  liveText: {
    fontSize: 9,
    fontWeight: '800',
    color: '#16A34A',
  },

  /* STATUS BAR */
  statusBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 10,
    backgroundColor: '#fff',
    marginHorizontal: 16,
    marginBottom: 12,
    borderRadius: 12,
    elevation: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
  },

  statusLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },

  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },

  statusText: {
    fontSize: 12,
    color: '#64748B',
    fontWeight: '700',
    fontFamily: 'Menlo',
  },

  refreshButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#F8FAFC',
    borderRadius: 8,
  },

  refreshButtonText: {
    fontSize: 12,
    fontWeight: '800',
    fontFamily: 'Menlo',
  },

  refreshBar: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    paddingHorizontal: 16,
    paddingBottom: 10,
    backgroundColor: '#fff',
    marginHorizontal: 16,
    marginBottom: 8,
  },

  viewModeToggle: {
    flexDirection: 'row',
    backgroundColor: '#F1F5F9',
    borderRadius: 10,
    padding: 3,
  },

  viewModeBtn: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8,
  },

  viewModeBtnActive: {
    backgroundColor: '#fff',
  },

  viewModeText: {
    fontSize: 10,
    fontWeight: '700',
    color: '#64748B',
  },

  viewModeTextActive: {
    color: '#0F172A',
  },

  /* OPTION CHAIN STYLES */
  optionChainHeader: {
    flexDirection: 'row',
    marginBottom: 16,
    backgroundColor: '#F1F5F9',
    borderRadius: 12,
    padding: 4,
  },

  optionTab: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: 10,
  },

  optionTabActive: {
    backgroundColor: '#fff',
  },

  optionTabText: {
    fontSize: 13,
    fontWeight: '800',
    color: '#64748B',
  },

  optionTabTextActive: {
    color: '#0F172A',
  },

  optionChainGrid: {
    flexDirection: 'row',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
  },

  optionChainCol: {
    flex: 1,
    alignItems: 'center',
  },

  optionChainColHeader: {
    fontSize: 9,
    fontWeight: '800',
    color: '#64748B',
    letterSpacing: 0.5,
  },

  optionChainRow: {
    flexDirection: 'row',
    paddingVertical: 12,
    paddingHorizontal: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#F1F5F9',
  },

  optionChainRowATM: {
    backgroundColor: '#FEF3C7',
  },

  optionChainStrike: {
    fontSize: 16,
    fontWeight: '900',
    color: '#0F172A',
    fontFamily: 'Menlo',
  },

  optionChainStrikeATM: {
    fontSize: 18,
  },

  optionChainATMBadge: {
    fontSize: 8,
    fontWeight: '800',
    color: '#78350F',
    backgroundColor: '#FDE047',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    marginTop: 2,
  },

  optionChainPremium: {
    fontSize: 14,
    fontWeight: '900',
    fontFamily: 'Menlo',
  },

  callPremium: {
    color: '#059669',
  },

  putPremium: {
    color: '#DC2626',
  },

  optionChainOiChange: {
    fontSize: 12,
    fontWeight: '800',
    fontFamily: 'Menlo',
  },

  oiUp: {
    color: '#059669',
  },

  oiDown: {
    color: '#DC2626',
  },

  optionChainVolume: {
    fontSize: 12,
    fontWeight: '700',
    color: '#64748B',
    fontFamily: 'Menlo',
  },

  optionChainFooter: {
    marginTop: 16,
    padding: 12,
    backgroundColor: '#F0FDF4',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#BBF7D0',
  },

  optionChainFooterText: {
    fontSize: 12,
    fontWeight: '700',
    color: '#166534',
    textAlign: 'center',
  },

  /* DECISION HERO */
  decisionHero: {
    marginHorizontal: 16,
    marginBottom: 16,
    padding: 24,
    borderRadius: 20,
    overflow: 'hidden',
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 12,
  },

  heroDecor: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },

  decorCircle: {
    position: 'absolute',
    borderRadius: 1000,
    backgroundColor: 'rgba(255,255,255,0.08)',
  },

  decorCircle1: {
    width: 180,
    height: 180,
    top: -90,
    right: -40,
  },

  decorCircle2: {
    width: 140,
    height: 140,
    bottom: -70,
    left: -30,
  },

  heroContent: {
    position: 'relative',
  },

  heroTop: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
    marginBottom: 8,
  },

  heroAction: {
    fontSize: 44,
    fontWeight: '900',
    color: '#fff',
    letterSpacing: 2,
    fontFamily: 'System',
    textTransform: 'uppercase',
  },

  heroStrategy: {
    fontSize: 16,
    fontWeight: '700',
    color: 'rgba(255,255,255,0.95)',
    textAlign: 'center',
    marginBottom: 16,
    letterSpacing: 1,
  },

  heroBadges: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 12,
    marginBottom: 16,
  },

  heroBadge: {
    backgroundColor: 'rgba(255,255,255,0.2)',
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.25)',
    alignItems: 'center',
  },

  heroBadgeLabel: {
    fontSize: 9,
    fontWeight: '800',
    color: 'rgba(255,255,255,0.8)',
    letterSpacing: 1,
    marginBottom: 2,
  },

  heroBadgeValue: {
    fontSize: 14,
    fontWeight: '900',
    color: '#fff',
  },

  heroReason: {
    backgroundColor: 'rgba(255,255,255,0.15)',
    padding: 12,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.2)',
  },

  heroReasonText: {
    fontSize: 13,
    fontWeight: '700',
    color: '#fff',
    textAlign: 'center',
    lineHeight: 18,
  },

  /* PULSE INDICATOR */
  pulseOuter: {
    width: 20,
    height: 20,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },

  pulseInner: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },

  /* CARD */
  card: {
    backgroundColor: '#fff',
    marginHorizontal: 16,
    marginBottom: 16,
    padding: 18,
    borderRadius: 16,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.08,
    shadowRadius: 4,
  },

  /* SECTION HEADER */
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },

  sectionIcon: {
    fontSize: 22,
    marginRight: 8,
  },

  sectionTitle: {
    fontSize: 17,
    fontWeight: '900',
    color: '#0F172A',
    flex: 1,
  },

  setupBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8,
  },

  setupBadgeText: {
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.5,
  },

  /* TRADE SETUP */
  tradeSetupGrid: {
    gap: 16,
  },

  tradeSetupItem: {
    gap: 8,
  },

  tradeLabel: {
    fontSize: 11,
    fontWeight: '800',
    color: '#64748B',
    letterSpacing: 1,
  },

  tradeRangeContainer: {
    alignItems: 'flex-start',
  },

  tradeRange: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    backgroundColor: '#F8FAFC',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 12,
    borderWidth: 2,
  },

  tradeRangeValue: {
    fontSize: 20,
    fontWeight: '900',
    color: '#0F172A',
    fontFamily: 'Menlo',
  },

  tradeRangeSeparator: {
    fontSize: 16,
    fontWeight: '700',
    color: '#94A3B8',
  },

  tradeValueBox: {
    backgroundColor: '#F8FAFC',
    padding: 12,
    borderRadius: 12,
    alignItems: 'flex-start',
  },

  tradeValue: {
    fontSize: 22,
    fontWeight: '900',
    color: '#0F172A',
    fontFamily: 'Menlo',
    marginBottom: 2,
  },

  tradeSubtext: {
    fontSize: 11,
    fontWeight: '700',
    color: '#64748B',
  },

  riskRewardNote: {
    marginTop: 16,
    backgroundColor: '#F8FAFC',
    padding: 12,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#3B82F6',
  },

  riskRewardText: {
    fontSize: 13,
    fontWeight: '700',
    color: '#475569',
    lineHeight: 18,
  },

  /* PRESSURE CONTAINER */
  pressureContainer: {
    gap: 16,
  },

  netBiasCard: {
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },

  netBiasLabel: {
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 1,
    color: '#64748B',
    marginBottom: 4,
  },

  netBiasValue: {
    fontSize: 32,
    fontWeight: '900',
    fontFamily: 'Menlo',
    marginBottom: 4,
  },

  netBiasInterpretation: {
    fontSize: 12,
    fontWeight: '700',
    color: '#64748B',
    textAlign: 'center',
  },

  pcrGrid: {
    flexDirection: 'row',
    gap: 12,
  },

  pcrCard: {
    flex: 1,
    backgroundColor: '#F8FAFC',
    padding: 14,
    borderRadius: 12,
    alignItems: 'center',
  },

  pcrLabel: {
    fontSize: 11,
    fontWeight: '800',
    color: '#64748B',
    marginBottom: 4,
  },

  pcrValue: {
    fontSize: 24,
    fontWeight: '900',
    color: '#0F172A',
    fontFamily: 'Menlo',
    marginBottom: 4,
  },

  pcrInterpretation: {
    fontSize: 10,
    fontWeight: '700',
    color: '#64748B',
    textAlign: 'center',
  },

  /* ZONE SUMMARY */
  zoneSummary: {
    fontSize: 13,
    fontWeight: '700',
    color: '#475569',
    lineHeight: 19,
    marginBottom: 16,
    backgroundColor: '#F8FAFC',
    padding: 12,
    borderRadius: 10,
  },

  /* STRIKE ROW */
  strikeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    gap: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#F1F5F9',
  },

  strikeLeft: {
    width: 80,
  },

  strikePrice: {
    fontSize: 18,
    fontWeight: '900',
    color: '#0F172A',
    fontFamily: 'Menlo',
  },

  strikeType: {
    fontSize: 11,
    fontWeight: '800',
    color: '#64748B',
  },

  strikeCenter: {
    flex: 1,
    gap: 6,
  },

  oiChangeRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },

  oiChangeLabel: {
    fontSize: 11,
    fontWeight: '700',
    color: '#64748B',
  },

  oiChangeValue: {
    fontSize: 14,
    fontWeight: '900',
    fontFamily: 'Menlo',
  },

  strengthBars: {
    flexDirection: 'row',
    gap: 3,
    alignItems: 'flex-end',
  },

  strengthBar: {
    width: 6,
    borderRadius: 2,
  },

  strikeCode: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 8,
  },

  strikeCodeText: {
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 0.3,
  },

  /* OI SUMMARY */
  oiSummaryGrid: {
    flexDirection: 'row',
    gap: 12,
  },

  oiSummaryItem: {
    flex: 1,
    backgroundColor: '#fff',
    padding: 14,
    borderRadius: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#E5E7EB',
  },

  oiSummaryLabel: {
    fontSize: 10,
    fontWeight: '800',
    color: '#64748B',
    marginBottom: 6,
    textAlign: 'center',
  },

  oiSummaryValue: {
    fontSize: 18,
    fontWeight: '900',
    fontFamily: 'Menlo',
  },

  /* METADATA */
  metadata: {
    marginHorizontal: 16,
    marginBottom: 8,
    padding: 12,
    backgroundColor: '#fff',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#E5E7EB',
  },

  metadataText: {
    fontSize: 10,
    fontWeight: '700',
    color: '#64748B',
    textAlign: 'center',
    fontFamily: 'Menlo',
  },
});
