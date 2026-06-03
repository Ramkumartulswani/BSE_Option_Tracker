import { StyleSheet, Dimensions } from 'react-native';

const { width } = Dimensions.get('window');

export default StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F8FAFC',
  },

  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F8FAFC',
    padding: 24,
  },

  /* ======================
     LOADING STATE
     ====================== */
  loadingEmoji: {
    fontSize: 80,
    textAlign: 'center',
  },

  loadingText: {
    marginTop: 24,
    fontSize: 18,
    color: '#475569',
    fontWeight: '600',
    letterSpacing: 0.3,
  },

  /* ======================
     ERROR STATE
     ====================== */
  errorEmoji: {
    fontSize: 72,
    marginBottom: 16,
  },

  errorText: {
    fontSize: 24,
    fontWeight: '700',
    color: '#1E293B',
    marginBottom: 8,
    letterSpacing: -0.5,
  },

  errorSubtext: {
    fontSize: 15,
    color: '#64748B',
    marginBottom: 24,
  },

  retryButton: {
    backgroundColor: '#6366F1',
    paddingHorizontal: 32,
    paddingVertical: 14,
    borderRadius: 14,
    shadowColor: '#6366F1',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.35,
    shadowRadius: 16,
    elevation: 8,
  },

  retryButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
    letterSpacing: 0.5,
  },

  /* ======================
     PREMIUM HEADER
     ====================== */
  premiumHeader: {
    backgroundColor: '#0F172A',
    overflow: 'hidden',
  },

  headerGradient: {
    paddingTop: 16,
    paddingBottom: 20,
    paddingHorizontal: 20,
  },

  headerContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },

  headerTitle: {
    fontSize: 32,
    fontWeight: '800',
    color: '#FFFFFF',
    letterSpacing: -1,
    textShadowColor: 'rgba(0, 0, 0, 0.3)',
    textShadowOffset: { width: 0, height: 2 },
    textShadowRadius: 4,
  },

  headerSubtitle: {
    fontSize: 13,
    color: '#94A3B8',
    marginTop: 4,
    fontWeight: '600',
    letterSpacing: 1,
    textTransform: 'uppercase',
  },

  viewModeButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    width: 48,
    height: 48,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },

  viewModeIcon: {
    fontSize: 20,
  },

  /* ======================
     INDEX SELECTOR
     ====================== */
  selectorContainer: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    paddingVertical: 16,
    backgroundColor: '#FFFFFF',
    gap: 12,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },

  selectorWrapper: {
    flex: 1,
    flexDirection: 'row',
    backgroundColor: '#F1F5F9',
    borderRadius: 12,
    padding: 4,
    gap: 4,
  },

  selectorButton: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: 8,
    position: 'relative',
  },

  selectorActive: {
    backgroundColor: '#FFFFFF',
    shadowColor: '#6366F1',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 8,
    elevation: 3,
  },

  selectorText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#64748B',
    letterSpacing: 0.3,
  },

  selectorTextActive: {
    color: '#6366F1',
    fontWeight: '700',
  },

  activeIndicator: {
    position: 'absolute',
    bottom: -2,
    left: '25%',
    right: '25%',
    height: 3,
    backgroundColor: '#6366F1',
    borderRadius: 2,
  },

  refreshToggle: {
    width: 48,
    height: 48,
    backgroundColor: '#F1F5F9',
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: '#E2E8F0',
  },

  refreshIcon: {
    fontSize: 20,
  },

  /* ======================
     HERO CARD
     ====================== */
  heroCard: {
    marginHorizontal: 20,
    marginTop: 20,
    backgroundColor: '#FFFFFF',
    borderRadius: 20,
    padding: 24,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.08,
    shadowRadius: 16,
    elevation: 4,
    borderWidth: 1,
    borderColor: '#F1F5F9',
  },

  heroContent: {
    gap: 20,
  },

  spotPriceSection: {
    position: 'relative',
  },

  indexLabel: {
    fontSize: 13,
    color: '#64748B',
    fontWeight: '700',
    letterSpacing: 1.5,
    textTransform: 'uppercase',
    marginBottom: 8,
  },

  spotPrice: {
    fontSize: 48,
    fontWeight: '800',
    color: '#0F172A',
    letterSpacing: -2,
  },

  liveIndicator: {
    position: 'absolute',
    top: 0,
    right: 0,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#10B981',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    gap: 6,
  },

  liveDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#FFFFFF',
  },

  liveText: {
    color: '#FFFFFF',
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 1,
  },

  quickStats: {
    flexDirection: 'row',
    backgroundColor: '#F8FAFC',
    borderRadius: 12,
    padding: 16,
    gap: 20,
  },

  quickStat: {
    flex: 1,
  },

  quickStatLabel: {
    fontSize: 11,
    color: '#64748B',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 6,
  },

  quickStatValue: {
    fontSize: 24,
    fontWeight: '700',
    color: '#0F172A',
    letterSpacing: -0.5,
  },

  quickStatDivider: {
    width: 1,
    backgroundColor: '#E2E8F0',
  },

  /* ======================
     BIAS CARD
     ====================== */
  biasCard: {
    marginHorizontal: 20,
    marginTop: 16,
    borderRadius: 20,
    padding: 20,
    borderWidth: 2,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 16,
    elevation: 4,
  },

  biasHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },

  biasLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },

  biasEmoji: {
    fontSize: 56,
  },

  biasTitle: {
    fontSize: 28,
    fontWeight: '800',
    letterSpacing: -0.5,
    marginBottom: 6,
  },

  confidenceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },

  confidenceEmoji: {
    fontSize: 12,
  },

  confidenceLabel: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.8,
  },

  biasIcon: {
    fontSize: 40,
  },

  /* ======================
     SENTIMENT METER
     ====================== */
  sentimentMeter: {
    marginTop: 20,
    paddingTop: 20,
    borderTopWidth: 1,
    borderTopColor: 'rgba(0, 0, 0, 0.05)',
  },

  meterHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },

  meterLabel: {
    fontSize: 13,
    fontWeight: '700',
    color: '#475569',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },

  meterPercentage: {
    fontSize: 18,
    fontWeight: '800',
    color: '#0F172A',
  },

  meterBar: {
    height: 12,
    backgroundColor: '#F1F5F9',
    borderRadius: 6,
    overflow: 'hidden',
    position: 'relative',
  },

  meterFillBull: {
    height: '100%',
    backgroundColor: '#10B981',
    position: 'absolute',
    left: 0,
  },

  meterFillBear: {
    height: '100%',
    backgroundColor: '#EF4444',
  },

  meterLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
  },

  meterLabelBull: {
    fontSize: 12,
    fontWeight: '600',
    color: '#10B981',
  },

  meterLabelBear: {
    fontSize: 12,
    fontWeight: '600',
    color: '#EF4444',
  },

  /* ======================
     INDICATORS GRID
     ====================== */
  indicatorsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginHorizontal: 20,
    marginTop: 16,
    gap: 12,
  },

  indicatorCard: {
    width: (width - 52) / 2,
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 16,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 8,
    elevation: 2,
    borderWidth: 1,
    borderColor: '#F1F5F9',
  },

  indicatorHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },

  indicatorIcon: {
    fontSize: 28,
  },

  indicatorBadge: {
    backgroundColor: '#F1F5F9',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },

  indicatorBadgeText: {
    fontSize: 10,
    fontWeight: '700',
    color: '#475569',
    letterSpacing: 0.5,
  },

  indicatorValue: {
    fontSize: 26,
    fontWeight: '800',
    color: '#0F172A',
    letterSpacing: -1,
    marginBottom: 8,
  },

  indicatorFooter: {
    gap: 4,
  },

  indicatorLabel: {
    fontSize: 11,
    color: '#64748B',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.3,
  },

  indicatorTag: {
    alignSelf: 'flex-start',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    marginTop: 4,
  },

  indicatorTagText: {
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 0.3,
  },

  indicatorSubtext: {
    fontSize: 10,
    color: '#94A3B8',
    fontWeight: '500',
  },

  /* ======================
     ZONE TOTALS CARD
     ====================== */
  zoneTotalsCard: {
    marginHorizontal: 20,
    marginTop: 16,
    backgroundColor: '#FFFBEB',
    borderRadius: 16,
    padding: 20,
    borderWidth: 2,
    borderColor: '#FDE047',
    shadowColor: '#EAB308',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 12,
    elevation: 3,
  },

  zoneTotalsTitle: {
    fontSize: 16,
    fontWeight: '800',
    color: '#713F12',
    marginBottom: 16,
    letterSpacing: -0.3,
  },

  zoneTotalsGrid: {
    flexDirection: 'row',
    gap: 16,
  },

  zoneTotalItem: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    padding: 14,
  },

  zoneTotalLabel: {
    fontSize: 10,
    color: '#92400E',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 8,
  },

  zoneTotalValueGreen: {
    fontSize: 22,
    fontWeight: '800',
    color: '#10B981',
    letterSpacing: -0.5,
    marginBottom: 4,
  },

  zoneTotalValueRed: {
    fontSize: 22,
    fontWeight: '800',
    color: '#EF4444',
    letterSpacing: -0.5,
    marginBottom: 4,
  },

  zoneTotalSub: {
    fontSize: 10,
    color: '#78716C',
    fontWeight: '500',
  },

  zoneTotalDivider: {
    width: 2,
    backgroundColor: '#FDE047',
  },

  /* ======================
     ZONE SECTION
     ====================== */
  zoneSection: {
    marginTop: 24,
  },

  zoneSectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginHorizontal: 20,
    marginBottom: 12,
  },

  zoneHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },

  zoneEmoji: {
    fontSize: 24,
  },

  zoneTitle: {
    fontSize: 20,
    fontWeight: '800',
    color: '#0F172A',
    letterSpacing: -0.5,
  },

  zoneBadge: {
    backgroundColor: '#DBEAFE',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
  },

  resistanceBadge: {
    backgroundColor: '#FEE2E2',
  },

  zoneBadgeText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#1E40AF',
    letterSpacing: 0.5,
  },

  zoneCards: {
    marginHorizontal: 20,
    gap: 12,
  },

  zoneCard: {
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    padding: 16,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 8,
    elevation: 2,
    borderLeftWidth: 4,
    borderLeftColor: '#6366F1',
  },

  zoneCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#F1F5F9',
  },

  strikePrice: {
    fontSize: 22,
    fontWeight: '800',
    color: '#0F172A',
    letterSpacing: -0.5,
    marginBottom: 4,
  },

  zoneLevel: {
    fontSize: 11,
    color: '#64748B',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },

  strengthBadge: {
    backgroundColor: '#10B98120',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 8,
  },

  resistanceStrengthBadge: {
    backgroundColor: '#EF444420',
  },

  strengthText: {
    fontSize: 13,
    fontWeight: '800',
    color: '#10B981',
  },

  oiRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 10,
  },

  oiItem: {
    flex: 1,
  },

  oiLabel: {
    fontSize: 10,
    color: '#64748B',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.3,
    marginBottom: 4,
  },

  oiValue: {
    fontSize: 16,
    fontWeight: '700',
    color: '#0F172A',
    letterSpacing: -0.3,
  },

  strengthBar: {
    height: 6,
    backgroundColor: '#F1F5F9',
    borderRadius: 3,
    overflow: 'hidden',
    marginTop: 8,
  },

  strengthFill: {
    height: '100%',
    borderRadius: 3,
  },

  /* ======================
     SIGNALS CARD
     ====================== */
  signalsCard: {
    marginHorizontal: 20,
    marginTop: 24,
    backgroundColor: '#F0F9FF',
    borderRadius: 16,
    padding: 20,
    borderWidth: 2,
    borderColor: '#BAE6FD',
  },

  signalsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 16,
  },

  signalsIcon: {
    fontSize: 24,
  },

  signalsTitle: {
    fontSize: 18,
    fontWeight: '800',
    color: '#075985',
    letterSpacing: -0.3,
  },

  signalsList: {
    gap: 12,
  },

  signalItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 12,
  },

  signalDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#0EA5E9',
    marginTop: 6,
  },

  signalText: {
    flex: 1,
    fontSize: 14,
    color: '#0C4A6E',
    lineHeight: 22,
    fontWeight: '500',
  },

  /* ======================
     FOOTER
     ====================== */
  footerInfo: {
    marginHorizontal: 20,
    marginTop: 24,
    gap: 12,
  },

  timestampRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: '#F8FAFC',
    padding: 14,
    borderRadius: 12,
  },

  timestampIcon: {
    fontSize: 18,
  },

  timestampText: {
    flex: 1,
    fontSize: 13,
    color: '#475569',
    fontWeight: '600',
  },

  disclaimerBox: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    backgroundColor: '#FEF2F2',
    padding: 14,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#FEE2E2',
  },

  disclaimerIcon: {
    fontSize: 18,
    marginTop: 2,
  },

  disclaimerText: {
    flex: 1,
    fontSize: 12,
    color: '#991B1B',
    lineHeight: 18,
    fontWeight: '500',
  },
});
