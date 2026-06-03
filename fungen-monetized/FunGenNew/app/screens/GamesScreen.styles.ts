import { StyleSheet, Dimensions } from 'react-native';

const { width, height } = Dimensions.get('window');

export default StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0F172A',
  },

  /* ======================
     MENU SCREEN
     ====================== */
  menuContainer: {
    padding: 20,
    paddingBottom: 40,
  },

  header: {
    marginBottom: 24,
    alignItems: 'center',
  },

  title: {
    fontSize: 42,
    fontWeight: '900',
    color: '#FFFFFF',
    marginBottom: 8,
    letterSpacing: -1,
    textShadowColor: 'rgba(99, 102, 241, 0.5)',
    textShadowOffset: { width: 0, height: 4 },
    textShadowRadius: 12,
  },

  subtitle: {
    fontSize: 16,
    color: '#94A3B8',
    fontWeight: '600',
    letterSpacing: 0.5,
  },

  /* ======================
     STATS CARD
     ====================== */
  statsCard: {
    backgroundColor: '#1E293B',
    borderRadius: 20,
    padding: 20,
    marginBottom: 24,
    borderWidth: 1,
    borderColor: '#334155',
  },

  statsTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#FFFFFF',
    marginBottom: 16,
    letterSpacing: -0.3,
  },

  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },

  statItem: {
    alignItems: 'center',
    flex: 1,
  },

  statValue: {
    fontSize: 28,
    fontWeight: '800',
    color: '#6366F1',
    marginBottom: 4,
    letterSpacing: -0.5,
  },

  statLabel: {
    fontSize: 12,
    color: '#94A3B8',
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },

  /* ======================
     GAME CARDS GRID
     ====================== */
  gamesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 16,
    marginBottom: 24,
  },

  gameCard: {
    width: (width - 56) / 2,
    aspectRatio: 1,
    borderRadius: 20,
    padding: 20,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 12,
    elevation: 8,
    borderWidth: 2,
  },

  reactionCard: {
    backgroundColor: '#FCD34D',
    borderColor: '#F59E0B',
  },

  sequenceCard: {
    backgroundColor: '#A78BFA',
    borderColor: '#8B5CF6',
  },

  aimCard: {
    backgroundColor: '#FB923C',
    borderColor: '#F97316',
  },

  memoryCard: {
    backgroundColor: '#60A5FA',
    borderColor: '#3B82F6',
  },

  gameEmoji: {
    fontSize: 48,
    marginBottom: 12,
  },

  gameTitle: {
    fontSize: 16,
    fontWeight: '800',
    color: '#1E293B',
    marginBottom: 4,
    textAlign: 'center',
    letterSpacing: -0.3,
  },

  gameDesc: {
    fontSize: 12,
    color: '#475569',
    fontWeight: '600',
    textAlign: 'center',
  },

  gameBest: {
    fontSize: 11,
    color: '#1E293B',
    fontWeight: '700',
    marginTop: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.5)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },

  /* ======================
     ACHIEVEMENTS
     ====================== */
  achievementsSection: {
    marginBottom: 20,
  },

  achievementsTitle: {
    fontSize: 20,
    fontWeight: '800',
    color: '#FFFFFF',
    marginBottom: 16,
    letterSpacing: -0.3,
  },

  achievementsList: {
    gap: 12,
  },

  achievementCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1E293B',
    borderRadius: 16,
    padding: 16,
    borderWidth: 2,
    borderColor: '#334155',
    gap: 12,
  },

  achievementUnlocked: {
    backgroundColor: '#1E3A2E',
    borderColor: '#10B981',
  },

  achievementEmoji: {
    fontSize: 32,
  },

  achievementLocked: {
    opacity: 0.3,
  },

  achievementText: {
    flex: 1,
  },

  achievementTitle: {
    fontSize: 15,
    fontWeight: '700',
    color: '#FFFFFF',
    marginBottom: 2,
  },

  achievementDesc: {
    fontSize: 12,
    color: '#94A3B8',
    fontWeight: '500',
  },

  checkmark: {
    fontSize: 24,
    color: '#10B981',
  },

  /* ======================
     GAME CONTAINER
     ====================== */
  gameContainer: {
    flex: 1,
    backgroundColor: '#0F172A',
  },

  backButton: {
    padding: 20,
  },

  backButtonText: {
    fontSize: 16,
    color: '#FFFFFF',
    fontWeight: '600',
  },

  /* ======================
     REACTION GAME
     ====================== */
  reactionGameArea: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    margin: 20,
    borderRadius: 24,
    backgroundColor: '#EF4444',
    borderWidth: 4,
    borderColor: '#DC2626',
  },

  reactionWaiting: {
    backgroundColor: '#F59E0B',
    borderColor: '#D97706',
  },

  reactionReady: {
    backgroundColor: '#10B981',
    borderColor: '#059669',
  },

  reactionMessage: {
    fontSize: 48,
    fontWeight: '900',
    color: '#FFFFFF',
    textAlign: 'center',
    letterSpacing: -1,
    textShadowColor: 'rgba(0, 0, 0, 0.3)',
    textShadowOffset: { width: 0, height: 4 },
    textShadowRadius: 8,
  },

  reactionRating: {
    fontSize: 24,
    fontWeight: '700',
    color: '#FFFFFF',
    textAlign: 'center',
    marginTop: 24,
  },

  tapAgain: {
    fontSize: 16,
    color: 'rgba(255, 255, 255, 0.8)',
    fontWeight: '600',
    marginTop: 16,
    textAlign: 'center',
  },

  bestTimeCard: {
    backgroundColor: '#1E293B',
    margin: 20,
    marginTop: 0,
    padding: 20,
    borderRadius: 16,
    borderWidth: 2,
    borderColor: '#FCD34D',
    alignItems: 'center',
  },

  bestTimeLabel: {
    fontSize: 14,
    color: '#94A3B8',
    fontWeight: '600',
    marginBottom: 8,
  },

  bestTimeValue: {
    fontSize: 36,
    fontWeight: '900',
    color: '#FCD34D',
    letterSpacing: -1,
  },

  /* ======================
     SEQUENCE GAME
     ====================== */
  sequenceHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 20,
    paddingTop: 0,
  },

  sequenceLevel: {
    fontSize: 24,
    fontWeight: '800',
    color: '#FFFFFF',
  },

  sequenceScore: {
    fontSize: 24,
    fontWeight: '800',
    color: '#A78BFA',
  },

  sequenceGrid: {
    flex: 1,
    flexDirection: 'row',
    flexWrap: 'wrap',
    padding: 20,
    gap: 16,
    justifyContent: 'center',
    alignContent: 'center',
  },

  sequenceButton: {
    width: (width - 72) / 2,
    aspectRatio: 1,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
  },

  sequenceButton0: {
    backgroundColor: '#FCD34D',
    borderColor: '#F59E0B',
  },

  sequenceButton1: {
    backgroundColor: '#60A5FA',
    borderColor: '#3B82F6',
  },

  sequenceButton2: {
    backgroundColor: '#34D399',
    borderColor: '#10B981',
  },

  sequenceButton3: {
    backgroundColor: '#FB923C',
    borderColor: '#F97316',
  },

  sequenceButtonActive: {
    transform: [{ scale: 0.95 }],
    shadowColor: '#FFFFFF',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 20,
    elevation: 12,
  },

  sequenceButtonText: {
    fontSize: 64,
    fontWeight: '900',
    color: '#1E293B',
  },

  sequenceInstruction: {
    fontSize: 18,
    fontWeight: '700',
    color: '#FFFFFF',
    textAlign: 'center',
    paddingHorizontal: 20,
    paddingBottom: 40,
  },

  /* ======================
     AIM TRAINER
     ====================== */
  aimHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 20,
  },

  aimScore: {
    fontSize: 24,
    fontWeight: '800',
    color: '#FFFFFF',
  },

  aimTimer: {
    fontSize: 24,
    fontWeight: '800',
    color: '#FB923C',
  },

  aimGameArea: {
    flex: 1,
    backgroundColor: '#1E293B',
    margin: 20,
    marginTop: 0,
    borderRadius: 20,
    position: 'relative',
  },

  target: {
    position: 'absolute',
    width: 60,
    height: 60,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#EF4444',
    borderRadius: 30,
    borderWidth: 3,
    borderColor: '#FFFFFF',
    shadowColor: '#EF4444',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.6,
    shadowRadius: 12,
    elevation: 8,
  },

  targetEmoji: {
    fontSize: 32,
  },

  /* ======================
     MEMORY GAME
     ====================== */
  memoryHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: 20,
    paddingTop: 0,
  },

  memoryMoves: {
    fontSize: 18,
    fontWeight: '700',
    color: '#FFFFFF',
  },

  memoryMatched: {
    fontSize: 18,
    fontWeight: '700',
    color: '#60A5FA',
  },

  memoryGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    padding: 20,
    gap: 12,
    justifyContent: 'center',
  },

  memoryCardItem: {
    width: (width - 80) / 4,
    aspectRatio: 1,
    backgroundColor: '#1E293B',
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 3,
    borderColor: '#334155',
  },

  memoryCardFlipped: {
    backgroundColor: '#3B82F6',
    borderColor: '#60A5FA',
  },

  memoryCardEmoji: {
    fontSize: 32,
  },
});
