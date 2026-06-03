import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
  Animated,
  Share,
  Modal,
  ActivityIndicator,
  Alert,
} from 'react-native';
import LinearGradient from 'react-native-linear-gradient';
import AsyncStorage from '@react-native-async-storage/async-storage';

const { width } = Dimensions.get('window');

type Fact = {
  id: number | string;
  fact: string;
  category: string;
  emoji: string;
  gradient: string[];
  source?: string;
  isRead?: boolean;
  isFavorite?: boolean;
};

type FactStats = {
  totalRead: number;
  favorites: number;
  shared: number;
  quizScore: number;
};

const STATIC_FACTS: Fact[] = [
  {
    id: 1,
    fact: "Honey never spoils! Archaeologists have found 3000-year-old honey in Egyptian tombs that was still perfectly edible.",
    category: 'Nature',
    emoji: '🍯',
    gradient: ['#667eea', '#764ba2'],
  },
  {
    id: 2,
    fact: "Octopuses have three hearts and blue blood! Two hearts pump blood to the gills, while the third pumps it to the rest of the body.",
    category: 'Animals',
    emoji: '🐙',
    gradient: ['#f093fb', '#f5576c'],
  },
  {
    id: 3,
    fact: "The human brain uses 20% of the body's energy but only makes up 2% of body weight.",
    category: 'Science',
    emoji: '🧠',
    gradient: ['#4facfe', '#00f2fe'],
  },
  {
    id: 4,
    fact: "Bananas are berries, but strawberries aren't! In botanical terms, a berry is a fruit produced from a single flower with one ovary.",
    category: 'Food',
    emoji: '🍌',
    gradient: ['#43e97b', '#38f9d7'],
  },
  {
    id: 5,
    fact: "The Eiffel Tower can be 15 cm taller during the summer due to thermal expansion of the metal.",
    category: 'History',
    emoji: '🗼',
    gradient: ['#fa709a', '#fee140'],
  },
  {
    id: 6,
    fact: "A day on Venus is longer than a year on Venus! It takes 243 Earth days to rotate but only 225 Earth days to orbit the Sun.",
    category: 'Space',
    emoji: '🌍',
    gradient: ['#30cfd0', '#330867'],
  },
  {
    id: 7,
    fact: "Sharks existed before trees! Sharks have been around for about 400 million years, while trees appeared about 350 million years ago.",
    category: 'Animals',
    emoji: '🦈',
    gradient: ['#a8edea', '#fed6e3'],
  },
  {
    id: 8,
    fact: "The shortest war in history lasted 38-45 minutes between Britain and Zanzibar on August 27, 1896.",
    category: 'History',
    emoji: '⚔️',
    gradient: ['#ff6b6b', '#ee5a6f'],
  },
];

const CATEGORIES = ['All', 'Nature', 'Animals', 'Science', 'Food', 'History', 'Space', 'Technology'];

const GRADIENT_SETS = [
  ['#667eea', '#764ba2'],
  ['#f093fb', '#f5576c'],
  ['#4facfe', '#00f2fe'],
  ['#43e97b', '#38f9d7'],
  ['#fa709a', '#fee140'],
  ['#30cfd0', '#330867'],
  ['#a8edea', '#fed6e3'],
  ['#ff6b6b', '#ee5a6f'],
];

export default function FactsScreen() {
  const [facts, setFacts] = useState<Fact[]>(STATIC_FACTS);
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [readFacts, setReadFacts] = useState<(number | string)[]>([]);
  const [favorites, setFavorites] = useState<Fact[]>([]);
  const [viewMode, setViewMode] = useState<'facts' | 'favorites' | 'quiz'>('facts');
  const [loading, setLoading] = useState(false);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [selectedFact, setSelectedFact] = useState<Fact | null>(null);
  const [stats, setStats] = useState<FactStats>({
    totalRead: 0,
    favorites: 0,
    shared: 0,
    quizScore: 0,
  });

  // Quiz state
  const [quizMode, setQuizMode] = useState(false);
  const [currentQuizIndex, setCurrentQuizIndex] = useState(0);
  const [quizAnswers, setQuizAnswers] = useState<boolean[]>([]);
  const [showQuizResult, setShowQuizResult] = useState(false);

  // Animations
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.9)).current;
  const heartAnim = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    loadSavedData();
    fetchMoreFacts();
  }, []);

  useEffect(() => {
    animateEntry();
  }, [viewMode]);

  const animateEntry = () => {
    fadeAnim.setValue(0);
    scaleAnim.setValue(0.9);
    
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 400,
        useNativeDriver: true,
      }),
      Animated.spring(scaleAnim, {
        toValue: 1,
        friction: 8,
        tension: 40,
        useNativeDriver: true,
      }),
    ]).start();
  };

  const animateHeart = () => {
    Animated.sequence([
      Animated.timing(heartAnim, {
        toValue: 1.3,
        duration: 150,
        useNativeDriver: true,
      }),
      Animated.spring(heartAnim, {
        toValue: 1,
        friction: 3,
        tension: 40,
        useNativeDriver: true,
      }),
    ]).start();
  };

  const loadSavedData = async () => {
    try {
      const savedStats = await AsyncStorage.getItem('FACT_STATS');
      const savedFavorites = await AsyncStorage.getItem('FAVORITE_FACTS');
      const savedReadFacts = await AsyncStorage.getItem('READ_FACTS');

      if (savedStats) setStats(JSON.parse(savedStats));
      if (savedFavorites) setFavorites(JSON.parse(savedFavorites));
      if (savedReadFacts) setReadFacts(JSON.parse(savedReadFacts));
    } catch (error) {
      console.error('Failed to load saved data:', error);
    }
  };

  const saveStats = async (newStats: FactStats) => {
    try {
      await AsyncStorage.setItem('FACT_STATS', JSON.stringify(newStats));
      setStats(newStats);
    } catch (error) {
      console.error('Failed to save stats:', error);
    }
  };

  const saveFavorites = async (newFavorites: Fact[]) => {
    try {
      await AsyncStorage.setItem('FAVORITE_FACTS', JSON.stringify(newFavorites));
      setFavorites(newFavorites);
    } catch (error) {
      console.error('Failed to save favorites:', error);
    }
  };

  const saveReadFacts = async (newReadFacts: (number | string)[]) => {
    try {
      await AsyncStorage.setItem('READ_FACTS', JSON.stringify(newReadFacts));
      setReadFacts(newReadFacts);
    } catch (error) {
      console.error('Failed to save read facts:', error);
    }
  };

  const fetchMoreFacts = async () => {
    try {
      setLoading(true);
      
      // Fetch random facts from API
      const response = await fetch('https://uselessfacts.jsph.pl/api/v2/facts/random?language=en');
      const data = await response.json();

      if (data && data.text) {
        const newFact: Fact = {
          id: `api-${Date.now()}`,
          fact: data.text,
          category: 'Random',
          emoji: getRandomEmoji(),
          gradient: GRADIENT_SETS[Math.floor(Math.random() * GRADIENT_SETS.length)],
          source: 'API',
        };

        setFacts((prev) => [...prev, newFact]);
      }
    } catch (error) {
      console.error('Failed to fetch facts:', error);
    } finally {
      setLoading(false);
    }
  };

  const getRandomEmoji = () => {
    const emojis = ['💡', '🌟', '✨', '🔬', '🌈', '🎨', '🎭', '🎪', '🎯', '🎲', '🧪', '🔭'];
    return emojis[Math.floor(Math.random() * emojis.length)];
  };

  const markAsRead = (id: number | string) => {
    if (!readFacts.includes(id)) {
      const newReadFacts = [...readFacts, id];
      saveReadFacts(newReadFacts);
      
      const newStats = {
        ...stats,
        totalRead: newReadFacts.length,
      };
      saveStats(newStats);
    }
  };

  const toggleFavorite = (fact: Fact) => {
    animateHeart();

    const isFavorite = favorites.some((f) => f.id === fact.id);

    if (isFavorite) {
      const updated = favorites.filter((f) => f.id !== fact.id);
      saveFavorites(updated);
    } else {
      const updated = [...favorites, { ...fact, isFavorite: true }];
      saveFavorites(updated);
      
      const newStats = {
        ...stats,
        favorites: updated.length,
      };
      saveStats(newStats);
    }
  };

  const shareFact = async (fact: Fact) => {
    try {
      await Share.share({
        message: `${fact.emoji} Did you know?\n\n${fact.fact}\n\n📚 Shared from Amazing Facts`,
      });

      const newStats = {
        ...stats,
        shared: stats.shared + 1,
      };
      saveStats(newStats);
    } catch (error) {
      console.error('Failed to share:', error);
    }
  };

  const openDetail = (fact: Fact) => {
    setSelectedFact(fact);
    setDetailModalVisible(true);
    markAsRead(fact.id);
  };

  const startQuiz = () => {
    if (facts.length < 3) {
      Alert.alert('Not Enough Facts', 'Read more facts to start the quiz!');
      return;
    }

    setQuizMode(true);
    setCurrentQuizIndex(0);
    setQuizAnswers([]);
    setShowQuizResult(false);
  };

  const answerQuiz = (answer: boolean) => {
    const newAnswers = [...quizAnswers, answer];
    setQuizAnswers(newAnswers);

    if (currentQuizIndex < 4) {
      setCurrentQuizIndex(currentQuizIndex + 1);
    } else {
      // Quiz completed
      const score = newAnswers.filter((a) => a).length;
      setShowQuizResult(true);
      
      const newStats = {
        ...stats,
        quizScore: Math.max(stats.quizScore, score),
      };
      saveStats(newStats);
    }
  };

  const filteredFacts =
    selectedCategory === 'All'
      ? facts
      : facts.filter((fact) => fact.category === selectedCategory);

  const displayFacts = viewMode === 'favorites' ? favorites : filteredFacts;

  // Quiz rendering
  if (quizMode && !showQuizResult) {
    const quizFacts = facts.slice(0, 5);
    const currentFact = quizFacts[currentQuizIndex];

    return (
      <View style={styles.container}>
        <LinearGradient colors={['#667eea', '#764ba2']} style={styles.quizHeader}>
          <TouchableOpacity onPress={() => setQuizMode(false)}>
            <Text style={styles.quizBackButton}>← Exit Quiz</Text>
          </TouchableOpacity>
          <Text style={styles.quizProgress}>Question {currentQuizIndex + 1}/5</Text>
        </LinearGradient>

        <View style={styles.quizContainer}>
          <LinearGradient
            colors={currentFact.gradient}
            style={styles.quizCard}
          >
            <Text style={styles.quizEmoji}>{currentFact.emoji}</Text>
            <Text style={styles.quizQuestion}>Is this fact true?</Text>
            <Text style={styles.quizFactText}>{currentFact.fact}</Text>

            <View style={styles.quizButtons}>
              <TouchableOpacity
                style={[styles.quizButton, styles.quizButtonTrue]}
                onPress={() => answerQuiz(true)}
              >
                <Text style={styles.quizButtonText}>✓ True</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.quizButton, styles.quizButtonFalse]}
                onPress={() => answerQuiz(false)}
              >
                <Text style={styles.quizButtonText}>✗ False</Text>
              </TouchableOpacity>
            </View>
          </LinearGradient>
        </View>
      </View>
    );
  }

  // Quiz result
  if (showQuizResult) {
    const score = quizAnswers.filter((a) => a).length;
    const percentage = (score / 5) * 100;

    return (
      <View style={styles.container}>
        <LinearGradient colors={['#43e97b', '#38f9d7']} style={styles.resultContainer}>
          <Text style={styles.resultEmoji}>
            {percentage >= 80 ? '🏆' : percentage >= 60 ? '🌟' : '💪'}
          </Text>
          <Text style={styles.resultTitle}>Quiz Complete!</Text>
          <Text style={styles.resultScore}>{score} / 5 Correct</Text>
          <Text style={styles.resultPercentage}>{percentage}%</Text>

          <TouchableOpacity
            style={styles.resultButton}
            onPress={() => {
              setQuizMode(false);
              setShowQuizResult(false);
            }}
          >
            <Text style={styles.resultButtonText}>Back to Facts</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.resultButton, styles.resultButtonSecondary]}
            onPress={startQuiz}
          >
            <Text style={styles.resultButtonTextSecondary}>Try Again</Text>
          </TouchableOpacity>
        </LinearGradient>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <LinearGradient
        colors={['#667eea', '#764ba2']}
        style={styles.header}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
      >
        <Text style={styles.headerTitle}>💡 Amazing Facts</Text>
        <Text style={styles.headerSubtitle}>Expand your knowledge!</Text>
      </LinearGradient>

      {/* View Mode Toggle */}
      <View style={styles.modeToggle}>
        <TouchableOpacity
          style={[styles.modeButton, viewMode === 'facts' && styles.modeButtonActive]}
          onPress={() => setViewMode('facts')}
        >
          <Text style={[styles.modeText, viewMode === 'facts' && styles.modeTextActive]}>
            📚 All Facts
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.modeButton, viewMode === 'favorites' && styles.modeButtonActive]}
          onPress={() => setViewMode('favorites')}
        >
          <Text style={[styles.modeText, viewMode === 'favorites' && styles.modeTextActive]}>
            ⭐ Favorites ({favorites.length})
          </Text>
        </TouchableOpacity>
      </View>

      {/* Stats Bar */}
      <View style={styles.statsBar}>
        <View style={styles.statItem}>
          <Text style={styles.statValue}>{stats.totalRead}</Text>
          <Text style={styles.statLabel}>Read</Text>
        </View>
        <View style={styles.divider} />
        <View style={styles.statItem}>
          <Text style={styles.statValue}>{facts.length}</Text>
          <Text style={styles.statLabel}>Total</Text>
        </View>
        <View style={styles.divider} />
        <View style={styles.statItem}>
          <Text style={styles.statValue}>{stats.shared}</Text>
          <Text style={styles.statLabel}>Shared</Text>
        </View>
        <View style={styles.divider} />
        <View style={styles.statItem}>
          <Text style={styles.statValue}>{stats.quizScore}</Text>
          <Text style={styles.statLabel}>Quiz Best</Text>
        </View>
      </View>

      {/* Category Filter */}
      {viewMode === 'facts' && (
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          style={styles.categoriesContainer}
          contentContainerStyle={styles.categoriesContent}
        >
          {CATEGORIES.map((category) => (
            <TouchableOpacity
              key={category}
              onPress={() => setSelectedCategory(category)}
              style={[
                styles.categoryButton,
                selectedCategory === category && styles.categoryButtonActive,
              ]}
            >
              <Text
                style={[
                  styles.categoryButtonText,
                  selectedCategory === category && styles.categoryButtonTextActive,
                ]}
              >
                {category}
              </Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      )}

      {/* Action Buttons */}
      <View style={styles.actionBar}>
        <TouchableOpacity style={styles.actionButton} onPress={startQuiz}>
          <Text style={styles.actionButtonText}>🎯 Take Quiz</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.actionButton}
          onPress={fetchMoreFacts}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator size="small" color="#6366f1" />
          ) : (
            <Text style={styles.actionButtonText}>➕ Load More</Text>
          )}
        </TouchableOpacity>
      </View>

      {/* Facts List */}
      <ScrollView
        style={styles.factsContainer}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.factsContent}
      >
        <Animated.View
          style={{
            opacity: fadeAnim,
            transform: [{ scale: scaleAnim }],
          }}
        >
          {displayFacts.map((fact, index) => {
            const isRead = readFacts.includes(fact.id);
            const isFavorite = favorites.some((f) => f.id === fact.id);

            return (
              <TouchableOpacity
                key={fact.id}
                activeOpacity={0.9}
                onPress={() => openDetail(fact)}
              >
                <LinearGradient
                  colors={fact.gradient}
                  style={styles.factCard}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                >
                  {isRead && (
                    <View style={styles.readBadge}>
                      <Text style={styles.readBadgeText}>✓ Read</Text>
                    </View>
                  )}

                  <Text style={styles.factEmoji}>{fact.emoji}</Text>

                  <View style={styles.categoryTag}>
                    <Text style={styles.categoryTagText}>{fact.category}</Text>
                  </View>

                  <Text style={styles.factText} numberOfLines={4}>
                    {fact.fact}
                  </Text>

                  <View style={styles.factFooter}>
                    <TouchableOpacity
                      style={styles.actionIcon}
                      onPress={() => shareFact(fact)}
                    >
                      <Text style={styles.actionIconText}>📤</Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                      style={styles.actionIcon}
                      onPress={() => toggleFavorite(fact)}
                    >
                      <Animated.Text
                        style={[
                          styles.actionIconText,
                          { transform: [{ scale: heartAnim }] },
                        ]}
                      >
                        {isFavorite ? '⭐' : '☆'}
                      </Animated.Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                      style={styles.actionIcon}
                      onPress={() => openDetail(fact)}
                    >
                      <Text style={styles.actionIconText}>📖</Text>
                    </TouchableOpacity>
                  </View>
                </LinearGradient>
              </TouchableOpacity>
            );
          })}
        </Animated.View>
        <View style={{ height: 20 }} />
      </ScrollView>

      {/* Detail Modal */}
      <Modal
        visible={detailModalVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setDetailModalVisible(false)}
      >
        {selectedFact && (
          <View style={styles.modalOverlay}>
            <View style={styles.modalContent}>
              <LinearGradient
                colors={selectedFact.gradient}
                style={styles.modalHeader}
              >
                <Text style={styles.modalEmoji}>{selectedFact.emoji}</Text>
                <Text style={styles.modalCategory}>{selectedFact.category}</Text>
              </LinearGradient>

              <ScrollView style={styles.modalBody}>
                <Text style={styles.modalFactText}>{selectedFact.fact}</Text>

                {selectedFact.source && (
                  <Text style={styles.modalSource}>Source: {selectedFact.source}</Text>
                )}

                <View style={styles.modalActions}>
                  <TouchableOpacity
                    style={styles.modalActionButton}
                    onPress={() => {
                      toggleFavorite(selectedFact);
                      setDetailModalVisible(false);
                    }}
                  >
                    <Text style={styles.modalActionText}>
                      {favorites.some((f) => f.id === selectedFact.id) ? '⭐ Unfavorite' : '☆ Favorite'}
                    </Text>
                  </TouchableOpacity>

                  <TouchableOpacity
                    style={styles.modalActionButton}
                    onPress={() => {
                      shareFact(selectedFact);
                      setDetailModalVisible(false);
                    }}
                  >
                    <Text style={styles.modalActionText}>📤 Share</Text>
                  </TouchableOpacity>
                </View>
              </ScrollView>

              <TouchableOpacity
                style={styles.modalCloseButton}
                onPress={() => setDetailModalVisible(false)}
              >
                <Text style={styles.modalCloseText}>Close</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8fafc',
  },

  header: {
    padding: 30,
    paddingTop: 60,
    paddingBottom: 30,
  },

  headerTitle: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 8,
  },

  headerSubtitle: {
    fontSize: 18,
    color: '#ffffff',
    opacity: 0.9,
  },

  modeToggle: {
    flexDirection: 'row',
    marginHorizontal: 20,
    marginTop: -20,
    marginBottom: 16,
    backgroundColor: '#e5e7eb',
    borderRadius: 14,
    padding: 4,
  },

  modeButton: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: 10,
  },

  modeButtonActive: {
    backgroundColor: '#ffffff',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },

  modeText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#64748b',
  },

  modeTextActive: {
    color: '#6366f1',
    fontWeight: '700',
  },

  statsBar: {
    flexDirection: 'row',
    backgroundColor: '#ffffff',
    marginHorizontal: 20,
    marginBottom: 16,
    borderRadius: 16,
    padding: 16,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
  },

  statItem: {
    flex: 1,
    alignItems: 'center',
  },

  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#6366f1',
    marginBottom: 4,
  },

  statLabel: {
    fontSize: 11,
    color: '#64748b',
  },

  divider: {
    width: 1,
    backgroundColor: '#e5e7eb',
    marginHorizontal: 8,
  },

  categoriesContainer: {
    maxHeight: 50,
    marginBottom: 12,
  },

  categoriesContent: {
    paddingHorizontal: 20,
    paddingVertical: 8,
  },

  categoryButton: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
    backgroundColor: '#ffffff',
    marginRight: 12,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
  },

  categoryButtonActive: {
    backgroundColor: '#6366f1',
  },

  categoryButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#64748b',
  },

  categoryButtonTextActive: {
    color: '#ffffff',
  },

  actionBar: {
    flexDirection: 'row',
    marginHorizontal: 20,
    marginBottom: 16,
    gap: 12,
  },

  actionButton: {
    flex: 1,
    backgroundColor: '#ffffff',
    paddingVertical: 12,
    borderRadius: 14,
    alignItems: 'center',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },

  actionButtonText: {
    fontSize: 14,
    fontWeight: '700',
    color: '#6366f1',
  },

  factsContainer: {
    flex: 1,
  },

  factsContent: {
    padding: 20,
  },

  factCard: {
    borderRadius: 24,
    padding: 24,
    marginBottom: 16,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 8,
    position: 'relative',
  },

  readBadge: {
    position: 'absolute',
    top: 16,
    right: 16,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },

  readBadgeText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#ffffff',
  },

  factEmoji: {
    fontSize: 48,
    marginBottom: 12,
  },

  categoryTag: {
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    alignSelf: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    marginBottom: 16,
  },

  categoryTagText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#ffffff',
  },

  factText: {
    fontSize: 16,
    lineHeight: 24,
    color: '#ffffff',
    marginBottom: 16,
  },

  factFooter: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    gap: 12,
  },

  actionIcon: {
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
  },

  actionIconText: {
    fontSize: 20,
  },

  // Quiz Styles
  quizHeader: {
    padding: 20,
    paddingTop: 60,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },

  quizBackButton: {
    fontSize: 16,
    color: '#ffffff',
    fontWeight: '600',
  },

  quizProgress: {
    fontSize: 16,
    color: '#ffffff',
    fontWeight: '700',
  },

  quizContainer: {
    flex: 1,
    justifyContent: 'center',
    padding: 20,
  },

  quizCard: {
    borderRadius: 24,
    padding: 32,
    alignItems: 'center',
  },

  quizEmoji: {
    fontSize: 64,
    marginBottom: 24,
  },

  quizQuestion: {
    fontSize: 24,
    fontWeight: '700',
    color: '#ffffff',
    marginBottom: 16,
    textAlign: 'center',
  },

  quizFactText: {
    fontSize: 18,
    lineHeight: 28,
    color: '#ffffff',
    textAlign: 'center',
    marginBottom: 32,
  },

  quizButtons: {
    flexDirection: 'row',
    gap: 16,
    width: '100%',
  },

  quizButton: {
    flex: 1,
    paddingVertical: 16,
    borderRadius: 14,
    alignItems: 'center',
  },

  quizButtonTrue: {
    backgroundColor: 'rgba(16, 185, 129, 0.3)',
  },

  quizButtonFalse: {
    backgroundColor: 'rgba(239, 68, 68, 0.3)',
  },

  quizButtonText: {
    fontSize: 18,
    fontWeight: '700',
    color: '#ffffff',
  },

  // Result Styles
  resultContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
  },

  resultEmoji: {
    fontSize: 100,
    marginBottom: 24,
  },

  resultTitle: {
    fontSize: 32,
    fontWeight: '900',
    color: '#ffffff',
    marginBottom: 16,
  },

  resultScore: {
    fontSize: 48,
    fontWeight: '900',
    color: '#ffffff',
    marginBottom: 8,
  },

  resultPercentage: {
    fontSize: 24,
    fontWeight: '700',
    color: 'rgba(255, 255, 255, 0.9)',
    marginBottom: 40,
  },

  resultButton: {
    backgroundColor: '#ffffff',
    paddingHorizontal: 48,
    paddingVertical: 16,
    borderRadius: 14,
    marginBottom: 16,
    width: '100%',
    alignItems: 'center',
  },

  resultButtonText: {
    fontSize: 18,
    fontWeight: '700',
    color: '#43e97b',
  },

  resultButtonSecondary: {
    backgroundColor: 'transparent',
    borderWidth: 2,
    borderColor: '#ffffff',
  },

  resultButtonTextSecondary: {
    color: '#ffffff',
  },

  // Modal Styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'flex-end',
  },

  modalContent: {
    backgroundColor: '#ffffff',
    borderTopLeftRadius: 32,
    borderTopRightRadius: 32,
    maxHeight: '85%',
  },

  modalHeader: {
    padding: 32,
    borderTopLeftRadius: 32,
    borderTopRightRadius: 32,
    alignItems: 'center',
  },

  modalEmoji: {
    fontSize: 72,
    marginBottom: 16,
  },

  modalCategory: {
    fontSize: 16,
    fontWeight: '700',
    color: '#ffffff',
    textTransform: 'uppercase',
    letterSpacing: 1,
  },

  modalBody: {
    padding: 24,
  },

  modalFactText: {
    fontSize: 18,
    lineHeight: 28,
    color: '#1e293b',
    marginBottom: 20,
  },

  modalSource: {
    fontSize: 14,
    color: '#64748b',
    fontStyle: 'italic',
    marginBottom: 24,
  },

  modalActions: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },

  modalActionButton: {
    flex: 1,
    backgroundColor: '#f1f5f9',
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },

  modalActionText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#334155',
  },

  modalCloseButton: {
    backgroundColor: '#6366f1',
    paddingVertical: 16,
    alignItems: 'center',
    borderBottomLeftRadius: 32,
    borderBottomRightRadius: 32,
  },

  modalCloseText: {
    fontSize: 18,
    fontWeight: '700',
    color: '#ffffff',
  },
});
