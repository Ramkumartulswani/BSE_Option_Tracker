import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  ScrollView,
  Platform,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

export default function JokesScreen() {
  const [jokes, setJokes] = useState([]);
  const [favorites, setFavorites] = useState([]);
  const [category, setCategory] = useState('ALL');
  const [index, setIndex] = useState(0);
  const [showPunchline, setShowPunchline] = useState(false);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState('jokes');
  const [totalViewed, setTotalViewed] = useState(0);

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      setLoading(true);
      
      // Load favorites
      const savedFavorites = await AsyncStorage.getItem('FAVORITE_JOKES');
      if (savedFavorites) {
        setFavorites(JSON.parse(savedFavorites));
      }

      // Load stats
      const savedViewed = await AsyncStorage.getItem('JOKES_VIEWED');
      if (savedViewed) {
        setTotalViewed(parseInt(savedViewed));
      }

      // Load initial jokes
      const initialJokes = await fetchJokes();
      setJokes(initialJokes);
      
      setLoading(false);
    } catch (error) {
      console.error('Failed to load:', error);
      setLoading(false);
    }
  };

  const fetchJokes = async () => {
    try {
      const res = await fetch('https://official-joke-api.appspot.com/jokes/general/ten');
      const data = await res.json();
      return data.map((j, idx) => ({
        id: j.id || Date.now() + idx,
        setup: j.setup,
        punchline: j.punchline,
        category: 'GENERAL',
      }));
    } catch (error) {
      console.error('Failed to fetch jokes:', error);
      return [];
    }
  };

  const saveFavorites = async (newFavorites) => {
    try {
      await AsyncStorage.setItem('FAVORITE_JOKES', JSON.stringify(newFavorites));
      setFavorites(newFavorites);
    } catch (error) {
      console.error('Failed to save:', error);
    }
  };

  const toggleFavorite = () => {
    if (!currentJoke) return;

    const isFav = favorites.some((f) => f.id === currentJoke.id);

    if (isFav) {
      saveFavorites(favorites.filter((f) => f.id !== currentJoke.id));
    } else {
      saveFavorites([...favorites, currentJoke]);
    }
  };

  const nextJoke = async () => {
    setShowPunchline(false);
    const displayJokes = viewMode === 'favorites' ? favorites : jokes;
    
    if (displayJokes.length === 0) return;
    
    setIndex((prev) => (prev + 1) % displayJokes.length);

    const newTotal = totalViewed + 1;
    setTotalViewed(newTotal);
    await AsyncStorage.setItem('JOKES_VIEWED', newTotal.toString());
  };

  const prevJoke = () => {
    setShowPunchline(false);
    const displayJokes = viewMode === 'favorites' ? favorites : jokes;
    
    if (displayJokes.length === 0) return;
    
    setIndex((prev) => (prev === 0 ? displayJokes.length - 1 : prev - 1));
  };

  const displayJokes = viewMode === 'favorites' ? favorites : jokes;
  const currentJoke = displayJokes[index];
  const isFavorite = currentJoke ? favorites.some((f) => f.id === currentJoke.id) : false;

  if (loading) {
    return (
      <View style={styles.center}>
        <Text style={styles.loadingEmoji}>😂</Text>
        <ActivityIndicator size="large" color="#6366f1" style={{ marginTop: 16 }} />
        <Text style={styles.loadingText}>Loading jokes...</Text>
      </View>
    );
  }

  if (!currentJoke || displayJokes.length === 0) {
    return (
      <View style={styles.center}>
        <Text style={styles.emptyEmoji}>😅</Text>
        <Text style={styles.emptyText}>
          {viewMode === 'favorites' ? 'No favorites yet' : 'No jokes available'}
        </Text>
        {viewMode !== 'favorites' && (
          <TouchableOpacity style={styles.retryButton} onPress={loadInitialData}>
            <Text style={styles.retryButtonText}>Retry</Text>
          </TouchableOpacity>
        )}
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <ScrollView showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={[styles.header, { backgroundColor: '#667eea' }]}>
          <Text style={styles.headerTitle}>😂 Jokes</Text>
          <Text style={styles.headerSubtitle}>Laugh out loud!</Text>
          <View style={styles.statsBox}>
            <Text style={styles.statsNumber}>{totalViewed}</Text>
            <Text style={styles.statsLabel}>Viewed</Text>
          </View>
        </View>

        {/* View Mode Toggle */}
        <View style={styles.modeToggle}>
          <TouchableOpacity
            style={[styles.modeButton, viewMode === 'jokes' && styles.modeButtonActive]}
            onPress={() => {
              setViewMode('jokes');
              setIndex(0);
              setShowPunchline(false);
            }}
          >
            <Text style={[styles.modeText, viewMode === 'jokes' && styles.modeTextActive]}>
              😂 All Jokes
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.modeButton, viewMode === 'favorites' && styles.modeButtonActive]}
            onPress={() => {
              setViewMode('favorites');
              setIndex(0);
              setShowPunchline(false);
            }}
          >
            <Text style={[styles.modeText, viewMode === 'favorites' && styles.modeTextActive]}>
              ❤️ Favorites ({favorites.length})
            </Text>
          </TouchableOpacity>
        </View>

        {/* Joke Card */}
        <View style={[styles.card, { backgroundColor: viewMode === 'favorites' ? '#f093fb' : '#4facfe' }]}>
          <View style={styles.categoryBadge}>
            <Text style={styles.categoryBadgeText}>{currentJoke.category}</Text>
          </View>

          <Text style={styles.setup}>{currentJoke.setup}</Text>

          {!showPunchline ? (
            <TouchableOpacity
              style={styles.revealButton}
              onPress={() => setShowPunchline(true)}
            >
              <Text style={styles.revealText}>Tap for punchline 👇</Text>
            </TouchableOpacity>
          ) : (
            <View>
              <View style={styles.punchlineDivider} />
              <Text style={styles.punchline}>{currentJoke.punchline}</Text>
            </View>
          )}

          {/* Action Buttons */}
          <View style={styles.actionButtons}>
            <TouchableOpacity style={styles.actionButton} onPress={toggleFavorite}>
              <Text style={styles.actionEmoji}>{isFavorite ? '❤️' : '🤍'}</Text>
            </TouchableOpacity>
          </View>
        </View>

        {/* Navigation */}
        <View style={styles.navigation}>
          <TouchableOpacity 
            style={styles.navButton} 
            onPress={prevJoke}
            disabled={displayJokes.length <= 1}
          >
            <Text style={styles.navArrow}>←</Text>
            <Text style={styles.navText}>Previous</Text>
          </TouchableOpacity>

          <View style={styles.counter}>
            <Text style={styles.counterText}>
              {index + 1} / {displayJokes.length}
            </Text>
          </View>

          <TouchableOpacity 
            style={styles.navButton} 
            onPress={nextJoke}
            disabled={displayJokes.length <= 1}
          >
            <Text style={styles.navText}>Next</Text>
            <Text style={styles.navArrow}>→</Text>
          </TouchableOpacity>
        </View>

        {/* Stats */}
        <View style={styles.statsCard}>
          <Text style={styles.statsCardTitle}>📊 Your Stats</Text>
          <View style={styles.statsGrid}>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>{totalViewed}</Text>
              <Text style={styles.statLabel}>Viewed</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>{favorites.length}</Text>
              <Text style={styles.statLabel}>Favorites</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statValue}>{jokes.length}</Text>
              <Text style={styles.statLabel}>In Library</Text>
            </View>
          </View>
        </View>

        <View style={{ height: 40 }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8fafc',
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f8fafc',
    paddingHorizontal: 20,
  },
  loadingEmoji: {
    fontSize: 60,
  },
  loadingText: {
    marginTop: 16,
    fontSize: 16,
    color: '#64748b',
    fontWeight: '600',
  },
  emptyEmoji: {
    fontSize: 60,
    marginBottom: 16,
  },
  emptyText: {
    fontSize: 18,
    color: '#334155',
    fontWeight: '600',
    marginBottom: 24,
    textAlign: 'center',
  },
  retryButton: {
    backgroundColor: '#6366f1',
    paddingHorizontal: 32,
    paddingVertical: 14,
    borderRadius: 14,
  },
  retryButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '700',
  },
  header: {
    padding: 30,
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingBottom: 24,
    borderBottomLeftRadius: 24,
    borderBottomRightRadius: 24,
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 4,
  },
  headerSubtitle: {
    fontSize: 16,
    color: '#fff',
    opacity: 0.9,
    marginBottom: 16,
  },
  statsBox: {
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 12,
  },
  statsNumber: {
    fontSize: 24,
    fontWeight: '900',
    color: '#ffffff',
  },
  statsLabel: {
    fontSize: 10,
    color: '#ffffff',
    opacity: 0.9,
  },
  modeToggle: {
    flexDirection: 'row',
    marginHorizontal: 20,
    marginTop: 20,
    marginBottom: 20,
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
  card: {
    marginHorizontal: 20,
    padding: 30,
    borderRadius: 24,
    minHeight: 300,
    justifyContent: 'center',
    marginBottom: 20,
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.15,
        shadowRadius: 12,
      },
      android: {
        elevation: 5,
      },
    }),
  },
  categoryBadge: {
    position: 'absolute',
    top: 20,
    right: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  categoryBadgeText: {
    color: '#ffffff',
    fontSize: 11,
    fontWeight: '700',
  },
  setup: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#fff',
    textAlign: 'center',
    marginBottom: 24,
    lineHeight: 30,
  },
  revealButton: {
    backgroundColor: 'rgba(255,255,255,0.3)',
    padding: 16,
    borderRadius: 14,
    alignSelf: 'center',
    borderWidth: 2,
    borderColor: 'rgba(255,255,255,0.5)',
  },
  revealText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  punchlineDivider: {
    height: 2,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    marginVertical: 16,
  },
  punchline: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    textAlign: 'center',
    lineHeight: 28,
  },
  actionButtons: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 24,
  },
  actionButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    width: 56,
    height: 56,
    borderRadius: 28,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.5)',
  },
  actionEmoji: {
    fontSize: 24,
  },
  navigation: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginHorizontal: 20,
    marginBottom: 20,
  },
  navButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#ffffff',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 14,
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 6,
      },
      android: {
        elevation: 2,
      },
    }),
    gap: 8,
  },
  navArrow: {
    fontSize: 20,
    color: '#6366f1',
    fontWeight: 'bold',
  },
  navText: {
    fontSize: 15,
    color: '#6366f1',
    fontWeight: '600',
  },
  counter: {
    alignItems: 'center',
  },
  counterText: {
    fontSize: 18,
    fontWeight: '700',
    color: '#334155',
  },
  statsCard: {
    backgroundColor: '#ffffff',
    borderRadius: 16,
    padding: 20,
    marginHorizontal: 20,
    marginBottom: 20,
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.08,
        shadowRadius: 8,
      },
      android: {
        elevation: 2,
      },
    }),
  },
  statsCardTitle: {
    fontSize: 18,
    fontWeight: '800',
    color: '#0f172a',
    marginBottom: 16,
    textAlign: 'center',
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 28,
    fontWeight: '900',
    color: '#6366f1',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    color: '#64748b',
  },
});
