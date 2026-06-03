import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Dimensions,
  Animated,
  ActivityIndicator,
  StatusBar,
  TouchableOpacity,
  Share,
  ScrollView,
  Modal,
  FlatList,
} from 'react-native';
import LinearGradient from 'react-native-linear-gradient';
import AsyncStorage from '@react-native-async-storage/async-storage';

const { height } = Dimensions.get('window');
const AnimatedFlatList = Animated.createAnimatedComponent(FlatList);

/* ====================== TYPES ====================== */
type Quote = {
  id: string;
  text: string;
  author: string;
  category?: string;
  isFavorite?: boolean;
};

type ViewMode = 'swipe' | 'list' | 'daily' | 'favorites';

type QuoteStats = {
  totalViewed: number;
  favorites: number;
  shared: number;
};

/* ====================== CONFIG ====================== */
const QUOTES_API = 'https://zenquotes.io/api/quotes';
const CACHE_KEY = 'QUOTES_CACHE';
const CACHE_TIME_KEY = 'QUOTES_CACHE_TIME';
const CACHE_TTL = 60 * 60 * 1000;
const FAVORITES_KEY = 'FAVORITE_QUOTES';
const DAILY_QUOTE_KEY = 'DAILY_QUOTE';
const DAILY_DATE_KEY = 'DAILY_QUOTE_DATE';
const QUOTE_STATS_KEY = 'QUOTE_STATS';

const GRADIENT_THEMES = [
  ['#667eea', '#764ba2'],
  ['#f093fb', '#f5576c'],
  ['#4facfe', '#00f2fe'],
  ['#43e97b', '#38f9d7'],
  ['#fa709a', '#fee140'],
  ['#30cfd0', '#330867'],
  ['#a8edea', '#fed6e3'],
  ['#ff6b6b', '#ee5a6f'],
];

export default function QuotesScreen() {
  const [quotes, setQuotes] = useState<Quote[]>([]);
  const [favorites, setFavorites] = useState<Quote[]>([]);
  const [dailyQuote, setDailyQuote] = useState<Quote | null>(null);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>('swipe');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [menuVisible, setMenuVisible] = useState(false);
  const [stats, setStats] = useState<QuoteStats>({
    totalViewed: 0,
    favorites: 0,
    shared: 0,
  });

  const scrollY = useRef(new Animated.Value(0)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.9)).current;
  const heartAnim = useRef(new Animated.Value(1)).current;

  const displayQuotes =
    viewMode === 'favorites' ? favorites : quotes;

  /* ====================== EFFECTS ====================== */
  useEffect(() => {
    loadInitialData();
  }, []);

  useEffect(() => {
    if (currentIndex >= displayQuotes.length) {
      setCurrentIndex(0);
    }
  }, [displayQuotes]);

  useEffect(() => {
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
        useNativeDriver: true,
      }),
    ]).start();
  }, [viewMode]);

  /* ====================== DATA ====================== */
  const loadInitialData = async () => {
    try {
      setLoading(true);

      const s = await AsyncStorage.getItem(QUOTE_STATS_KEY);
      if (s) setStats(JSON.parse(s));

      const f = await AsyncStorage.getItem(FAVORITES_KEY);
      if (f) setFavorites(JSON.parse(f));

      await loadDailyQuote();
      await loadQuotes();

      setLoading(false);
    } catch {
      setLoading(false);
    }
  };

  const loadQuotes = async () => {
    try {
      const now = Date.now();
      const cached = await AsyncStorage.getItem(CACHE_KEY);
      const last = await AsyncStorage.getItem(CACHE_TIME_KEY);

      if (cached && last && now - Number(last) < CACHE_TTL) {
        setQuotes(JSON.parse(cached));
        return;
      }

      const res = await fetch(QUOTES_API, {
        headers: { Accept: 'application/json' },
      });

      const data = await res.json();
      if (!Array.isArray(data)) throw new Error('Invalid API');

      const formatted = data.map((q: any, i: number) => ({
        id: `${Date.now()}-${i}`,
        text: q.q,
        author: q.a || 'Unknown',
      }));

      setQuotes(formatted);
      await AsyncStorage.setItem(CACHE_KEY, JSON.stringify(formatted));
      await AsyncStorage.setItem(CACHE_TIME_KEY, now.toString());
    } catch {
      setQuotes([]);
    }
  };

  const loadDailyQuote = async () => {
    try {
      const today = new Date().toDateString();
      const savedDate = await AsyncStorage.getItem(DAILY_DATE_KEY);
      const saved = await AsyncStorage.getItem(DAILY_QUOTE_KEY);

      if (savedDate === today && saved) {
        setDailyQuote(JSON.parse(saved));
        return;
      }

      const res = await fetch('https://zenquotes.io/api/today');
      const data = await res.json();

      if (data?.[0]) {
        const q = {
          id: `daily-${Date.now()}`,
          text: data[0].q,
          author: data[0].a || 'Unknown',
        };
        setDailyQuote(q);
        await AsyncStorage.setItem(DAILY_QUOTE_KEY, JSON.stringify(q));
        await AsyncStorage.setItem(DAILY_DATE_KEY, today);
      }
    } catch {}
  };

  /* ====================== ACTIONS ====================== */
  const toggleFavorite = (q: Quote) => {
    Animated.sequence([
      Animated.timing(heartAnim, { toValue: 1.3, duration: 150, useNativeDriver: true }),
      Animated.spring(heartAnim, { toValue: 1, useNativeDriver: true }),
    ]).start();

    const exists = favorites.some(f => f.id === q.id);
    const updated = exists
      ? favorites.filter(f => f.id !== q.id)
      : [...favorites, q];

    setFavorites(updated);
    AsyncStorage.setItem(FAVORITES_KEY, JSON.stringify(updated));
  };

  const shareQuote = async (q: Quote) => {
    await Share.share({
      message: `"${q.text}"\n— ${q.author}`,
    });
  };

  /* ====================== UI ====================== */
  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#6366F1" />
        <Text style={styles.loadingText}>Loading quotes...</Text>
      </View>
    );
  }

  if (!displayQuotes.length) {
    return (
      <View style={styles.center}>
        <Text style={styles.emptyText}>No quotes available</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      <AnimatedFlatList
        data={displayQuotes}
        keyExtractor={(item) => item.id}
        pagingEnabled
        showsVerticalScrollIndicator={false}
        onMomentumScrollEnd={(e) => {
          const i = Math.round(e.nativeEvent.contentOffset.y / height);
          setCurrentIndex(i);
        }}
        renderItem={({ item, index }) => (
          <View style={{ height }}>
            <LinearGradient
              colors={GRADIENT_THEMES[index % GRADIENT_THEMES.length]}
              style={styles.page}
            >
              <Animated.View
                style={[
                  styles.card,
                  { opacity: fadeAnim, transform: [{ scale: scaleAnim }] },
                ]}
              >
                <Text style={styles.quoteText}>"{item.text}"</Text>
                <Text style={styles.authorText}>— {item.author}</Text>

                <View style={styles.actions}>
                  <TouchableOpacity onPress={() => toggleFavorite(item)}>
                    <Animated.Text style={{ transform: [{ scale: heartAnim }] }}>
                      ❤️
                    </Animated.Text>
                  </TouchableOpacity>
                  <TouchableOpacity onPress={() => shareQuote(item)}>
                    <Text>📤</Text>
                  </TouchableOpacity>
                </View>
              </Animated.View>
            </LinearGradient>
          </View>
        )}
      />
    </View>
  );
}

/* ====================== STYLES ====================== */
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  loadingText: { color: '#9CA3AF', marginTop: 12 },
  emptyText: { color: '#9CA3AF', fontSize: 18 },
  page: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  card: {
    padding: 32,
    borderRadius: 24,
    backgroundColor: 'rgba(0,0,0,0.3)',
    maxWidth: '85%',
  },
  quoteText: {
    fontSize: 26,
    lineHeight: 38,
    color: '#fff',
    textAlign: 'center',
    marginBottom: 20,
  },
  authorText: {
    fontSize: 18,
    color: 'rgba(255,255,255,0.8)',
    textAlign: 'center',
  },
  actions: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 20,
    marginTop: 20,
  },
});
