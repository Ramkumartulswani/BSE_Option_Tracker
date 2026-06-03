import React, { useRef, useEffect, useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
  Animated,
} from 'react-native';
import LinearGradient from 'react-native-linear-gradient';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import { useStats } from '../context/StatsContext';

const { width } = Dimensions.get('window');

/* ---------- Animated Counter (NON-BREAKING) ---------- */
const AnimatedNumber = ({ value }: { value: number }) => {
  const animated = useRef(new Animated.Value(value)).current;
  const [display, setDisplay] = useState(value);

  useEffect(() => {
    Animated.timing(animated, {
      toValue: value,
      duration: 600,
      useNativeDriver: false,
    }).start();

    const id = animated.addListener(({ value }) =>
      setDisplay(Math.floor(value))
    );

    return () => animated.removeListener(id);
  }, [value]);

  return <Text style={styles.statValue}>{display}</Text>;
};

export default function HomeScreen() {
  const navigation = useNavigation();
  const { stats, syncStats } = useStats();

  useFocusEffect(
    useCallback(() => {
      syncStats();
    }, [syncStats])
  );

  const quickStats = [
    {
      emoji: '🎮',
      label: 'Games Played',
      value: stats.gamesPlayed,
      color: ['#667eea', '#764ba2'],
    },
    {
      emoji: '😂',
      label: 'Jokes Read',
      value: stats.jokesRead,
      color: ['#f093fb', '#f5576c'],
    },
    {
      emoji: '💡',
      label: 'Facts Learned',
      value: stats.factsLearned,
      color: ['#4facfe', '#00f2fe'],
    },
    {
      emoji: '🏆',
      label: 'Achievements',
      value: stats.achievements,
      color: ['#43e97b', '#38f9d7'],
    },
  ];

  const quickActions = [
    {
      title: 'Market Insights',
      emoji: '📈',
      gradient: ['#11998e', '#38ef7d'],
      onPress: () => navigation.navigate('Market' as never),
    },
    {
      title: 'Play a Game',
      emoji: '🎮',
      gradient: ['#fa709a', '#fee140'],
      onPress: () => navigation.navigate('Games' as never),
    },
    {
      title: 'Get a Joke',
      emoji: '😄',
      gradient: ['#30cfd0', '#330867'],
      onPress: () => navigation.navigate('Jokes' as never),
    },
    {
      title: 'Learn Something',
      emoji: '🧠',
      gradient: ['#a8edea', '#fed6e3'],
      onPress: () => navigation.navigate('Facts' as never),
    },
  ];

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* Header */}
      <LinearGradient
        colors={['#667eea', '#764ba2']}
        style={styles.header}
      >
        <Text style={styles.greeting}>Hello! 👋</Text>
        <Text style={styles.subtitle}>Ready to have some fun today?</Text>
      </LinearGradient>

      {/* Quick Stats */}
      <View style={styles.statsContainer}>
        <Text style={styles.sectionTitle}>Your Fun Summary</Text>
        <View style={styles.statsGrid}>
          {quickStats.map((stat, index) => (
            <LinearGradient
              key={index}
              colors={stat.color}
              style={styles.statCard}
            >
              <Text style={styles.statEmoji}>{stat.emoji}</Text>
              <AnimatedNumber value={stat.value} />
              <Text style={styles.statLabel}>{stat.label}</Text>
            </LinearGradient>
          ))}
        </View>
      </View>

      {/* Quick Actions */}
      <View style={styles.actionsContainer}>
        <Text style={styles.sectionTitle}>Quick Actions</Text>
        {quickActions.map((action, index) => (
          <TouchableOpacity
            key={index}
            activeOpacity={0.85}
            onPress={action.onPress}
          >
            <LinearGradient
              colors={action.gradient}
              style={styles.actionCard}
            >
              <Text style={styles.actionEmoji}>{action.emoji}</Text>
              <Text style={styles.actionTitle}>{action.title}</Text>
              <Text style={styles.actionArrow}>→</Text>
            </LinearGradient>
          </TouchableOpacity>
        ))}
      </View>

      {/* Daily Challenge */}
      <View style={styles.challengeContainer}>
        <Text style={styles.sectionTitle}>Daily Challenge 🎯</Text>
        <LinearGradient
          colors={['#ff6b6b', '#ee5a6f']}
          style={styles.challengeCard}
        >
          <Text style={styles.challengeTitle}>
            Beat your best Reaction Time!
          </Text>
          <Text style={styles.challengeReward}>
            🏆 Reward: Speed Master Badge
          </Text>
          <TouchableOpacity
            style={styles.challengeButton}
            onPress={() => navigation.navigate('Games' as never)}
          >
            <Text style={styles.challengeButtonText}>Start Now</Text>
          </TouchableOpacity>
        </LinearGradient>
      </View>

      <View style={{ height: 40 }} />
    </ScrollView>
  );
}

/* ---------- STYLES (100% SAME AS YOUR ORIGINAL) ---------- */
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8fafc',
  },
  header: {
    padding: 30,
    paddingTop: 60,
    paddingBottom: 40,
    borderBottomLeftRadius: 30,
    borderBottomRightRadius: 30,
  },
  greeting: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 18,
    color: '#ffffff',
    opacity: 0.9,
  },
  sectionTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#1e293b',
    marginBottom: 16,
  },
  statsContainer: {
    padding: 20,
    marginTop: 20,
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  statCard: {
    width: (width - 60) / 2,
    padding: 20,
    borderRadius: 20,
    marginBottom: 16,
    alignItems: 'center',
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
  },
  statEmoji: {
    fontSize: 40,
    marginBottom: 8,
  },
  statValue: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    color: '#ffffff',
    opacity: 0.9,
    textAlign: 'center',
  },
  actionsContainer: {
    padding: 20,
  },
  actionCard: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
    borderRadius: 16,
    marginBottom: 12,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
  },
  actionEmoji: {
    fontSize: 32,
    marginRight: 16,
  },
  actionTitle: {
    flex: 1,
    fontSize: 18,
    fontWeight: '600',
    color: '#ffffff',
  },
  actionArrow: {
    fontSize: 24,
    color: '#ffffff',
    fontWeight: 'bold',
  },
  challengeContainer: {
    padding: 20,
  },
  challengeCard: {
    padding: 24,
    borderRadius: 20,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 8,
  },
  challengeTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 8,
  },
  challengeReward: {
    fontSize: 16,
    color: '#ffffff',
    opacity: 0.9,
    marginBottom: 16,
  },
  challengeButton: {
    backgroundColor: '#ffffff',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 12,
    alignSelf: 'flex-start',
  },
  challengeButtonText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ff6b6b',
  },
});
