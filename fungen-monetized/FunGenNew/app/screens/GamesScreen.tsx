import React, { useEffect, useRef, useState } from 'react';
import styles from './GamesScreen.styles';
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Animated,
  Dimensions,
  Alert,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

const { width } = Dimensions.get('window');

type GameMode = 'menu' | 'reaction' | 'sequence' | 'aim' | 'memory' | 'stats';

type Achievement = {
  id: string;
  title: string;
  description: string;
  emoji: string;
  unlocked: boolean;
};

type Stats = {
  gamesPlayed: number;
  totalScore: number;
  bestReaction: number | null;
  longestSequence: number;
  perfectAims: number;
  memoryHighScore: number;
};

export default function GamesScreen() {
  // Game Mode State
  const [gameMode, setGameMode] = useState<GameMode>('menu');

  // Reaction Game State
  const [status, setStatus] = useState<'idle' | 'waiting' | 'ready' | 'result'>('idle');
  const [message, setMessage] = useState('Tap to Start');
  const [startTime, setStartTime] = useState<number>(0);
  const [reactionTime, setReactionTime] = useState<number | null>(null);

  // Sequence Game State
  const [sequence, setSequence] = useState<number[]>([]);
  const [playerSequence, setPlayerSequence] = useState<number[]>([]);
  const [sequenceLevel, setSequenceLevel] = useState(1);
  const [showingSequence, setShowingSequence] = useState(false);
  const [activeButton, setActiveButton] = useState<number | null>(null);

  // Aim Game State
  const [targets, setTargets] = useState<Array<{ id: number; x: number; y: number }>>([]);
  const [aimScore, setAimScore] = useState(0);
  const [aimTimeLeft, setAimTimeLeft] = useState(30);
  const [aimActive, setAimActive] = useState(false);

  // Memory Game State
  const [memoryCards, setMemoryCards] = useState<Array<{ id: number; emoji: string; flipped: boolean; matched: boolean }>>([]);
  const [memoryFlipped, setMemoryFlipped] = useState<number[]>([]);
  const [memoryMoves, setMemoryMoves] = useState(0);
  const [memoryMatched, setMemoryMatched] = useState(0);

  // Stats & Achievements
  const [stats, setStats] = useState<Stats>({
    gamesPlayed: 0,
    totalScore: 0,
    bestReaction: null,
    longestSequence: 0,
    perfectAims: 0,
    memoryHighScore: 0,
  });
  const [achievements, setAchievements] = useState<Achievement[]>([
    { id: 'speed_demon', title: 'Speed Demon', description: 'React in under 200ms', emoji: '⚡', unlocked: false },
    { id: 'perfect_memory', title: 'Perfect Memory', description: 'Match all pairs in under 20 moves', emoji: '🧠', unlocked: false },
    { id: 'sharpshooter', title: 'Sharpshooter', description: 'Hit 50 targets in Aim Trainer', emoji: '🎯', unlocked: false },
    { id: 'sequence_master', title: 'Sequence Master', description: 'Reach level 10 in Sequence', emoji: '🔢', unlocked: false },
    { id: 'century', title: 'Century Club', description: 'Play 100 games', emoji: '💯', unlocked: false },
  ]);

  // Animations
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.8)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const timeoutRef = useRef<any>(null);
  const aimIntervalRef = useRef<any>(null);

  useEffect(() => {
    loadGameData();
    animateEntry();
  }, []);

  useEffect(() => {
    if (gameMode !== 'menu') {
      animateEntry();
    }
  }, [gameMode]);

  const animateEntry = () => {
    fadeAnim.setValue(0);
    scaleAnim.setValue(0.8);
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 400,
        useNativeDriver: true,
      }),
      Animated.spring(scaleAnim, {
        toValue: 1,
        friction: 6,
        tension: 40,
        useNativeDriver: true,
      }),
    ]).start();
  };

  const loadGameData = async () => {
    try {
      const savedStats = await AsyncStorage.getItem('GAME_STATS');
      const savedAchievements = await AsyncStorage.getItem('GAME_ACHIEVEMENTS');
      if (savedStats) setStats(JSON.parse(savedStats));
      if (savedAchievements) setAchievements(JSON.parse(savedAchievements));
    } catch (error) {
      console.error('Failed to load game data:', error);
    }
  };

  const saveGameData = async (newStats: Stats, newAchievements: Achievement[]) => {
    try {
      await AsyncStorage.setItem('GAME_STATS', JSON.stringify(newStats));
      await AsyncStorage.setItem('GAME_ACHIEVEMENTS', JSON.stringify(newAchievements));
      setStats(newStats);
      setAchievements(newAchievements);
    } catch (error) {
      console.error('Failed to save game data:', error);
    }
  };

  const checkAchievements = (newStats: Stats) => {
    const newAchievements = [...achievements];
    let updated = false;

    // Speed Demon
    if (newStats.bestReaction && newStats.bestReaction < 200 && !newAchievements[0].unlocked) {
      newAchievements[0].unlocked = true;
      updated = true;
      Alert.alert('🏆 Achievement Unlocked!', 'Speed Demon - React in under 200ms!');
    }

    // Century Club
    if (newStats.gamesPlayed >= 100 && !newAchievements[4].unlocked) {
      newAchievements[4].unlocked = true;
      updated = true;
      Alert.alert('🏆 Achievement Unlocked!', 'Century Club - 100 games played!');
    }

    if (updated) {
      saveGameData(newStats, newAchievements);
    }
  };

  // ==================== REACTION GAME ====================
  const startReactionGame = () => {
    setStatus('waiting');
    setMessage('Wait for GREEN...');
    setReactionTime(null);
    const delay = Math.random() * 3000 + 2000;
    timeoutRef.current = setTimeout(() => {
      setStatus('ready');
      setMessage('TAP NOW!');
      setStartTime(Date.now());
    }, delay);
  };

  const handleReactionTap = () => {
    if (status === 'waiting') {
      clearTimeout(timeoutRef.current);
      Alert.alert('Too Soon! 😅', 'Wait for the green signal');
      setStatus('idle');
      setMessage('Tap to Start');
      return;
    }
    if (status === 'ready') {
      const time = Date.now() - startTime;
      setReactionTime(time);
      setStatus('result');
      setMessage(`⏱️ ${time} ms`);

      const newStats = {
        ...stats,
        gamesPlayed: stats.gamesPlayed + 1,
        totalScore: stats.totalScore + Math.max(1000 - time, 0),
        bestReaction: !stats.bestReaction || time < stats.bestReaction ? time : stats.bestReaction,
      };
      saveGameData(newStats, achievements);
      checkAchievements(newStats);
    }
  };

  // ==================== SEQUENCE GAME ====================
  const startSequenceGame = () => {
    setSequenceLevel(1);
    playSequence([Math.floor(Math.random() * 4)]);
  };

  const playSequence = (seq: number[]) => {
    setSequence(seq);
    setPlayerSequence([]);
    setShowingSequence(true);

    seq.forEach((num, index) => {
      setTimeout(() => {
        setActiveButton(num);
        setTimeout(() => setActiveButton(null), 400);
      }, index * 600);
    });

    setTimeout(() => {
      setShowingSequence(false);
    }, seq.length * 600 + 400);
  };

  const handleSequenceButton = (num: number) => {
    if (showingSequence) return;

    const newPlayerSeq = [...playerSequence, num];
    setPlayerSequence(newPlayerSeq);

    setActiveButton(num);
    setTimeout(() => setActiveButton(null), 200);

    if (newPlayerSeq[newPlayerSeq.length - 1] !== sequence[newPlayerSeq.length - 1]) {
      Alert.alert('Game Over! 😢', `You reached level ${sequenceLevel}`);
      const newStats = {
        ...stats,
        gamesPlayed: stats.gamesPlayed + 1,
        totalScore: stats.totalScore + sequenceLevel * 100,
        longestSequence: Math.max(stats.longestSequence, sequenceLevel),
      };
      saveGameData(newStats, achievements);
      setGameMode('menu');
      return;
    }

    if (newPlayerSeq.length === sequence.length) {
      setTimeout(() => {
        const newLevel = sequenceLevel + 1;
        setSequenceLevel(newLevel);
        playSequence([...sequence, Math.floor(Math.random() * 4)]);
      }, 500);
    }
  };

  // ==================== AIM TRAINER ====================
  const startAimGame = () => {
    setAimScore(0);
    setAimTimeLeft(30);
    setAimActive(true);
    spawnTarget();

    aimIntervalRef.current = setInterval(() => {
      setAimTimeLeft((prev) => {
        if (prev <= 1) {
          endAimGame();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const spawnTarget = () => {
    const newTarget = {
      id: Date.now(),
      x: Math.random() * (width - 100) + 20,
      y: Math.random() * 400 + 100,
    };
    setTargets([newTarget]);
  };

  const hitTarget = (id: number) => {
    setAimScore((prev) => prev + 1);
    setTargets([]);
    setTimeout(spawnTarget, 300);
  };

  const endAimGame = () => {
    clearInterval(aimIntervalRef.current);
    setAimActive(false);
    setTargets([]);
    Alert.alert('Time\'s Up! 🎯', `You hit ${aimScore} targets!`);
    const newStats = {
      ...stats,
      gamesPlayed: stats.gamesPlayed + 1,
      totalScore: stats.totalScore + aimScore * 10,
      perfectAims: aimScore >= 50 ? stats.perfectAims + 1 : stats.perfectAims,
    };
    saveGameData(newStats, achievements);
    setGameMode('menu');
  };

  // ==================== MEMORY GAME ====================
  const startMemoryGame = () => {
    const emojis = ['🎮', '🎯', '⚡', '🔥', '💎', '🌟', '🎨', '🎭'];
    const cards = [...emojis, ...emojis]
      .sort(() => Math.random() - 0.5)
      .map((emoji, index) => ({
        id: index,
        emoji,
        flipped: false,
        matched: false,
      }));
    setMemoryCards(cards);
    setMemoryFlipped([]);
    setMemoryMoves(0);
    setMemoryMatched(0);
  };

  const handleMemoryCard = (id: number) => {
    if (memoryFlipped.length === 2) return;
    if (memoryFlipped.includes(id)) return;
    if (memoryCards[id].matched) return;

    const newFlipped = [...memoryFlipped, id];
    setMemoryFlipped(newFlipped);

    const newCards = [...memoryCards];
    newCards[id].flipped = true;
    setMemoryCards(newCards);

    if (newFlipped.length === 2) {
      setMemoryMoves((prev) => prev + 1);
      const [first, second] = newFlipped;
      if (memoryCards[first].emoji === memoryCards[second].emoji) {
        newCards[first].matched = true;
        newCards[second].matched = true;
        setMemoryCards(newCards);
        setMemoryFlipped([]);
        const newMatched = memoryMatched + 2;
        setMemoryMatched(newMatched);

        if (newMatched === 16) {
          setTimeout(() => {
            Alert.alert('You Won! 🎉', `Completed in ${memoryMoves + 1} moves!`);
            const newStats = {
              ...stats,
              gamesPlayed: stats.gamesPlayed + 1,
              totalScore: stats.totalScore + Math.max(500 - memoryMoves * 10, 100),
              memoryHighScore: Math.min(stats.memoryHighScore || 999, memoryMoves + 1),
            };
            saveGameData(newStats, achievements);
            setGameMode('menu');
          }, 500);
        }
      } else {
        setTimeout(() => {
          newCards[first].flipped = false;
          newCards[second].flipped = false;
          setMemoryCards(newCards);
          setMemoryFlipped([]);
        }, 1000);
      }
    }
  };

  // ==================== RENDER GAME MENU ====================
  if (gameMode === 'menu') {
    return (
      <ScrollView style={styles.container}>
        <Animated.View
          style={[
            styles.menuContainer,
            { opacity: fadeAnim, transform: [{ scale: scaleAnim }] },
          ]}
        >
          {/* Header */}
          <View style={styles.header}>
            <Text style={styles.title}>🎮 Game Hub</Text>
            <Text style={styles.subtitle}>Test your skills & reflexes!</Text>
          </View>

          {/* Stats Card */}
          <View style={styles.statsCard}>
            <Text style={styles.statsTitle}>Your Stats</Text>
            <View style={styles.statsGrid}>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{stats.gamesPlayed}</Text>
                <Text style={styles.statLabel}>Games</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{stats.totalScore}</Text>
                <Text style={styles.statLabel}>Score</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>
                  {stats.bestReaction ? `${stats.bestReaction}ms` : '-'}
                </Text>
                <Text style={styles.statLabel}>Best Time</Text>
              </View>
            </View>
          </View>

          {/* Game Modes */}
          <View style={styles.gamesGrid}>
            <TouchableOpacity
              style={[styles.gameCard, styles.reactionCard]}
              onPress={() => {
                setGameMode('reaction');
                setStatus('idle');
                setMessage('Tap to Start');
              }}
              activeOpacity={0.8}
            >
              <Text style={styles.gameEmoji}>⚡</Text>
              <Text style={styles.gameTitle}>Reaction Time</Text>
              <Text style={styles.gameDesc}>Test your reflexes</Text>
              {stats.bestReaction && (
                <Text style={styles.gameBest}>Best: {stats.bestReaction}ms</Text>
              )}
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.gameCard, styles.sequenceCard]}
              onPress={() => {
                setGameMode('sequence');
                startSequenceGame();
              }}
              activeOpacity={0.8}
            >
              <Text style={styles.gameEmoji}>🔢</Text>
              <Text style={styles.gameTitle}>Sequence Memory</Text>
              <Text style={styles.gameDesc}>Remember the pattern</Text>
              {stats.longestSequence > 0 && (
                <Text style={styles.gameBest}>Best: Level {stats.longestSequence}</Text>
              )}
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.gameCard, styles.aimCard]}
              onPress={() => {
                setGameMode('aim');
                startAimGame();
              }}
              activeOpacity={0.8}
            >
              <Text style={styles.gameEmoji}>🎯</Text>
              <Text style={styles.gameTitle}>Aim Trainer</Text>
              <Text style={styles.gameDesc}>Hit the targets fast</Text>
              <Text style={styles.gameBest}>30 seconds</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.gameCard, styles.memoryCard]}
              onPress={() => {
                setGameMode('memory');
                startMemoryGame();
              }}
              activeOpacity={0.8}
            >
              <Text style={styles.gameEmoji}>🧠</Text>
              <Text style={styles.gameTitle}>Memory Match</Text>
              <Text style={styles.gameDesc}>Find the pairs</Text>
              {stats.memoryHighScore > 0 && (
                <Text style={styles.gameBest}>Best: {stats.memoryHighScore} moves</Text>
              )}
            </TouchableOpacity>
          </View>

          {/* Achievements */}
          <View style={styles.achievementsSection}>
            <Text style={styles.achievementsTitle}>🏆 Achievements</Text>
            <View style={styles.achievementsList}>
              {achievements.map((achievement) => (
                <View
                  key={achievement.id}
                  style={[
                    styles.achievementCard,
                    achievement.unlocked && styles.achievementUnlocked,
                  ]}
                >
                  <Text
                    style={[
                      styles.achievementEmoji,
                      !achievement.unlocked && styles.achievementLocked,
                    ]}
                  >
                    {achievement.emoji}
                  </Text>
                  <View style={styles.achievementText}>
                    <Text style={styles.achievementTitle}>{achievement.title}</Text>
                    <Text style={styles.achievementDesc}>{achievement.description}</Text>
                  </View>
                  {achievement.unlocked && (
                    <Text style={styles.checkmark}>✓</Text>
                  )}
                </View>
              ))}
            </View>
          </View>
        </Animated.View>
      </ScrollView>
    );
  }

  // ==================== REACTION GAME ====================
  if (gameMode === 'reaction') {
    return (
      <View style={styles.gameContainer}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => setGameMode('menu')}
        >
          <Text style={styles.backButtonText}>← Back</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.reactionGameArea,
            status === 'ready' && styles.reactionReady,
            status === 'waiting' && styles.reactionWaiting,
          ]}
          activeOpacity={1}
          onPress={status === 'idle' || status === 'result' ? startReactionGame : handleReactionTap}
        >
          <Animated.View style={{ opacity: fadeAnim }}>
            <Text style={styles.reactionMessage}>{message}</Text>
            {status === 'result' && reactionTime && (
              <>
                <Text style={styles.reactionRating}>
                  {reactionTime < 200
                    ? '🔥 Lightning Fast!'
                    : reactionTime < 300
                    ? '⚡ Excellent!'
                    : reactionTime < 400
                    ? '👍 Good!'
                    : '🐌 Try Again!'}
                </Text>
                <Text style={styles.tapAgain}>Tap to Play Again</Text>
              </>
            )}
          </Animated.View>
        </TouchableOpacity>

        {stats.bestReaction && (
          <View style={styles.bestTimeCard}>
            <Text style={styles.bestTimeLabel}>🏆 Personal Best</Text>
            <Text style={styles.bestTimeValue}>{stats.bestReaction} ms</Text>
          </View>
        )}
      </View>
    );
  }

  // ==================== SEQUENCE GAME ====================
  if (gameMode === 'sequence') {
    return (
      <View style={styles.gameContainer}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => setGameMode('menu')}
        >
          <Text style={styles.backButtonText}>← Back</Text>
        </TouchableOpacity>

        <View style={styles.sequenceHeader}>
          <Text style={styles.sequenceLevel}>Level {sequenceLevel}</Text>
          <Text style={styles.sequenceScore}>Score: {sequenceLevel * 100}</Text>
        </View>

        <View style={styles.sequenceGrid}>
          {[0, 1, 2, 3].map((num) => (
            <TouchableOpacity
              key={num}
              style={[
                styles.sequenceButton,
                activeButton === num && styles.sequenceButtonActive,
                num === 0 && styles.sequenceButton0,
                num === 1 && styles.sequenceButton1,
                num === 2 && styles.sequenceButton2,
                num === 3 && styles.sequenceButton3,
              ]}
              onPress={() => handleSequenceButton(num)}
              disabled={showingSequence}
              activeOpacity={0.7}
            >
              <Text style={styles.sequenceButtonText}>{num + 1}</Text>
            </TouchableOpacity>
          ))}
        </View>

        {showingSequence && (
          <Text style={styles.sequenceInstruction}>👀 Watch the pattern...</Text>
        )}
        {!showingSequence && playerSequence.length === 0 && (
          <Text style={styles.sequenceInstruction}>🎮 Your turn! Repeat the sequence</Text>
        )}
      </View>
    );
  }

  // ==================== AIM TRAINER ====================
  if (gameMode === 'aim') {
    return (
      <View style={styles.gameContainer}>
        <View style={styles.aimHeader}>
          <Text style={styles.aimScore}>🎯 Score: {aimScore}</Text>
          <Text style={styles.aimTimer}>⏰ {aimTimeLeft}s</Text>
        </View>

        <View style={styles.aimGameArea}>
          {targets.map((target) => (
            <TouchableOpacity
              key={target.id}
              style={[
                styles.target,
                { top: target.y, left: target.x },
              ]}
              onPress={() => hitTarget(target.id)}
              activeOpacity={0.7}
            >
              <Text style={styles.targetEmoji}>🎯</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>
    );
  }

  // ==================== MEMORY GAME ====================
  if (gameMode === 'memory') {
    return (
      <View style={styles.gameContainer}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => setGameMode('menu')}
        >
          <Text style={styles.backButtonText}>← Back</Text>
        </TouchableOpacity>

        <View style={styles.memoryHeader}>
          <Text style={styles.memoryMoves}>Moves: {memoryMoves}</Text>
          <Text style={styles.memoryMatched}>Matched: {memoryMatched}/16</Text>
        </View>

        <View style={styles.memoryGrid}>
          {memoryCards.map((card) => (
            <TouchableOpacity
              key={card.id}
              style={[
                styles.memoryCardItem,
                (card.flipped || card.matched) && styles.memoryCardFlipped,
              ]}
              onPress={() => handleMemoryCard(card.id)}
              activeOpacity={0.8}
              disabled={card.matched || memoryFlipped.length === 2}
            >
              <Text style={styles.memoryCardEmoji}>
                {card.flipped || card.matched ? card.emoji : '?'}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>
    );
  }

  return null;
}
