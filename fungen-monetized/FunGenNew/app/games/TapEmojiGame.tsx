import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
  Dimensions,
} from 'react-native';
import LinearGradient from 'react-native-linear-gradient';


const { width, height } = Dimensions.get('window');
const EMOJIS = ['😂', '🔥', '🎉', '😜', '🚀', '🍕', '🐶', '⚽', '🌟', '💎', '🎮', '🍔'];

export default function TapEmojiGame() {
  const [emoji, setEmoji] = useState('😂');
  const [score, setScore] = useState(0);
  const [timeLeft, setTimeLeft] = useState(30);
  const [gameOver, setGameOver] = useState(false);
  const [highScore, setHighScore] = useState(42);
  const [scaleAnim] = useState(new Animated.Value(1));
  const [particles, setParticles] = useState<{ id: number; x: number; y: number }[]>([]);

  useEffect(() => {
    if (timeLeft === 0) {
      setGameOver(true);
      if (score > highScore) {
        setHighScore(score);
      }
      return;
    }
    if (!gameOver) {
      const timer = setTimeout(() => setTimeLeft((t) => t - 1), 1000);
      return () => clearTimeout(timer);
    }
  }, [timeLeft, gameOver, score, highScore]);

  const onTap = () => {
    if (gameOver) return;

    // Increase score
    setScore((s) => s + 1);

    // Change emoji
    const random = EMOJIS[Math.floor(Math.random() * EMOJIS.length)];
    setEmoji(random);

    // Animate
    Animated.sequence([
      Animated.timing(scaleAnim, {
        toValue: 1.2,
        duration: 100,
        useNativeDriver: true,
      }),
      Animated.timing(scaleAnim, {
        toValue: 1,
        duration: 100,
        useNativeDriver: true,
      }),
    ]).start();

    // Add particle effect
    const newParticle = {
      id: Date.now(),
      x: Math.random() * width,
      y: height * 0.4 + Math.random() * 100,
    };
    setParticles((prev) => [...prev, newParticle]);
    setTimeout(() => {
      setParticles((prev) => prev.filter((p) => p.id !== newParticle.id));
    }, 1000);
  };

  const restart = () => {
    setScore(0);
    setTimeLeft(30);
    setGameOver(false);
    setEmoji('😂');
    setParticles([]);
  };

  return (
    <LinearGradient
      colors={['#667eea', '#764ba2']}
      style={styles.container}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
    >
      {/* Particles */}
      {particles.map((particle) => (
        <Text
          key={particle.id}
          style={[
            styles.particle,
            { left: particle.x, top: particle.y },
          ]}
        >
          ✨
        </Text>
      ))}

      {/* Header */}
      <View style={styles.header}>
        <View style={styles.statBox}>
          <Text style={styles.statLabel}>⏱️ Time</Text>
          <Text style={styles.statValue}>{timeLeft}s</Text>
        </View>
        <View style={styles.statBox}>
          <Text style={styles.statLabel}>🏆 High Score</Text>
          <Text style={styles.statValue}>{highScore}</Text>
        </View>
      </View>

      {/* Score Display */}
      <View style={styles.scoreContainer}>
        <Text style={styles.scoreLabel}>Score</Text>
        <Text style={styles.score}>{score}</Text>
      </View>

      {/* Game Area */}
      {!gameOver ? (
        <View style={styles.gameArea}>
          <TouchableOpacity onPress={onTap} activeOpacity={0.8}>
            <Animated.View
              style={[
                styles.emojiBox,
                {
                  transform: [{ scale: scaleAnim }],
                },
              ]}
            >
              <Text style={styles.emoji}>{emoji}</Text>
            </Animated.View>
          </TouchableOpacity>
          <Text style={styles.instruction}>👆 Tap me!</Text>
        </View>
      ) : (
        <View style={styles.gameOverContainer}>
          <Text style={styles.gameOverEmoji}>
            {score > highScore ? '🎉' : score > 30 ? '🌟' : '🎮'}
          </Text>
          <Text style={styles.gameOverText}>Game Over!</Text>
          <Text style={styles.finalScore}>Final Score: {score}</Text>
          {score > highScore && (
            <Text style={styles.newRecord}>🏆 New High Score! 🏆</Text>
          )}
          <TouchableOpacity onPress={restart} style={styles.restartButton}>
            <LinearGradient
              colors={['#43e97b', '#38f9d7']}
              style={styles.restartGradient}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
            >
              <Text style={styles.restartText}>Play Again 🔄</Text>
            </LinearGradient>
          </TouchableOpacity>
        </View>
      )}

      {/* Tips */}
      {!gameOver && (
        <View style={styles.tips}>
          <Text style={styles.tipText}>
            💡 Tip: Tap as fast as you can to beat your high score!
          </Text>
        </View>
      )}
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingTop: 60,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  statBox: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 16,
    alignItems: 'center',
    minWidth: 140,
  },
  statLabel: {
    fontSize: 14,
    color: '#ffffff',
    opacity: 0.9,
    marginBottom: 4,
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  scoreContainer: {
    alignItems: 'center',
    marginBottom: 40,
  },
  scoreLabel: {
    fontSize: 18,
    color: '#ffffff',
    opacity: 0.9,
    marginBottom: 8,
  },
  score: {
    fontSize: 72,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  gameArea: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emojiBox: {
    width: 160,
    height: 160,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    borderRadius: 80,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: '#ffffff',
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 12,
  },
  emoji: {
    fontSize: 80,
  },
  instruction: {
    fontSize: 20,
    color: '#ffffff',
    fontWeight: '600',
    marginTop: 24,
  },
  gameOverContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  gameOverEmoji: {
    fontSize: 80,
    marginBottom: 20,
  },
  gameOverText: {
    fontSize: 36,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 12,
  },
  finalScore: {
    fontSize: 28,
    color: '#ffffff',
    marginBottom: 20,
  },
  newRecord: {
    fontSize: 20,
    color: '#fbbf24',
    fontWeight: 'bold',
    marginBottom: 30,
    textAlign: 'center',
  },
  restartButton: {
    borderRadius: 16,
    overflow: 'hidden',
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
  },
  restartGradient: {
    paddingHorizontal: 40,
    paddingVertical: 16,
  },
  restartText: {
    color: '#ffffff',
    fontSize: 20,
    fontWeight: 'bold',
  },
  tips: {
    paddingHorizontal: 20,
    paddingBottom: 40,
  },
  tipText: {
    fontSize: 14,
    color: '#ffffff',
    textAlign: 'center',
    opacity: 0.9,
  },
  particle: {
    position: 'absolute',
    fontSize: 24,
    opacity: 0.8,
  },
});
