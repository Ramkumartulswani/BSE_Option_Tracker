import React, { createContext, useContext, useState, ReactNode, useCallback, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

type Stats = {
  gamesPlayed: number;
  jokesRead: number;
  factsLearned: number;
  achievements: number;
};

type StatsContextType = {
  stats: Stats;
  incrementGame: () => void;
  incrementJoke: () => void;
  incrementFact: () => void;
  syncStats: () => void;
};

const StatsContext = createContext<StatsContextType | null>(null);

const STATS_KEYS = {
  gamesPlayed: 'GAME_STATS',
  jokesRead: 'JOKES_VIEWED',
  factsLearned: 'FACTS_READ',
};

export const StatsProvider = ({ children }: { children: ReactNode }) => {
  const [stats, setStats] = useState<Stats>({
    gamesPlayed: 0,
    jokesRead: 0,
    factsLearned: 0,
    achievements: 0,
  });

  const syncStats = useCallback(async () => {
    try {
      const [gameStats, jokesViewed, factsRead] = await Promise.all([
        AsyncStorage.getItem(STATS_KEYS.gamesPlayed),
        AsyncStorage.getItem(STATS_KEYS.jokesRead),
        AsyncStorage.getItem(STATS_KEYS.factsLearned),
      ]);

      const parsedGameStats = gameStats ? JSON.parse(gameStats) : { gamesPlayed: 0 };

      setStats({
        gamesPlayed: parsedGameStats.gamesPlayed || 0,
        jokesRead: jokesViewed ? parseInt(jokesViewed, 10) : 0,
        factsLearned: factsRead ? parseInt(factsRead, 10) : 0,
        achievements: 0,
      });
    } catch (e) {
      console.error('Failed to sync stats:', e);
    }
  }, []);

  useEffect(() => {
    syncStats();
  }, [syncStats]);

  const incrementGame = () =>
    setStats(prev => ({ ...prev, gamesPlayed: prev.gamesPlayed + 1 }));

  const incrementJoke = () =>
    setStats(prev => ({ ...prev, jokesRead: prev.jokesRead + 1 }));

  const incrementFact = () =>
    setStats(prev => ({ ...prev, factsLearned: prev.factsLearned + 1 }));

  return (
    <StatsContext.Provider
      value={{ stats, incrementGame, incrementJoke, incrementFact, syncStats }}
    >
      {children}
    </StatsContext.Provider>
  );
};

export const useStats = () => {
  const ctx = useContext(StatsContext);
  if (!ctx) {
    throw new Error('useStats must be used inside StatsProvider');
  }
  return ctx;
};
