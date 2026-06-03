import React from 'react';
import { View } from 'react-native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import Ionicons from 'react-native-vector-icons/Ionicons';
import { AdBanner } from '../components/AdBanner';

import HomeScreen from '../screens/HomeScreen';
import JokesScreen from '../screens/JokesScreen';
import QuotesScreen from '../screens/QuotesScreen';
import FactsScreen from '../screens/FactsScreen';
import GamesScreen from '../screens/GamesScreen';
import MarketScreen from '../screens/MarketScreen';
import MarketJournalScreen from '../screens/MarketJournalScreen';

const Tab = createBottomTabNavigator();

function TabNavigatorWithAds() {
  return (
    <View style={{ flex: 1 }}>
      <Tab.Navigator
        screenOptions={({ route }) => ({
          headerShown: true,
          tabBarActiveTintColor: '#4CAF50',
          tabBarInactiveTintColor: '#999',
          tabBarIcon: ({ color, size }) => {
            let iconName: any;

            switch (route.name) {
              case 'Home':
                iconName = 'home-outline';
                break;
              case 'Games':
                iconName = 'game-controller-outline';
                break;
              case 'Market':
                iconName = 'trending-up-outline';
                break;
              case 'Jokes':
                iconName = 'happy-outline';
                break;
              case 'Quotes':
                iconName = 'chatbubble-ellipses-outline';
                break;
              case 'Facts':
                iconName = 'bulb-outline';
                break;
              default:
                iconName = 'ellipse-outline';
            }

            return <Ionicons name={iconName} size={size} color={color} />;
          },
        })}
      >
        <Tab.Screen name="Home" component={HomeScreen} />
        <Tab.Screen name="Journal" component={MarketJournalScreen}
          options={{ tabBarIcon: ({ color, size }) => <Ionicons name="journal-outline" size={size} color={color} /> }} />
        <Tab.Screen name="Games" component={GamesScreen} />
        <Tab.Screen name="Market" component={MarketScreen} />
        <Tab.Screen name="Jokes" component={JokesScreen} />
        <Tab.Screen name="Quotes" component={QuotesScreen} />
        <Tab.Screen name="Facts" component={FactsScreen} />
      </Tab.Navigator>
      <AdBanner />
    </View>
  );
}

export default TabNavigatorWithAds;
