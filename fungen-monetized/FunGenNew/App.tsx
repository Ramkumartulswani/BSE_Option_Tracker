import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import AppNavigator from './app/navigation/AppNavigator';
import { StatsProvider } from './app/context/StatsContext';
import { ErrorBoundary } from './app/components/ErrorBoundary';
import mobileAds from 'react-native-google-mobile-ads';

mobileAds().initialize().then(() => {
  console.log('Ads initialized');
});

export default function App() {
  return (
    <ErrorBoundary>
      <StatsProvider>
        <NavigationContainer>
          <AppNavigator />
        </NavigationContainer>
      </StatsProvider>
    </ErrorBoundary>
  );
}
