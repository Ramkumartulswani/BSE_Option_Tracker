import { NativeModules, Platform } from 'react-native';

type FeatureFlags = {
  paywallEnabled: boolean;
  premiumEnabled: boolean;
  maxFreeUses: number;
  showUpgradeBanner: boolean;
  showMarketProBanner: boolean;
};

const defaultFlags: FeatureFlags = {
  paywallEnabled: false,
  premiumEnabled: true,
  maxFreeUses: 5,
  showUpgradeBanner: false,
  showMarketProBanner: true, // ✅ default = existing behavior
};

export async function getFeatureFlags(): Promise<FeatureFlags> {
  // iOS / Web / fallback
  if (Platform.OS !== 'android') {
    return defaultFlags;
  }

  try {
    const nativeModule = NativeModules.FeatureFlags;

    if (!nativeModule || typeof nativeModule.getFlags !== 'function') {
      return defaultFlags;
    }

    const nativeFlags = await nativeModule.getFlags();

    return {
      ...defaultFlags,
      ...nativeFlags,

      // 👇 banner visibility derived safely
      showMarketProBanner:
        nativeFlags?.paywallEnabled ?? defaultFlags.showMarketProBanner,
    };
  } catch (e) {
    return defaultFlags;
  }
}
