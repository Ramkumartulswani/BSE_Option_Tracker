package com.fungen.app

import com.facebook.react.bridge.*

class FeatureFlagModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    override fun getName() = "FeatureFlags"

    @ReactMethod
    fun getFlags(promise: Promise) {
        val map = Arguments.createMap()

        map.putBoolean("paywallEnabled", FeatureFlags.PAYWALL_ENABLED)
        map.putBoolean("premiumEnabled", FeatureFlags.PREMIUM_ENABLED)
        map.putInt("maxFreeUses", FeatureFlags.MAX_FREE_USES)
        map.putBoolean("showUpgradeBanner", FeatureFlags.SHOW_UPGRADE_BANNER)
        map.putBoolean("showMarketProBanner", FeatureFlags.SHOW_MARKET_PRO_BANNER)

        promise.resolve(map)
    }
}
