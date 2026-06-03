package com.fungen.app

/**
 * Static feature flags.
 * Safe, crash-free, and compatible with existing code.
 */
object FeatureFlags {

    @JvmField var PAYWALL_ENABLED = false
    @JvmField var PREMIUM_ENABLED = true
    @JvmField var MAX_FREE_USES = 5
    @JvmField var SHOW_UPGRADE_BANNER = false

    // New flag used by MarketScreen
    @JvmField var SHOW_MARKET_PRO_BANNER = true
}
