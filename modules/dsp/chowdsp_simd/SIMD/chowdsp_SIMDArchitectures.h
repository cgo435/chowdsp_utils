#pragma once

namespace chowdsp
{
#if CHOWDSP_NO_XSIMD
struct DummySIMDArchitecture
{
    static constexpr bool supported() noexcept { return true; }
    static constexpr bool available() noexcept { return true; }
    static constexpr unsigned version() noexcept { return 0; }
    static constexpr std::size_t alignment() noexcept { return 16; }
    static constexpr bool requires_alignment() noexcept { return true; }
    static constexpr char const* name() noexcept { return "dummy"; }
};

using baseSIMDArch = DummySIMDArchitecture;
using advancedSIMDArch = DummySIMDArchitecture;

#else
#if JUCE_INTEL
/** Base SIMD architecture for this platform. */
using baseSIMDArch = xsimd::sse2;

#if XSIMD_WITH_AVX
/** Advanced SIMD architecture for this platform. */
using advancedSIMDArch = xsimd::avx;
#else
/** Advanced SIMD architecture for this platform. */
using advancedSIMDArch = xsimd::sse2;
#endif

#elif JUCE_ARM
/** Base SIMD architecture for this platform. */
using baseSIMDArch = xsimd::neon64;

/** Advanced SIMD architecture for this platform. */
using advancedSIMDArch = xsimd::neon64;
#endif

/** Default SIMD architecture. */
using defaultSIMDArch = xsimd::default_arch;
#endif
}
