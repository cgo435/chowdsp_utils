#include <cmath>
#include "chowdsp_FloatVectorOperations.h"

namespace chowdsp
{
#if ! CHOWDSP_NO_XSIMD
#ifndef DOXYGEN
namespace fvo_detail
{
    template <typename T, typename Op>
    void unaryOpFallback (T* dest, const T* src, int numValues, Op&& op)
    {
        for (int i = 0; i < numValues; ++i)
            dest[i] = op (src[i]);
    }

    template <typename Arch, typename T, typename ScalarOp, typename VecOp, typename LoadOpType, typename StoreOpType>
    void unaryOp (T* dest, const T* src, int numValues, ScalarOp&& scalarOp, VecOp&& vecOp, LoadOpType&& loadOp, StoreOpType&& storeOp)
    {
        constexpr auto vecSize = (int) xsimd::batch<T, Arch>::size;
        auto numVecOps = numValues / vecSize;

        // Fallback: not enough operations to justify vectorizing!
        if (numVecOps < 2)
        {
            unaryOpFallback (dest, src, numValues, std::forward<ScalarOp> (scalarOp));
            return;
        }

        // Main loop here...
        while (--numVecOps >= 0)
        {
            storeOp (dest, vecOp (loadOp (src)));
            dest += vecSize;
            src += vecSize;
        }

        // leftover values that can't be vectorized...
        auto leftoverValues = numValues % vecSize;
        if (leftoverValues > 0)
            unaryOpFallback (dest, src, leftoverValues, std::forward<ScalarOp> (scalarOp));
    }

    template <typename Arch, typename T, typename ScalarOp, typename VecOp>
    void unaryOp (T* dest, const T* src, int numValues, ScalarOp&& scalarOp, VecOp&& vecOp)
    {
        auto loadA = [] (const auto* ptr)
        { return xsimd::load_aligned<Arch> (ptr); };

        auto loadU = [] (const auto* ptr)
        { return xsimd::load_unaligned<Arch> (ptr); };

        auto storeA = [] (auto* ptr, const auto& reg)
        { xsimd::store_aligned<Arch> (ptr, reg); };

        auto storeU = [] (auto* ptr, const auto& reg)
        { xsimd::store_unaligned<Arch> (ptr, reg); };

        if (SIMDUtils::isAligned<Arch> (dest))
        {
            if (SIMDUtils::isAligned<Arch> (src))
                unaryOp<Arch> (dest, src, numValues, std::forward<ScalarOp> (scalarOp), std::forward<VecOp> (vecOp), loadA, storeA);
            else
                unaryOp<Arch> (dest, src, numValues, std::forward<ScalarOp> (scalarOp), std::forward<VecOp> (vecOp), loadU, storeA);
        }
        else
        {
            if (SIMDUtils::isAligned<Arch> (src))
                unaryOp<Arch> (dest, src, numValues, std::forward<ScalarOp> (scalarOp), std::forward<VecOp> (vecOp), loadA, storeU);
            else
                unaryOp<Arch> (dest, src, numValues, std::forward<ScalarOp> (scalarOp), std::forward<VecOp> (vecOp), loadU, storeU);
        }
    }

    template <typename Arch, typename T, typename Op>
    void unaryOp (T* dest, const T* src, int numValues, Op&& op)
    {
        unaryOp<Arch> (dest, src, numValues, std::forward<Op> (op), std::forward<Op> (op));
    }

    template <typename T, typename Op>
    void binaryOpFallback (T* dest, const T* src1, const T* src2, int numValues, Op&& op)
    {
        for (int i = 0; i < numValues; ++i)
            dest[i] = op (src1[i], src2[i]);
    }

    template <typename Arch, typename T, typename ScalarOp, typename VecOp, typename LoadOp1Type, typename LoadOp2Type, typename StoreOpType>
    void binaryOp (T* dest, const T* src1, const T* src2, int numValues, ScalarOp&& scalarOp, VecOp&& vecOp, LoadOp1Type&& loadOp1, LoadOp2Type&& loadOp2, StoreOpType&& storeOp) // NOSONAR (too many parameters)
    {
        constexpr auto vecSize = (int) xsimd::batch<T, Arch>::size;
        auto numVecOps = numValues / vecSize;

        // Fallback: not enough operations to justify vectorizing!
        if (numVecOps < 2)
        {
            binaryOpFallback (dest, src1, src2, numValues, std::forward<ScalarOp> (scalarOp));
            return;
        }

        // Main loop here...
        while (--numVecOps >= 0)
        {
            storeOp (dest, vecOp (loadOp1 (src1), loadOp2 (src2)));
            dest += vecSize;
            src1 += vecSize;
            src2 += vecSize;
        }

        // leftover values that can't be vectorized...
        auto leftoverValues = numValues % vecSize;
        if (leftoverValues > 0)
            binaryOpFallback (dest, src1, src2, leftoverValues, std::forward<ScalarOp> (scalarOp));
    }

    template <typename Arch, typename T, typename ScalarOp, typename VecOp>
    void binaryOp (T* dest, const T* src1, const T* src2, int numValues, ScalarOp&& scalarOp, VecOp&& vecOp)
    {
        auto loadA = [] (const auto* ptr)
        { return xsimd::load_aligned<Arch> (ptr); };

        auto loadU = [] (const auto* ptr)
        { return xsimd::load_unaligned<Arch> (ptr); };

        auto storeA = [] (auto* ptr, const auto& reg)
        { xsimd::store_aligned<Arch> (ptr, reg); };

        auto storeU = [] (auto* ptr, const auto& reg)
        { xsimd::store_unaligned<Arch> (ptr, reg); };

        if (SIMDUtils::isAligned<Arch> (dest))
        {
            if (SIMDUtils::isAligned<Arch> (src1))
            {
                if (SIMDUtils::isAligned<Arch> (src2))
                    binaryOp<Arch> (dest, src1, src2, numValues, std::forward<ScalarOp> (scalarOp), std::forward<VecOp> (vecOp), loadA, loadA, storeA);
                else
                    binaryOp<Arch> (dest, src1, src2, numValues, std::forward<ScalarOp> (scalarOp), std::forward<VecOp> (vecOp), loadA, loadU, storeA);
            }
            else
            {
                if (SIMDUtils::isAligned<Arch> (src2))
                    binaryOp<Arch> (dest, src1, src2, numValues, std::forward<ScalarOp> (scalarOp), std::forward<VecOp> (vecOp), loadU, loadA, storeA);
                else
                    binaryOp<Arch> (dest, src1, src2, numValues, std::forward<ScalarOp> (scalarOp), std::forward<VecOp> (vecOp), loadU, loadU, storeA);
            }
        }
        else
        {
            if (SIMDUtils::isAligned<Arch> (src1))
            {
                if (SIMDUtils::isAligned<Arch> (src2))
                    binaryOp<Arch> (dest, src1, src2, numValues, std::forward<ScalarOp> (scalarOp), std::forward<VecOp> (vecOp), loadA, loadA, storeU);
                else
                    binaryOp<Arch> (dest, src1, src2, numValues, std::forward<ScalarOp> (scalarOp), std::forward<VecOp> (vecOp), loadA, loadU, storeU);
            }
            else
            {
                if (SIMDUtils::isAligned<Arch> (src2))
                    binaryOp<Arch> (dest, src1, src2, numValues, std::forward<ScalarOp> (scalarOp), std::forward<VecOp> (vecOp), loadU, loadA, storeU);
                else
                    binaryOp<Arch> (dest, src1, src2, numValues, std::forward<ScalarOp> (scalarOp), std::forward<VecOp> (vecOp), loadU, loadU, storeU);
            }
        }
    }

    template <typename Arch, typename T, typename Op>
    void binaryOp (T* dest, const T* src1, const T* src2, int numValues, Op&& op)
    {
        binaryOp<Arch> (dest, src1, src2, numValues, std::forward<Op> (op), std::forward<Op> (op));
    }

    template <typename T, typename Op>
    T reduceFallback (const T* src, int numValues, T init, Op&& op)
    {
        for (int i = 0; i < numValues; ++i)
            init = op (init, src[i]);

        return init;
    }

    template <typename T, typename Op>
    T reduceFallback (const T* src1, const T* src2, int numValues, T init, Op&& op)
    {
        for (int i = 0; i < numValues; ++i)
            init = op (init, src1[i], src2[i]);

        return init;
    }

    template <typename Arch, typename T, typename ScalarOp, typename VecOp, typename VecReduceOp>
    T reduce (const T* src, int numValues, T init, ScalarOp&& scalarOp, VecOp&& vecOp, VecReduceOp&& vecReduceOp)
    {
        constexpr auto vecSize = (int) xsimd::batch<T, Arch>::size;
        auto numVecOps = numValues / vecSize;

        // Fallback: not enough operations to justify vectorizing!
        if (numVecOps < 2)
            return reduceFallback (src, numValues, init, std::forward<ScalarOp> (scalarOp));

        // Fallback: starting pointer is not aligned!
        if (! SIMDUtils::isAligned<Arch> (src))
        {
            auto* nextAlignedPtr = SIMDUtils::getNextAlignedPtr<Arch> (src);
            auto diff = int (nextAlignedPtr - src);
            auto initResult = reduceFallback (src, diff, init, std::forward<ScalarOp> (scalarOp));
            return reduce<Arch> (nextAlignedPtr, numValues - diff, initResult, std::forward<ScalarOp> (scalarOp), std::forward<VecOp> (vecOp), std::forward<VecReduceOp> (vecReduceOp));
        }

        // Main loop here...
        T initData alignas (Arch::alignment())[(size_t) vecSize] {};
        initData[0] = init;
        auto resultVec = xsimd::load_aligned<Arch> (initData);
        while (--numVecOps >= 0)
        {
            resultVec = vecOp (resultVec, xsimd::load_aligned<Arch> (src));
            src += vecSize;
        }

        auto result = vecReduceOp (resultVec);

        // leftover values that can't be vectorized...
        auto leftoverValues = numValues % vecSize;
        if (leftoverValues > 0)
            result = reduceFallback (src, leftoverValues, result, std::forward<ScalarOp> (scalarOp));

        return result;
    }

    template <typename Arch, typename T, typename ScalarOp, typename VecOp>
    T reduce (const T* src, int numValues, T init, ScalarOp&& scalarOp, VecOp&& vecOp)
    {
        return reduce<Arch> (src, numValues, init, std::forward<ScalarOp> (scalarOp), std::forward<VecOp> (vecOp), [] (auto val)
                             { return xsimd::reduce_add (val); });
    }

    template <typename Arch, typename T, typename Op>
    T reduce (const T* src, int numValues, T init, Op&& op)
    {
        return reduce<Arch> (src, numValues, init, std::forward<Op> (op), std::forward<Op> (op));
    }

    template <typename Arch, typename T, typename ScalarOp, typename VecOp, typename VecReduceOp>
    T reduce (const T* src1, const T* src2, int numValues, T init, ScalarOp&& scalarOp, VecOp&& vecOp, VecReduceOp&& vecReduceOp)
    {
        constexpr auto vecSize = (int) xsimd::batch<T, Arch>::size;
        auto numVecOps = numValues / vecSize;

        // Fallback: not enough operations to justify vectorizing!
        if (numVecOps < 2)
            return reduceFallback (src1, src2, numValues, init, std::forward<ScalarOp> (scalarOp));

        // Main loop here:
        auto vecLoop = [&] (auto&& loadOp1, auto&& loadOp2)
        {
            xsimd::batch<T, Arch> resultVec {};
            while (--numVecOps >= 0)
            {
                resultVec = vecOp (resultVec, loadOp1 (src1), loadOp2 (src2));
                src1 += vecSize;
                src2 += vecSize;
            }

            return resultVec;
        };

        // define load operations
        auto loadA = [] (const T* val)
        { return xsimd::load_aligned<Arch> (val); };
        auto loadU = [] (const T* val)
        { return xsimd::load_unaligned<Arch> (val); };

        // select load operations based on data alignment
        const auto isSrc1Aligned = SIMDUtils::isAligned<Arch> (src1);
        const auto isSrc2Aligned = SIMDUtils::isAligned<Arch> (src2);
        T result {};
        if (isSrc1Aligned && isSrc2Aligned)
            result = vecReduceOp (vecLoop (loadA, loadA));
        else if (isSrc1Aligned)
            result = vecReduceOp (vecLoop (loadA, loadU));
        else if (isSrc2Aligned)
            result = vecReduceOp (vecLoop (loadU, loadA));
        else
            result = vecReduceOp (vecLoop (loadU, loadU));

        // leftover values that can't be vectorized...
        auto leftoverValues = numValues % vecSize;
        if (leftoverValues > 0)
            result = reduceFallback (src1, src2, leftoverValues, result, std::forward<ScalarOp> (scalarOp));

        return result;
    }

    template <typename Arch, typename T, typename ScalarOp, typename VecOp>
    T reduce (const T* src1, const T* src2, int numValues, T init, ScalarOp&& scalarOp, VecOp&& vecOp)
    {
        return reduce<Arch> (src1, src2, numValues, init, std::forward<ScalarOp> (scalarOp), std::forward<VecOp> (vecOp), [] (auto val)
                             { return xsimd::reduce_add (val); });
    }

    template <typename Arch, typename T, typename Op>
    T reduce (const T* src1, const T* src2, int numValues, T init, Op&& op)
    {
        return reduce<Arch> (src1, src2, numValues, init, std::forward<Op> (op), std::forward<Op> (op));
    }
} // namespace fvo_detail
#endif // DOXYGEN
#endif // ! CHOWDSP_NO_XSIMD

bool FloatVectorOperations::isUsingVDSP()
{
#if JUCE_USE_VDSP_FRAMEWORK
    return true;
#else
    return false;
#endif
}

bool FloatVectorOperations::isUsingAdvancedSIMDArch()
{
    return canUseAdvancedSIMDArch;
}

void FloatVectorOperations::setUsingAdvancedSIMDArch (bool canUse)
{
    canUseAdvancedSIMDArch = canUse;
}

template <typename Arch>
void FloatVectorOperations::divide (float* dest, const float* dividend, const float* divisor, int numValues) noexcept
{
#if JUCE_USE_VDSP_FRAMEWORK
    vDSP_vdiv (divisor, 1, dividend, 1, dest, 1, (vDSP_Length) numValues);
#elif CHOWDSP_NO_XSIMD
    std::transform (dividend, dividend + numValues, divisor, dest, [] (auto a, auto b)
                    { return a / b; });
#else
    fvo_detail::binaryOp<Arch> (dest,
                                dividend,
                                divisor,
                                numValues,
                                [] (auto num, auto den)
                                {
                                    return num / den;
                                });
#endif
}

template <>
void FloatVectorOperations::divide<void> (float* dest, const float* dividend, const float* divisor, int numValues) noexcept
{
    if (canUseAdvancedSIMDArch)
        divide<advancedSIMDArch> (dest, dividend, divisor, numValues);
    else
        divide<baseSIMDArch> (dest, dividend, divisor, numValues);
}

template <typename Arch>
void FloatVectorOperations::divide (double* dest, const double* dividend, const double* divisor, int numValues) noexcept
{
#if JUCE_USE_VDSP_FRAMEWORK
    vDSP_vdivD (divisor, 1, dividend, 1, dest, 1, (vDSP_Length) numValues);
#elif CHOWDSP_NO_XSIMD
    std::transform (dividend, dividend + numValues, divisor, dest, [] (auto a, auto b)
                    { return a / b; });
#else
    fvo_detail::binaryOp<Arch> (dest,
                                dividend,
                                divisor,
                                numValues,
                                [] (auto num, auto den)
                                {
                                    return num / den;
                                });
#endif
}

template <>
void FloatVectorOperations::divide<void> (double* dest, const double* dividend, const double* divisor, int numValues) noexcept
{
    if (canUseAdvancedSIMDArch)
        divide<advancedSIMDArch> (dest, dividend, divisor, numValues);
    else
        divide<baseSIMDArch> (dest, dividend, divisor, numValues);
}

template <typename Arch>
void FloatVectorOperations::divide (float* dest, float dividend, const float* divisor, int numValues) noexcept
{
#if JUCE_USE_VDSP_FRAMEWORK
    vDSP_svdiv (&dividend, divisor, 1, dest, 1, (vDSP_Length) numValues);
#elif CHOWDSP_NO_XSIMD
    std::transform (divisor, divisor + numValues, dest, [dividend] (auto x)
                    { return dividend / x; });
#else
    fvo_detail::unaryOp<Arch> (dest,
                               divisor,
                               numValues,
                               [dividend] (auto x)
                               {
                                   return dividend / x;
                               });
#endif
}

template <>
void FloatVectorOperations::divide<void> (float* dest, float dividend, const float* divisor, int numValues) noexcept
{
    if (canUseAdvancedSIMDArch)
        divide<advancedSIMDArch> (dest, dividend, divisor, numValues);
    else
        divide<baseSIMDArch> (dest, dividend, divisor, numValues);
}

template <typename Arch>
void FloatVectorOperations::divide (double* dest, double dividend, const double* divisor, int numValues) noexcept
{
#if JUCE_USE_VDSP_FRAMEWORK
    vDSP_svdivD (&dividend, divisor, 1, dest, 1, (vDSP_Length) numValues);
#elif CHOWDSP_NO_XSIMD
    std::transform (divisor, divisor + numValues, dest, [dividend] (auto x)
                    { return dividend / x; });
#else
    fvo_detail::unaryOp<Arch> (dest,
                               divisor,
                               numValues,
                               [dividend] (auto x)
                               {
                                   return dividend / x;
                               });
#endif
}

template <>
void FloatVectorOperations::divide<void> (double* dest, double dividend, const double* divisor, int numValues) noexcept
{
    if (canUseAdvancedSIMDArch)
        divide<advancedSIMDArch> (dest, dividend, divisor, numValues);
    else
        divide<baseSIMDArch> (dest, dividend, divisor, numValues);
}

template <typename Arch>
float FloatVectorOperations::accumulate (const float* src, int numValues) noexcept
{
#if JUCE_USE_VDSP_FRAMEWORK
    float result = 0.0f;
    vDSP_sve (src, 1, &result, (vDSP_Length) numValues);
    return result;
#elif CHOWDSP_NO_XSIMD
    return std::accumulate (src, src + numValues, 0.0f);
#else
    return fvo_detail::reduce<Arch> (
        src,
        numValues,
        0.0f,
        [] (auto prev, auto next)
        { return prev + next; });
#endif
}

template <>
float FloatVectorOperations::accumulate<void> (const float* src, int numValues) noexcept
{
    if (canUseAdvancedSIMDArch)
        return accumulate<advancedSIMDArch> (src, numValues);
    else
        return accumulate<baseSIMDArch> (src, numValues);
}

template <typename Arch>
double FloatVectorOperations::accumulate (const double* src, int numValues) noexcept
{
#if JUCE_USE_VDSP_FRAMEWORK
    double result = 0.0;
    vDSP_sveD (src, 1, &result, (vDSP_Length) numValues);
    return result;
#elif CHOWDSP_NO_XSIMD
    return std::accumulate (src, src + numValues, 0.0);
#else
    return fvo_detail::reduce<Arch> (
        src,
        numValues,
        0.0,
        [] (auto prev, auto next)
        { return prev + next; });
#endif
}

template <>
double FloatVectorOperations::accumulate<void> (const double* src, int numValues) noexcept
{
    if (canUseAdvancedSIMDArch)
        return accumulate<advancedSIMDArch> (src, numValues);
    else
        return accumulate<baseSIMDArch> (src, numValues);
}

template <typename Arch>
float FloatVectorOperations::innerProduct (const float* src1, const float* src2, int numValues) noexcept
{
#if JUCE_USE_VDSP_FRAMEWORK
    float result = 0.0f;
    vDSP_dotpr (src1, 1, src2, 1, &result, (vDSP_Length) numValues);
    return result;
#elif CHOWDSP_NO_XSIMD
    return std::inner_product (src1, src1 + numValues, src2, 0.0f);
#else
    return fvo_detail::reduce<Arch> (
        src1,
        src2,
        numValues,
        0.0f,
        [] (auto prev, auto next1, auto next2)
        { return prev + next1 * next2; });
#endif
}

template <>
float FloatVectorOperations::innerProduct<void> (const float* src1, const float* src2, int numValues) noexcept
{
    if (canUseAdvancedSIMDArch)
        return innerProduct<advancedSIMDArch> (src1, src2, numValues);
    else
        return innerProduct<baseSIMDArch> (src1, src2, numValues);
}

template <typename Arch>
double FloatVectorOperations::innerProduct (const double* src1, const double* src2, int numValues) noexcept
{
#if JUCE_USE_VDSP_FRAMEWORK
    double result = 0.0;
    vDSP_dotprD (src1, 1, src2, 1, &result, (vDSP_Length) numValues);
    return result;
#elif CHOWDSP_NO_XSIMD
    return std::inner_product (src1, src1 + numValues, src2, 0.0);
#else
    return fvo_detail::reduce<Arch> (
        src1,
        src2,
        numValues,
        0.0,
        [] (auto prev, auto next1, auto next2)
        { return prev + next1 * next2; });
#endif
}

template <>
double FloatVectorOperations::innerProduct<void> (const double* src1, const double* src2, int numValues) noexcept
{
    if (canUseAdvancedSIMDArch)
        return innerProduct<advancedSIMDArch> (src1, src2, numValues);
    else
        return innerProduct<baseSIMDArch> (src1, src2, numValues);
}

template <typename Arch>
float FloatVectorOperations::findAbsoluteMaximum (const float* src, int numValues) noexcept
{
#if JUCE_USE_VDSP_FRAMEWORK
    float result = 0.0f;
    vDSP_maxmgv (src, 1, &result, (vDSP_Length) numValues);
    return result;
#elif CHOWDSP_NO_XSIMD
    return [] (const auto& begin, const auto end) -> float
    { return std::abs (*std::max_element (begin, end, [] (auto a, auto b)
                                          { return std::abs (a) < std::abs (b); })); }(src, src + numValues);
#else
    return fvo_detail::reduce<Arch> (
        src,
        numValues,
        0.0f,
        [] (auto a, auto b)
        { return juce::jmax (std::abs (a), std::abs (b)); },
        [] (auto a, auto b)
        { return xsimd::max (xsimd::abs (a), xsimd::abs (b)); },
        [] (auto x)
        { return xsimd::reduce_max (x); });
#endif
}

template <>
float FloatVectorOperations::findAbsoluteMaximum<void> (const float* src, int numValues) noexcept
{
    if (canUseAdvancedSIMDArch)
        return findAbsoluteMaximum<advancedSIMDArch> (src, numValues);
    else
        return findAbsoluteMaximum<baseSIMDArch> (src, numValues);
}

template <typename Arch>
double FloatVectorOperations::findAbsoluteMaximum (const double* src, int numValues) noexcept
{
#if JUCE_USE_VDSP_FRAMEWORK
    double result = 0.0;
    vDSP_maxmgvD (src, 1, &result, (vDSP_Length) numValues);
    return result;
#elif CHOWDSP_NO_XSIMD
    return [] (const auto& begin, const auto end) -> double
    { return std::abs (*std::max_element (begin, end, [] (auto a, auto b)
                                          { return std::abs (a) < std::abs (b); })); }(src, src + numValues);
#else
    return fvo_detail::reduce<Arch> (
        src,
        numValues,
        0.0,
        [] (auto a, auto b)
        { return juce::jmax (a, std::abs (b)); },
        [] (auto a, auto b)
        { return xsimd::max (a, xsimd::abs (b)); },
        [] (auto x)
        { return xsimd::reduce_max (x); });
#endif
}

template <>
double FloatVectorOperations::findAbsoluteMaximum<void> (const double* src, int numValues) noexcept
{
    if (canUseAdvancedSIMDArch)
        return findAbsoluteMaximum<advancedSIMDArch> (src, numValues);
    else
        return findAbsoluteMaximum<baseSIMDArch> (src, numValues);
}

#if ! CHOWDSP_NO_XSIMD
template <typename Arch, typename T>
void integerPowerT (T* dest, const T* src, int exponent, int numValues) noexcept
{
    // negative values are not supported!
    jassert (exponent >= 0);

    using Power::ipow;
    switch (exponent)
    {
        case 0:
            juce::FloatVectorOperations::fill (dest, (T) 1, numValues);
            break;
        case 1:
            juce::FloatVectorOperations::copy (dest, src, numValues);
            break;
        case 2:
            juce::FloatVectorOperations::multiply (dest, src, src, numValues);
            break;
        case 3:
            fvo_detail::unaryOp<Arch> (dest, src, numValues, [] (auto x)
                                       { return ipow<3> (x); });
            break;
        case 4:
            fvo_detail::unaryOp<Arch> (dest, src, numValues, [] (auto x)
                                       { return ipow<4> (x); });
            break;
        case 5:
            fvo_detail::unaryOp<Arch> (dest, src, numValues, [] (auto x)
                                       { return ipow<5> (x); });
            break;
        case 6:
            fvo_detail::unaryOp<Arch> (dest, src, numValues, [] (auto x)
                                       { return ipow<6> (x); });
            break;
        case 7:
            fvo_detail::unaryOp<Arch> (dest, src, numValues, [] (auto x)
                                       { return ipow<7> (x); });
            break;
        case 8:
            fvo_detail::unaryOp<Arch> (dest, src, numValues, [] (auto x)
                                       { return ipow<8> (x); });
            break;
        case 9:
            fvo_detail::unaryOp<Arch> (dest, src, numValues, [] (auto x)
                                       { return ipow<9> (x); });
            break;
        case 10:
            fvo_detail::unaryOp<Arch> (dest, src, numValues, [] (auto x)
                                       { return ipow<10> (x); });
            break;
        case 11:
            fvo_detail::unaryOp<Arch> (dest, src, numValues, [] (auto x)
                                       { return ipow<11> (x); });
            break;
        case 12:
            fvo_detail::unaryOp<Arch> (dest, src, numValues, [] (auto x)
                                       { return ipow<12> (x); });
            break;
        case 13:
            fvo_detail::unaryOp<Arch> (dest, src, numValues, [] (auto x)
                                       { return ipow<13> (x); });
            break;
        case 14:
            fvo_detail::unaryOp<Arch> (dest, src, numValues, [] (auto x)
                                       { return ipow<14> (x); });
            break;
        case 15:
            fvo_detail::unaryOp<Arch> (dest, src, numValues, [] (auto x)
                                       { return ipow<15> (x); });
            break;
        case 16:
            fvo_detail::unaryOp<Arch> (dest, src, numValues, [] (auto x)
                                       { return ipow<16> (x); });
            break;
        default:
            // this method will not be as fast for values outside the range [0, 16]
            fvo_detail::unaryOp<Arch> (
                dest,
                src,
                numValues,
                [exponent] (auto x)
                { return std::pow (x, (T) exponent); },
                [exponent] (auto x)
                { return xsimd::pow (x, xsimd::batch<T> ((T) exponent)); });
            break;
    }
}
#endif // ! CHOWDSP_NO_XSIMD

template <typename Arch>
void FloatVectorOperations::integerPower (float* dest, const float* src, int exponent, int numValues) noexcept
{
#if CHOWDSP_NO_XSIMD
    for (int i = 0; i < numValues; ++i)
        dest[i] = std::pow (src[i], (float) exponent);
#else
    integerPowerT<Arch> (dest, src, exponent, numValues);
#endif
}

template <>
void FloatVectorOperations::integerPower<void> (float* dest, const float* src, int exponent, int numValues) noexcept
{
    if (canUseAdvancedSIMDArch)
        integerPower<advancedSIMDArch> (dest, src, exponent, numValues);
    else
        integerPower<baseSIMDArch> (dest, src, exponent, numValues);
}

template <typename Arch>
void FloatVectorOperations::integerPower (double* dest, const double* src, int exponent, int numValues) noexcept
{
#if CHOWDSP_NO_XSIMD
    for (int i = 0; i < numValues; ++i)
        dest[i] = std::pow (src[i], (double) exponent);
#else
    integerPowerT<Arch> (dest, src, exponent, numValues);
#endif
}

template <>
void FloatVectorOperations::integerPower<void> (double* dest, const double* src, int exponent, int numValues) noexcept
{
    if (canUseAdvancedSIMDArch)
        integerPower<advancedSIMDArch> (dest, src, exponent, numValues);
    else
        integerPower<baseSIMDArch> (dest, src, exponent, numValues);
}

template <typename Arch>
float FloatVectorOperations::computeRMS (const float* src, int numValues) noexcept
{
#if JUCE_USE_VDSP_FRAMEWORK
    float result = 0.0f;
    vDSP_rmsqv (src, 1, &result, (vDSP_Length) numValues);
    return result;
#elif CHOWDSP_NO_XSIMD
    return [] (const float* data, int numSamples) -> float
    {
        auto squareSum = 0.0;
        for (int i = 0; i < numSamples; ++i)
            squareSum += data[i] * data[i];
        return std::sqrt (squareSum / (float) numSamples);
    }(src, numValues);
#else
    const auto squareSum = fvo_detail::reduce<Arch> (src,
                                                     numValues,
                                                     0.0f,
                                                     [] (auto prev, auto next)
                                                     { return prev + next * next; });
    return std::sqrt (squareSum / (float) numValues);
#endif
}

template <>
float FloatVectorOperations::computeRMS<void> (const float* src, int numValues) noexcept
{
    if (canUseAdvancedSIMDArch)
        return computeRMS<advancedSIMDArch> (src, numValues);
    else
        return computeRMS<baseSIMDArch> (src, numValues);
}

template <typename Arch>
double FloatVectorOperations::computeRMS (const double* src, int numValues) noexcept
{
#if JUCE_USE_VDSP_FRAMEWORK
    double result = 0.0;
    vDSP_rmsqvD (src, 1, &result, (vDSP_Length) numValues);
    return result;
#elif CHOWDSP_NO_XSIMD
    return [] (const double* data, int numSamples) -> double
    {
        auto squareSum = 0.0;
        for (int i = 0; i < numSamples; ++i)
            squareSum += data[i] * data[i];
        return std::sqrt (squareSum / (double) numSamples);
    }(src, numValues);
#else
    const auto squareSum = fvo_detail::reduce<Arch> (src,
                                                     numValues,
                                                     0.0,
                                                     [] (auto prev, auto next)
                                                     { return prev + next * next; });
    return std::sqrt (squareSum / (double) numValues);
#endif
}

template <>
double FloatVectorOperations::computeRMS<void> (const double* src, int numValues) noexcept
{
    if (canUseAdvancedSIMDArch)
        return computeRMS<advancedSIMDArch> (src, numValues);
    else
        return computeRMS<baseSIMDArch> (src, numValues);
}

int FloatVectorOperations::countNaNs (const float* src, int numValues) noexcept
{
    return [] (const float* data, int numSamples) -> int
    {
        int nanCount = 0;
        for (int i = 0; i < numSamples; ++i)
            nanCount += (int) std::isnan (data[i]);
        return nanCount;
    }(src, numValues);
}

int FloatVectorOperations::countNaNs (const double* src, int numValues) noexcept
{
    return [] (const double* data, int numSamples) -> int
    {
        int nanCount = 0;
        for (int i = 0; i < numSamples; ++i)
            nanCount += (int) std::isnan (data[i]);
        return nanCount;
    }(src, numValues);
}

int FloatVectorOperations::countInfs (const float* src, int numValues) noexcept
{
    return [] (const float* data, int numSamples) -> int
    {
        int nanCount = 0;
        for (int i = 0; i < numSamples; ++i)
            nanCount += (int) std::isinf (data[i]);
        return nanCount;
    }(src, numValues);
}

int FloatVectorOperations::countInfs (const double* src, int numValues) noexcept
{
    return [] (const double* data, int numSamples) -> int
    {
        int nanCount = 0;
        for (int i = 0; i < numSamples; ++i)
            nanCount += (int) std::isinf (data[i]);
        return nanCount;
    }(src, numValues);
}

template void FloatVectorOperations::divide<baseSIMDArch> (float*, const float*, const float*, int);
template void FloatVectorOperations::divide<baseSIMDArch> (double*, const double*, const double*, int);
template void FloatVectorOperations::divide<baseSIMDArch> (float*, float, const float*, int);
template void FloatVectorOperations::divide<baseSIMDArch> (double*, double, const double*, int);
template float FloatVectorOperations::accumulate<baseSIMDArch> (const float*, int);
template double FloatVectorOperations::accumulate<baseSIMDArch> (const double*, int);
template float FloatVectorOperations::innerProduct<baseSIMDArch> (const float*, const float*, int) noexcept;
template double FloatVectorOperations::innerProduct<baseSIMDArch> (const double*, const double*, int) noexcept;
template float FloatVectorOperations::findAbsoluteMaximum<baseSIMDArch> (const float*, int);
template double FloatVectorOperations::findAbsoluteMaximum<baseSIMDArch> (const double*, int);
#if JUCE_INTEL && XSIMD_WITH_AVX
template void FloatVectorOperations::divide<advancedSIMDArch> (float*, const float*, const float*, int);
template void FloatVectorOperations::divide<advancedSIMDArch> (double*, const double*, const double*, int);
template void FloatVectorOperations::divide<advancedSIMDArch> (float*, float, const float*, int);
template void FloatVectorOperations::divide<advancedSIMDArch> (double*, double, const double*, int);
template float FloatVectorOperations::accumulate<advancedSIMDArch> (const float*, int);
template double FloatVectorOperations::accumulate<advancedSIMDArch> (const double*, int);
template float FloatVectorOperations::innerProduct<advancedSIMDArch> (const float*, const float*, int) noexcept;
template double FloatVectorOperations::innerProduct<advancedSIMDArch> (const double*, const double*, int) noexcept;
template float FloatVectorOperations::findAbsoluteMaximum<advancedSIMDArch> (const float*, int);
template double FloatVectorOperations::findAbsoluteMaximum<advancedSIMDArch> (const double*, int);
#endif
} // namespace chowdsp
