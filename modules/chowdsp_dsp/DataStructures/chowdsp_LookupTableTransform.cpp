#include "chowdsp_LookupTableTransform.h"

namespace chowdsp
{
/** Helpful methods for working with atomic data types */
namespace AtomicHelpers
{
    /**
     * Compares an atomic variable to some compareVal, and sets the atomic
     * variable to a new value if the comparison returns true.
     */
    template <typename T, typename atomic_type = std::atomic<T>>
    bool compareExchange (atomic_type& atomic_val, T compareVal, T valToSetIfTrue) noexcept
    {
        return atomic_val.compare_exchange_strong (compareVal, valToSetIfTrue);
    }

    /** Simplified version of compareExchange for boolean types */
    template <typename T = bool, typename atomic_type = std::atomic<T>>
    bool compareNegate (atomic_type& atomic_val, T compareVal = (T) true) noexcept
    {
        return compareExchange (atomic_val, compareVal, ! compareVal);
    }
} // namespace AtomicHelpers

template <typename FloatType>
void LookupTableTransform<FloatType>::initialise (const std::function<FloatType (FloatType)>& functionToApproximate,
                                                  FloatType minInputValueToUse,
                                                  FloatType maxInputValueToUse,
                                                  size_t numPoints)
{
    jassert (maxInputValueToUse > minInputValueToUse);

    isInitialised.store (true);

    minInputValue = minInputValueToUse;
    maxInputValue = maxInputValueToUse;
    scaler = FloatType (numPoints - 1) / (maxInputValueToUse - minInputValueToUse);
    offset = -minInputValueToUse * scaler;

    const auto initFn = [functionToApproximate, minInputValueToUse, maxInputValueToUse, numPoints] (size_t i)
    {
        return functionToApproximate (
            juce::jlimit (
                minInputValueToUse, maxInputValueToUse, juce::jmap (FloatType (i), FloatType (0), FloatType (numPoints - 1), minInputValueToUse, maxInputValueToUse)));
    };

    lookupTable.initialise (initFn, numPoints);
}

template <typename FloatType>
bool LookupTableTransform<FloatType>::initialiseIfNotAlreadyInitialised() noexcept
{
    return AtomicHelpers::compareNegate (isInitialised, false);
}

//==============================================================================
template class LookupTableTransform<float>;
template class LookupTableTransform<double>;
} // namespace chowdsp
