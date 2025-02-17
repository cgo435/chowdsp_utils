#pragma once

namespace chowdsp
{
/**
 * Utility class for *smoothly* bypassing a processor,
 * including latency compensation.
 * 
 * BypassProcessor should be used as follows:
 * ```
 * BypassProcessor bypass;
 * std::atomic<float>* onOffParam;
 * 
 * void processBlock (AudioBuffer<float> buffer)
 * {
 *     if (! bypass.processBlockIn (buffer, bypass.toBool (onOffParam)))
 *         return;
 * 
 *     // do my processing here....
 *   
 *     bypass.processBlockOut (buffer, bypass.toBool (onOffParam));
 * }
 * ```
 */
template <typename SampleType, typename DelayInterpType = std::nullptr_t, typename = void>
class BypassProcessor;

template <typename SampleType, typename DelayInterpType>
class BypassProcessor<SampleType, DelayInterpType, std::enable_if_t<std::is_same_v<DelayInterpType, std::nullptr_t>>>
{
public:
    using NumericType = SampleTypeHelpers::NumericType<SampleType>;

    BypassProcessor() = default;

    /** Converts a parameter handle to a boolean */
    static bool toBool (const std::atomic<float>* param)
    {
        return static_cast<bool> (param->load());
    }

    /** Allocated required memory, and resets the property */
    void prepare (const juce::dsp::ProcessSpec& spec, bool onOffParam);

    /**
      * Call this at the start of your processBlock().
      * If it returns false, you can safely skip all other
      * processing.
      */
    bool processBlockIn (const BufferView<SampleType>& buffer, bool onOffParam);

    /**
      * Call this at the end of your processBlock().
      * It will fade the dry signal back in with the main
      * signal as needed.
      */
    void processBlockOut (const BufferView<SampleType>& buffer, bool onOffParam);

private:
    bool prevOnOffParam = false;
    Buffer<SampleType> fadeBuffer;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (BypassProcessor)
};

template <typename SampleType, typename DelayInterpType>
class BypassProcessor<SampleType, DelayInterpType, std::enable_if_t<! std::is_same_v<DelayInterpType, std::nullptr_t>>>
{
public:
    using NumericType = SampleTypeHelpers::NumericType<SampleType>;

    BypassProcessor() = default;

    /** Converts a parameter handle to a boolean */
    static bool toBool (const std::atomic<float>* param)
    {
        return static_cast<bool> (param->load());
    }

    /** Allocated required memory, and resets the property */
    void prepare (const juce::dsp::ProcessSpec& spec, bool onOffParam);

    /**
     * If the non-bypassed processing has some associated
     * latency, it is recommended to report the latency
     * time here, so that the bypass processing will be
     * correctly time-aligned.
     */
    void setLatencySamples (int delaySamples);

    /**
     * If the non-bypassed processing has some associated
     * latency, it is recommended to report the latency
     * time here, so that the bypass processing will be
     * correctly time-aligned.
     */
    void setLatencySamples (NumericType delaySamples);

    /**
      * Call this at the start of your processBlock().
      * If it returns false, you can safely skip all other
      * processing.
      */
    bool processBlockIn (const BufferView<SampleType>& buffer, bool onOffParam);

    /**
      * Call this at the end of your processBlock().
      * It will fade the dry signal back in with the main
      * signal as needed.
      */
    void processBlockOut (const BufferView<SampleType>& buffer, bool onOffParam);

private:
    void setLatencySamplesInternal (NumericType delaySamples);
    int getFadeStartSample (const int numSamples);

    bool prevOnOffParam = false;
    Buffer<SampleType> fadeBuffer;

    DelayLine<SampleType, DelayInterpType> compDelay { 1 << 18 }; // max latency = 2^18 = 262144 samples
    NumericType prevDelay {};
    int latencySampleCount = -1;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (BypassProcessor)
};

} // namespace chowdsp

#include "chowdsp_BypassProcessor.cpp"
