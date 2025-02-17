#pragma once

namespace chowdsp
{
/** Parameter information needed for forwarding */
struct ParameterForwardingInfo
{
    juce::RangedAudioParameter* param = nullptr;
    juce::String name {};
};

/**
 * Useful class for managing groups of ForwardingParameter
 *
 * @tparam Provider                     Template provider type, which must provide static
 *                                      `juce::String getForwardingParameterID()` method.
 * @tparam totalNumForwardingParameters The total number of parameters to be created and managed by this class.
 */
template <typename Provider, int totalNumForwardingParameters>
class ForwardingParametersManager
{
public:
    /** Initializes handles to the forwarding parameters, and connects them to the given processor */
    explicit ForwardingParametersManager (juce::AudioProcessorValueTreeState& vts) : ForwardingParametersManager (vts.processor)
    {
    }

    /** Initializes handles to the forwarding parameters, and connects them to the given processor */
    explicit ForwardingParametersManager (juce::AudioProcessor& audioProcessor) : processor (audioProcessor)
    {
        for (int i = 0; i < totalNumForwardingParameters; ++i)
        {
            auto id = Provider::getForwardingParameterID (i);
            auto forwardedParam = std::make_unique<ForwardingParameter> (id, nullptr, "Blank");

            forwardedParam->setProcessor (&processor);
            forwardedParams[(size_t) i] = forwardedParam.get();
            processor.addParameter (forwardedParam.release());
        }
    }

    ~ForwardingParametersManager()
    {
        clearParameterRange (0, (int) forwardedParams.size());
    }

    /** Returns a flat array of the forwarded parameters */
    [[maybe_unused]] auto& getForwardedParameters() { return forwardedParams; }

    /** Returns a flat array of the forwarded parameters */
    [[maybe_unused]] const auto& getForwardedParameters() const { return forwardedParams; }

    /**
     * Use this type to force the host notification changes to be deferred until the object is destroyed.
     *
     * Note that the host _will_ be notified of a parameter change, when this object is destroyed,
     * regardless of whether or not any forwarding parameters were actually changed in the meantime.
     */
    struct ScopedForceDeferHostNotifications
    {
        explicit ScopedForceDeferHostNotifications (ForwardingParametersManager& manager)
            : mgr (manager),
              previousForceValue (mgr.forceDeferHostNotifications)
        {
            mgr.forceDeferHostNotifications = true;
        }

        ~ScopedForceDeferHostNotifications()
        {
            mgr.forceDeferHostNotifications = previousForceValue;
            if (! mgr.forceDeferHostNotifications)
                ForwardingParameter::reportParameterInfoChange (&mgr.processor);
        }

    private:
        ForwardingParametersManager& mgr;
        const bool previousForceValue;

        JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ScopedForceDeferHostNotifications)
    };

    /**
     * Sets a range of parameters.
     *
     * @param startIndex            The start of the range to set.
     * @param endIndex              The end (exclusive) of the range to set.
     * @param paramInfoProvider     A lambda which accepts the parameter index, and returns the corresponding ParameterForwardingInfo.
     * @param deferHostNotification If this is true, `updateHostDisplay()` will be deferred until all the parameter have been updated.
     */
    template <typename ParamInfoProvider>
    void setParameterRange (int startIndex,
                            int endIndex,
                            const ParamInfoProvider& paramInfoProvider,
                            bool deferHostNotification = true)
    {
        for (int i = startIndex; i < endIndex; ++i)
        {
            auto [param, paramName] = paramInfoProvider (i);
            forwardedParams[(size_t) i]->setParam (param, paramName, deferHostNotification || forceDeferHostNotifications);
        }

        if (deferHostNotification && ! forceDeferHostNotifications)
            ForwardingParameter::reportParameterInfoChange (&processor);
    }

    /**
     * Clears a range of parameters to nullptr.
     *
     * @param startIndex            The start of the range to set.
     * @param endIndex              The end (exclusive) of the range to set.
     * @param deferHostNotification If this is true, `updateHostDisplay()` will be deferred until all the parameter have been updated.
     */
    void clearParameterRange (int startIndex, int endIndex, bool deferHostNotification = true)
    {
        setParameterRange (
            startIndex,
            endIndex,
            [] (int) -> ParameterForwardingInfo
            { return {}; },
            deferHostNotification);
    }

protected:
    std::array<ForwardingParameter*, (size_t) totalNumForwardingParameters> forwardedParams;

private:
    juce::AudioProcessor& processor;

    bool forceDeferHostNotifications = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ForwardingParametersManager)
};
} // namespace chowdsp
