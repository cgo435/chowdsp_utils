#include "ModalReverbPlugin.h"

namespace
{
const auto bestArch = xsimd::available_architectures().best;
const auto canUseAVX = xsimd::avx2::version() <= bestArch;
} // namespace

ModalReverbPlugin::ModalReverbPlugin()
{
    modalFilterBank.setModeAmplitudes (ModeParams::amps_real, ModeParams::amps_imag);
#if XSIMD_WITH_AVX
    modalFilterBankAVX.setModeAmplitudes (ModeParams::amps_real, ModeParams::amps_imag);
#endif
}

void ModalReverbPlugin::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    modalFilterBank.prepare (sampleRate, samplesPerBlock);
#if XSIMD_WITH_AVX
    modalFilterBankAVX.prepare (sampleRate, samplesPerBlock);
#endif

    modSine.prepare ({ sampleRate, (juce::uint32) samplesPerBlock, 1 });
    modBuffer.setMaxSize (1, samplesPerBlock);

    mixer.prepare ({ sampleRate, (juce::uint32) samplesPerBlock, (juce::uint32) getTotalNumInputChannels() });
    mixer.setMixingRule (juce::dsp::DryWetMixingRule::linear);
}

void ModalReverbPlugin::processAudioBlock (juce::AudioBuffer<float>& buffer)
{
    const auto numChannels = buffer.getNumChannels();
    const auto numSamples = buffer.getNumSamples();
    const auto& params = state.params;

    // push dry data int mixer
    mixer.setWetMixProportion (params.mixParam->getCurrentValue());
    mixer.pushDrySamples (juce::dsp::AudioBlock<float> { buffer });

    // make buffer mono
    for (int ch = 1; ch < numChannels; ++ch)
        buffer.addFrom (0, 0, buffer, ch, 0, numSamples);
    buffer.applyGain (1.0f / (float) numChannels);

    // set up modulation data
    modBuffer.setCurrentSize (1, numSamples);
    modBuffer.clear();
    modSine.setFrequency (params.modFreqParam->getCurrentValue());
    modSine.processBlock (modBuffer);

    const auto processFilterBank = [this, &params, &buffer, numChannels] (auto& filterBank)
    {
        // update mode frequencies
        const auto freqMult = std::pow (2.0f, params.pitchParam->getCurrentValue());
        filterBank.setModeFrequencies (ModeParams::freqs, freqMult);

        // update mode decay rates
        const auto decayFactor = std::pow (4.0f, 2.0f * params.decayParam->getCurrentValue() - 1.0f);
        filterBank.setModeDecays (ModeParams::taus, ModeParams::analysisFs, decayFactor);

        const auto numModesToMod = (int) params.modModesParam->getCurrentValue();
        if (numModesToMod == 0)
        {
            // process modal filter without modulation
            filterBank.process (buffer);
        }
        else
        {
            // process modal filter with modulation
            const auto modDepth = params.modDepthParam->getCurrentValue();
            auto&& block = juce::dsp::AudioBlock<float> { buffer };
            const auto* modData = modBuffer.getReadPointer (0);
            filterBank.processWithModulation (
                block,
                [modData, numModesToMod, freqMult, modDepth] (auto& mode, size_t vecModeIndex, int sampleIndex)
                {
                    using Vec = typename std::decay_t<decltype (filterBank)>::Vec;
                    using Arch = typename Vec::arch_type;
                    const auto modeScalarIndex = vecModeIndex * Vec::size;
                    if (modeScalarIndex >= (size_t) numModesToMod)
                        return;

                    auto modOffset = Vec (modData[sampleIndex]);
                    if (modeScalarIndex + Vec::size > (size_t) numModesToMod)
                    {
                        const auto numLeftoverModes = (size_t) numModesToMod - modeScalarIndex;
                        alignas (Arch::alignment()) float modVec[Vec::size] {};
                        std::fill (modVec, modVec + numLeftoverModes, modData[sampleIndex]);
                        modOffset = xsimd::load_aligned<Arch> (modVec);
                    }

                    const auto freqOffset = xsimd::pow (Vec (2.0f), modOffset) * modDepth;
                    mode.setFreq (xsimd::load_aligned<Arch> (ModeParams::freqs + vecModeIndex * Vec::size) * freqMult * freqOffset);
                });
        }

        // split out of mono
        const auto& renderBuffer = filterBank.getRenderBuffer();
        for (int ch = 0; ch < numChannels; ++ch)
            chowdsp::BufferMath::copyBufferChannels (renderBuffer, buffer, 0, ch);
        buffer.applyGain (juce::Decibels::decibelsToGain (-120.0f, -1000.0f));
    };

#if XSIMD_WITH_AVX
    if (canUseAVX && params.useAVXParam->get())
        processFilterBank (modalFilterBankAVX);
    else
#endif
        processFilterBank (modalFilterBank);

    // mix dry/wet buffer
    mixer.mixWetSamples (juce::dsp::AudioBlock<float> { buffer });
}

juce::AudioProcessorEditor* ModalReverbPlugin::createEditor()
{
    return new juce::GenericAudioProcessorEditor (*this);
}

// This creates new instances of the plugin
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new ModalReverbPlugin();
}
