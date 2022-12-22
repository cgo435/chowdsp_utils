#pragma once

#include <chowdsp_plugin_base/chowdsp_plugin_base.h>
#include <chowdsp_plugin_state/chowdsp_plugin_state.h>
#include <chowdsp_modal_dsp/chowdsp_modal_dsp.h>
#include <chowdsp_sources/chowdsp_sources.h>

#include "ModeParams.h"

struct Parameters : chowdsp::ParamHolder
{
    Parameters()
    {
        add (pitchParam,
             decayParam,
             mixParam,
             modModesParam,
             modFreqParam,
             modDepthParam,
             useAVXParam);
    }

    chowdsp::PercentParameter::Ptr pitchParam { juce::ParameterID { "pitch", 100 },
                                                "Pitch",
                                                0.0f,
                                                true };

    chowdsp::PercentParameter::Ptr decayParam { juce::ParameterID { "decay", 100 },
                                                "Decay" };

    chowdsp::PercentParameter::Ptr mixParam { juce::ParameterID { "mix", 100 },
                                              "Mix" };

    chowdsp::FloatParameter::Ptr modModesParam { juce::ParameterID { "mod_modes", 100 },
                                                 "Mod. Modes",
                                                 chowdsp::ParamUtils::createNormalisableRange (0.0f, 200.0f, 20.0f),
                                                 0.0f,
                                                 [] (float x)
                                                 { return juce::String ((int) x); },
                                                 &chowdsp::ParamUtils::stringToFloatVal };

    chowdsp::FreqHzParameter::Ptr modFreqParam { juce::ParameterID { "mod_freq", 100 },
                                                 "Mod. Freq",
                                                 chowdsp::ParamUtils::createNormalisableRange (0.5f, 10.0f, 2.0f),
                                                 1.0f };

    chowdsp::PercentParameter::Ptr modDepthParam { juce::ParameterID { "mod_depth", 100 },
                                                   "Mod. Depth" };

#if XSIMD_WITH_AVX
    chowdsp::BoolParameter::Ptr useAVXParam { juce::ParameterID { "use_avx", 100 },
                                              "Use AVX",
                                              false };
#endif
};

class ModalReverbPlugin : public chowdsp::PluginBase<chowdsp::PluginStateImpl<Parameters>>
{
public:
    ModalReverbPlugin();

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    void processAudioBlock (juce::AudioBuffer<float>& buffer) override;

    juce::AudioProcessorEditor* createEditor() override;

private:
#if JUCE_INTEL
#if XSIMD_WITH_AVX
    chowdsp::ModalFilterBank<ModeParams::numModes, float, xsimd::avx> modalFilterBankAVX;
#endif
    chowdsp::ModalFilterBank<ModeParams::numModes, float, xsimd::sse4_1> modalFilterBank;
#elif JUCE_ARM
    chowdsp::ModalFilterBank<ModeParams::numModes> modalFilterBank;
#endif

    chowdsp::SineWave<float> modSine;
    chowdsp::Buffer<float> modBuffer;

    juce::dsp::DryWetMixer<float> mixer;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ModalReverbPlugin)
};
