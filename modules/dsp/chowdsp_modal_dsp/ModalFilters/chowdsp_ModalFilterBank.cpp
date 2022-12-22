namespace chowdsp
{
template <size_t maxNumModes, typename SampleType, typename Arch>
template <typename PerModeFunc, typename PerVecModeFunc>
void ModalFilterBank<maxNumModes, SampleType, Arch>::doForModes (PerModeFunc&& perModeFunc, PerVecModeFunc&& perVecModeFunc)
{
    size_t i = 0;
    for (auto& mode : modes)
    {
        for (size_t j = 0; j < vecSize; ++j)
            perModeFunc (j, i * vecSize + j);

        perVecModeFunc (mode);
        i++;
    }
}

template <size_t maxNumModes, typename SampleType, typename Arch>
typename ModalFilterBank<maxNumModes, SampleType, Arch>::Vec ModalFilterBank<maxNumModes, SampleType, Arch>::tau2t60 (Vec tau, SampleType originalSampleRate)
{
    return Vec ((SampleType) 1) / (xsimd::log (xsimd::exp (originalSampleRate / tau)) / log1000);
}

template <size_t maxNumModes, typename SampleType, typename Arch>
void ModalFilterBank<maxNumModes, SampleType, Arch>::setModeAmplitudes (const SampleType (&ampsReal)[maxNumModes], const SampleType (&ampsImag)[maxNumModes], SampleType normalize)
{
    for (size_t i = 0; i < (size_t) maxNumModes; ++i)
        amplitudeData[i] = std::complex<SampleType> { ampsReal[i], ampsImag[i] };

    updateAmplitudeNormalizationFactor (normalize);
    setModeAmplitudesInternal();
}

template <size_t maxNumModes, typename SampleType, typename Arch>
void ModalFilterBank<maxNumModes, SampleType, Arch>::setModeAmplitudes (const std::complex<SampleType> (&amps)[maxNumModes], SampleType normalize)
{
    std::copy (amps, amps + maxNumModes, amplitudeData.begin());
    updateAmplitudeNormalizationFactor (normalize);
    setModeAmplitudesInternal();
}

template <size_t maxNumModes, typename SampleType, typename Arch>
void ModalFilterBank<maxNumModes, SampleType, Arch>::updateAmplitudeNormalizationFactor (SampleType normalize)
{
    auto lowestModeMag = std::abs (amplitudeData[0]);
    if (normalize > (SampleType) 0 && lowestModeMag > (SampleType) 0)
        amplitudeNormalizationFactor = normalize / lowestModeMag;
    else
        amplitudeNormalizationFactor = (SampleType) 1;
}

template <size_t maxNumModes, typename SampleType, typename Arch>
void ModalFilterBank<maxNumModes, SampleType, Arch>::setModeAmplitudesInternal()
{
    std::complex<SampleType> modeAmps alignas (Arch::alignment())[vecSize] {};
    doForModes (
        [&] (size_t j, size_t modeIndex)
        {
            modeAmps[j] = modeIndex < numModesToProcess ? amplitudeData[modeIndex] * amplitudeNormalizationFactor : (SampleType) 0;
        },
        [&] (auto& mode)
        { mode.setAmp (xsimd::load_aligned<Arch> (modeAmps)); });
}

template <size_t maxNumModes, typename SampleType, typename Arch>
void ModalFilterBank<maxNumModes, SampleType, Arch>::setModeFrequencies (const SampleType (&baseFrequencies)[maxNumModes], SampleType frequencyMultiplier)
{
    SampleType modeFreqs alignas (Arch::alignment())[vecSize] {};
    doForModes (
        [&] (size_t j, size_t modeIndex)
        {
            auto freq = modeIndex < maxNumModes ? (baseFrequencies[modeIndex] * frequencyMultiplier) : (SampleType) 0;
            modeFreqs[j] = freq > maxFreq ? (SampleType) 0 : freq;
        },
        [&] (auto& mode)
        { mode.setFreq (xsimd::load_aligned<Arch> (modeFreqs)); });
}

template <size_t maxNumModes, typename SampleType, typename Arch>
void ModalFilterBank<maxNumModes, SampleType, Arch>::setModeDecays (const SampleType (&baseTaus)[maxNumModes], SampleType originalSampleRate, SampleType decayFactor)
{
    SampleType modeTaus alignas (Arch::alignment())[vecSize] {};
    doForModes (
        [&] (size_t j, size_t modeIndex)
        { modeTaus[j] = modeIndex < maxNumModes ? baseTaus[modeIndex] : (SampleType) 1; },
        [&] (auto& mode)
        { mode.setDecay (tau2t60 (xsimd::load_aligned<Arch> (modeTaus), originalSampleRate) * decayFactor); });
}

template <size_t maxNumModes, typename SampleType, typename Arch>
void ModalFilterBank<maxNumModes, SampleType, Arch>::setModeDecays (const SampleType (&t60s)[maxNumModes])
{
    SampleType modeT60s alignas (Arch::alignment())[vecSize] {};
    doForModes (
        [&] (size_t j, size_t modeIndex)
        { modeT60s[j] = modeIndex < maxNumModes ? t60s[modeIndex] : (SampleType) 0; },
        [&] (auto& mode)
        { mode.setDecay (xsimd::load_aligned<Arch> (modeT60s)); });
}

template <size_t maxNumModes, typename SampleType, typename Arch>
void ModalFilterBank<maxNumModes, SampleType, Arch>::setNumModesToProcess (size_t newNumModesToProcess)
{
    jassert (newNumModesToProcess <= maxNumModes);

    numModesToProcess = newNumModesToProcess;
    numVecModesToProcess = Math::ceiling_divide (newNumModesToProcess, vecSize);
    setModeAmplitudesInternal();

    for (size_t modeIndex = numVecModesToProcess; modeIndex < maxNumVecModes; ++modeIndex)
        modes[modeIndex].reset();
}

template <size_t maxNumModes, typename SampleType, typename Arch>
void ModalFilterBank<maxNumModes, SampleType, Arch>::prepare (double sampleRate, int samplesPerBlock)
{
    maxFreq = SampleType (0.495 * sampleRate);
    renderBuffer.setMaxSize (1, samplesPerBlock);

    for (auto& mode : modes)
        mode.prepare ((SampleType) sampleRate);
}

template <size_t maxNumModes, typename SampleType, typename Arch>
void ModalFilterBank<maxNumModes, SampleType, Arch>::reset()
{
    for (auto& mode : modes)
        mode.reset();
}

template <size_t maxNumModes, typename SampleType, typename Arch>
void ModalFilterBank<maxNumModes, SampleType, Arch>::process (const BufferView<const SampleType>& block) noexcept
{
    const auto numSamples = block.getNumSamples();

    renderBuffer.setCurrentSize (1, numSamples);
    renderBuffer.clear();

    const auto* blockPtr = block.getReadPointer (0);
    auto* renderPtr = renderBuffer.getWritePointer (0);

    for (size_t modeIdx = 0; modeIdx < numVecModesToProcess; ++modeIdx)
    {
        for (int n = 0; n < numSamples; ++n)
            renderPtr[n] += xsimd::reduce_add (modes[modeIdx].processSample (blockPtr[n]));
    }
}

template <size_t maxNumModes, typename SampleType, typename Arch>
template <typename Modulator>
void ModalFilterBank<maxNumModes, SampleType, Arch>::processWithModulation (const BufferView<const SampleType>& block, Modulator&& modulator) noexcept
{
    const auto numSamples = block.getNumSamples();

    renderBuffer.setCurrentSize (1, numSamples);
    renderBuffer.clear();

    const auto* blockPtr = block.getReadPointer (0);
    auto* renderPtr = renderBuffer.getWritePointer (0);

    for (size_t modeIdx = 0; modeIdx < numVecModesToProcess; ++modeIdx)
    {
        for (int n = 0; n < numSamples; ++n)
        {
            modulator (modes[modeIdx], modeIdx, n);
            renderPtr[n] += xsimd::reduce_add (modes[modeIdx].processSample (blockPtr[n]));
        }
    }
}

template <size_t maxNumModes, typename SampleType, typename Arch>
const SampleType ModalFilterBank<maxNumModes, SampleType, Arch>::log1000 = std::log ((SampleType) 1000);
} // namespace chowdsp
