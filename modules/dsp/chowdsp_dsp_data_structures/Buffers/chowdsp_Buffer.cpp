namespace chowdsp
{
template <typename SampleType, size_t alignment>
Buffer<SampleType, alignment>::Buffer (int numChannels, int numSamples)
{
    setMaxSize (numChannels, numSamples);
}

template <typename SampleType, size_t alignment>
void Buffer<SampleType, alignment>::setMaxSize (int numChannels, int numSamples)
{
    jassert (juce::isPositiveAndBelow (numChannels, maxNumChannels));
    jassert (numSamples > 0);

    rawData.clear();
    hasBeenCleared = true;
    currentNumChannels = 0;
    currentNumSamples = 0;

    rawData.resize ((size_t) numChannels, ChannelData ((size_t) numSamples, SampleType {}));
    std::fill (channelPointers.begin(), channelPointers.end(), nullptr);
    for (int ch = 0; ch < numChannels; ++ch)
        channelPointers[(size_t) ch] = rawData[(size_t) ch].data();

    setCurrentSize (numChannels, numSamples);
}

template <typename SampleType, size_t alignment>
void Buffer<SampleType, alignment>::setCurrentSize (int numChannels, int numSamples) noexcept
{
    const auto increasingNumChannels = numChannels > currentNumChannels;
    const auto increasingNumSamples = numSamples > currentNumSamples;

    if (increasingNumSamples)
        buffer_detail::clear (channelPointers.data(), 0, currentNumChannels, currentNumSamples, numSamples);

    if (increasingNumChannels)
        buffer_detail::clear (channelPointers.data(), currentNumChannels, numChannels, 0, numSamples);

    currentNumChannels = numChannels;
    currentNumSamples = numSamples;
}

template <typename SampleType, size_t alignment>
SampleType* Buffer<SampleType, alignment>::getWritePointer (int channel) noexcept
{
    hasBeenCleared = false;
    return channelPointers[(size_t) channel];
}

template <typename SampleType, size_t alignment>
const SampleType* Buffer<SampleType, alignment>::getReadPointer (int channel) const noexcept
{
    return channelPointers[(size_t) channel];
}

template <typename SampleType, size_t alignment>
SampleType** Buffer<SampleType, alignment>::getArrayOfWritePointers() noexcept
{
    hasBeenCleared = false;
    return channelPointers.data();
}

template <typename SampleType, size_t alignment>
const SampleType** Buffer<SampleType, alignment>::getArrayOfReadPointers() const noexcept
{
    return const_cast<const SampleType**> (channelPointers.data()); // NOSONAR (using const_cast to be more strict)
}

#if CHOWDSP_USING_JUCE
template <typename SampleType, size_t alignment>
juce::AudioBuffer<SampleType> Buffer<SampleType, alignment>::toAudioBuffer()
{
    return { getArrayOfWritePointers(), currentNumChannels, currentNumSamples };
}

template <typename SampleType, size_t alignment>
juce::AudioBuffer<SampleType> Buffer<SampleType, alignment>::toAudioBuffer() const
{
    return { const_cast<SampleType* const*> (getArrayOfReadPointers()), currentNumChannels, currentNumSamples }; // NOSONAR
}

#if JUCE_MODULE_AVAILABLE_juce_dsp
template <typename SampleType, size_t alignment>
AudioBlock<SampleType> Buffer<SampleType, alignment>::toAudioBlock()
{
    return { getArrayOfWritePointers(), (size_t) currentNumChannels, (size_t) currentNumSamples };
}

template <typename SampleType, size_t alignment>
AudioBlock<const SampleType> Buffer<SampleType, alignment>::toAudioBlock() const
{
    return { getArrayOfReadPointers(), (size_t) currentNumChannels, (size_t) currentNumSamples };
}
#endif
#endif

template <typename SampleType, size_t alignment>
void Buffer<SampleType, alignment>::clear() noexcept
{
    if (hasBeenCleared)
        return;

    buffer_detail::clear (channelPointers.data(), 0, currentNumChannels, 0, currentNumSamples);
    hasBeenCleared = true;
}
} // namespace chowdsp
