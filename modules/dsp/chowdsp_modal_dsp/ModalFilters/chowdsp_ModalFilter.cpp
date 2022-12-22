namespace chowdsp
{
template <typename T>
void ModalFilter<T>::prepare (T sampleRate)
{
    fs = sampleRate;

    decayFactor = calcDecayFactor();
    oscCoef = calcOscCoef();

    updateParams();
    reset();
}

template <typename T>
void ModalFilter<T>::processBlock (T* buffer, const int numSamples)
{
    for (int n = 0; n < numSamples; ++n)
        buffer[n] = processSample (buffer[n]);
}

#if ! CHOWDSP_NO_XSIMD
//============================================================
template <typename FloatType, typename Arch>
void ModalFilter<xsimd::batch<FloatType, Arch>>::prepare (FloatType sampleRate)
{
    fs = sampleRate;

    decayFactor = calcDecayFactor();
    oscCoef = calcOscCoef();

    updateParams();
    reset();
}

template <typename FloatType, typename Arch>
void ModalFilter<xsimd::batch<FloatType, Arch>>::processBlock (VType* buffer, const int numSamples)
{
    for (int n = 0; n < numSamples; ++n)
        buffer[n] = processSample (buffer[n]);
}
#endif // ! CHOWDSP_NO_XSIMD

} // namespace chowdsp
