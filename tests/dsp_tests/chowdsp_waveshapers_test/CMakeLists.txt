setup_catch_test(chowdsp_waveshapers_test MODULES chowdsp_waveshapers)

# These tests take _forever_ on Windows CI, so let's skip most of them for now.
if (WIN32)
    target_sources(chowdsp_waveshapers_test
        PRIVATE
            ADAAFullWaveRectifierTest.cpp
            ADAAHardClipperTest.cpp
            ADAATanhClipperTest.cpp
            ADAASoftClipperTest.cpp
    )
else()
    target_sources(chowdsp_waveshapers_test
            PRIVATE
            SoftClipperTest.cpp
            ADAAFullWaveRectifierTest.cpp
            ADAAHardClipperTest.cpp
            ADAATanhClipperTest.cpp
            ADAASoftClipperTest.cpp
            WaveMultiplierTest.cpp
            WestCoastFolderTest.cpp
            LookupTableLoadingTest.cpp
            )
endif()
