setup_example_plugin(ModalSpringReverb msRv)

target_link_libraries(ModalSpringReverb
    PRIVATE
        juce::juce_dsp
        chowdsp::chowdsp_modal_dsp
        chowdsp::chowdsp_sources
        chowdsp::chowdsp_plugin_state
)

target_sources(ModalSpringReverb
    PRIVATE
        ModalReverbPlugin.cpp
)

include(CheckCXXCompilerFlag)
if(MSVC)
    CHECK_CXX_COMPILER_FLAG("/arch:AVX" COMPILER_OPT_ARCH_AVX_SUPPORTED)
    if(COMPILER_OPT_ARCH_AVX_SUPPORTED)
        message(STATUS "AVX flags enabled for MSVC compiler!")
        target_compile_options(ModalSpringReverb PUBLIC /arch:AVX2)
    else()
        message(STATUS "Unable to add AVX flags for MSVC compiler!")
    endif()
else()
    CHECK_CXX_COMPILER_FLAG("-mavx -mfma" COMPILER_OPT_ARCH_AVX_SUPPORTED)
    if (COMPILER_OPT_ARCH_AVX_SUPPORTED)
        message(STATUS "AVX flags enabled for ${CMAKE_CXX_COMPILER_ID} compiler!")
        target_compile_options(ModalSpringReverb PUBLIC -mavx -mfma)
    else()
        message(STATUS "Unable to enable AVX flags for ${CMAKE_CXX_COMPILER_ID} compiler!")
    endif()
endif()
