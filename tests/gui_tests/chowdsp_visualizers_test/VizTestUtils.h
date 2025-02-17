#include <juce_graphics/juce_graphics.h>

namespace VizTestUtils
{
inline juce::File getImagesDir()
{
    auto imagesDir = juce::File::getCurrentWorkingDirectory();
    while (imagesDir.getFileName() != "chowdsp_utils")
        imagesDir = imagesDir.getParentDirectory();
    return imagesDir.getChildFile ("tests/gui_tests/chowdsp_visualizers_test/Images");
}

inline void saveImage (const juce::Image& testImage, const juce::String& fileName)
{
    juce::File testPNGFile = getImagesDir().getChildFile (fileName);
    testPNGFile.deleteFile();
    testPNGFile.create();
    auto pngStream = testPNGFile.createOutputStream();
    if (pngStream->openedOk())
    {
        juce::PNGImageFormat pngImage;
        pngImage.writeImageToStream (testImage, *pngStream);
    }
}

inline juce::Image loadImage (const juce::String& fileName)
{
    return juce::PNGImageFormat::loadFrom (VizTestUtils::getImagesDir()
                                               .getChildFile (fileName));
}

inline void compareImages (juce::UnitTest& test, const juce::Image& testImage, const juce::Image& refImage)
{
    const auto width = testImage.getWidth();
    const auto height = testImage.getHeight();

    test.expectEquals (width, refImage.getWidth(), "Incorrect image width!");
    test.expectEquals (height, refImage.getHeight(), "Incorrect image height!");

    const auto testImageData = juce::Image::BitmapData { testImage, juce::Image::BitmapData::readOnly };
    const auto refImageData = juce::Image::BitmapData { refImage, juce::Image::BitmapData::readOnly };

    for (int w = 0; w < width; ++w)
    {
        for (int h = 0; h < height; ++h)
            test.expect (testImageData.getPixelColour (w, h) == refImageData.getPixelColour (w, h));
    }
}
} // namespace VizTestUtils
