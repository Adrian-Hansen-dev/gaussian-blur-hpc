__kernel void gaussianBlur2D(__global const unsigned char* inputImage,
                             __global unsigned char* outputImage,
                             const int width,
                             const int height) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    const float kernelValues[81] = {
        0.000000f, 0.000001f, 0.000007f, 0.000032f, 0.000053f, 0.000032f, 0.000007f, 0.000001f, 0.000000f,
        0.000001f, 0.000020f, 0.000239f, 0.001072f, 0.001768f, 0.001072f, 0.000239f, 0.000020f, 0.000001f,
        0.000007f, 0.000239f, 0.002915f, 0.013064f, 0.021539f, 0.013064f, 0.002915f, 0.000239f, 0.000007f,
        0.000032f, 0.001072f, 0.013064f, 0.058550f, 0.096533f, 0.058550f, 0.013064f, 0.001072f, 0.000032f,
        0.000053f, 0.001768f, 0.021539f, 0.096533f, 0.159156f, 0.096533f, 0.021539f, 0.001768f, 0.000053f,
        0.000032f, 0.001072f, 0.013064f, 0.058550f, 0.096533f, 0.058550f, 0.013064f, 0.001072f, 0.000032f,
        0.000007f, 0.000239f, 0.002915f, 0.013064f, 0.021539f, 0.013064f, 0.002915f, 0.000239f, 0.000007f,
        0.000001f, 0.000020f, 0.000239f, 0.001072f, 0.001768f, 0.001072f, 0.000239f, 0.000020f, 0.000001f,
        0.000000f, 0.000001f, 0.000007f, 0.000032f, 0.000053f, 0.000032f, 0.000007f, 0.000001f, 0.000000f
    };

    int radius = 4;
    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;

    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {

            int targetX = x + kx;
            int targetY = y + ky;

            // horizontal border limits
            if (targetX < 0) {
                targetX = 0;               // Too far left? Snap to the leftmost pixel
            } else if (targetX >= width) {
                targetX = width - 1;       // Too far right? Snap to the rightmost pixel
            }

            // vertical border limits
            if (targetY < 0) {
                targetY = 0;               // Too far up? Snap to the top pixel
            } else if (targetY >= height) {
                targetY = height - 1;      // Too far down? Snap to the bottom pixel
            }

            int inputIndex = (targetY * width + targetX) * 4;

            int kernelIndex = (ky + radius) * 9 + (kx + radius);
            float weight = kernelValues[kernelIndex];

            sumR += (float)inputImage[inputIndex + 0] * weight;
            sumG += (float)inputImage[inputIndex + 1] * weight;
            sumB += (float)inputImage[inputIndex + 2] * weight;
        }
    }

    int outputIndex = (y * width + x) * 4;
    outputImage[outputIndex + 0] = (unsigned char)sumR;
    outputImage[outputIndex + 1] = (unsigned char)sumG;
    outputImage[outputIndex + 2] = (unsigned char)sumB;
    outputImage[outputIndex + 3] = inputImage[outputIndex + 3];
}