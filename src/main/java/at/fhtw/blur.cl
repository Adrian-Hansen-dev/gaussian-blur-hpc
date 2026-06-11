__kernel void gaussianBlur2D(__global const unsigned char* inputImage,
                             __global unsigned char* outputImage,
                             const int width,
                             const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    const float kernelValues[81] = {
   0.010648, 0.011247, 0.011695, 0.011972, 0.012066, 0.011972, 0.011695, 0.011247, 0.010648,
   0.011247, 0.011879, 0.012352, 0.012645, 0.012744, 0.012645, 0.012352, 0.011879, 0.011247,
   0.011695, 0.012352, 0.012844, 0.013149, 0.013252, 0.013149, 0.012844, 0.012352, 0.011695,
   0.011972, 0.012645, 0.013149, 0.013461, 0.013566, 0.013461, 0.013149, 0.012645, 0.011972,
   0.012066, 0.012744, 0.013252, 0.013566, 0.013673, 0.013566, 0.013252, 0.012744, 0.012066,
   0.011972, 0.012645, 0.013149, 0.013461, 0.013566, 0.013461, 0.013149, 0.012645, 0.011972,
   0.011695, 0.012352, 0.012844, 0.013149, 0.013252, 0.013149, 0.012844, 0.012352, 0.011695,
   0.011247, 0.011879, 0.012352, 0.012645, 0.012744, 0.012645, 0.012352, 0.011879, 0.011247,
   0.010648, 0.011247, 0.011695, 0.011972, 0.012066, 0.011972, 0.011695, 0.011247, 0.010648,
    };

    int radius = 4;
    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;

    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {

            int targetX = x + kx;
            int targetY = y + ky;

            // horizontal border limits
            if (targetX < 0) {
                targetX = 0;
            } else if (targetX >= width) {
                targetX = width - 1;
            }

            // vertical border limits
            if (targetY < 0) {
                targetY = 0;
            } else if (targetY >= height) {
                targetY = height - 1;
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

__kernel void gaussianBlur1D(__global const unsigned char* inputImage,
                             __global unsigned char* outputImage,
                             const int width,
                             const int height,
                             __global const float* filterKernel,
                             const int kernelSize,
                             const int horizontal,
                             __local unsigned char* localPixels)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    int localIdx;
    int lineLength;

    if (horizontal) {
        localIdx = get_local_id(0);
        lineLength = width;
    } else {
        localIdx = get_local_id(1);
        lineLength = height;
    }

    int pixelIndex = (y * width + x) * 4;
    int localBase = localIdx * 4;

    localPixels[localBase + 0] = inputImage[pixelIndex + 0];
    localPixels[localBase + 1] = inputImage[pixelIndex + 1];
    localPixels[localBase + 2] = inputImage[pixelIndex + 2];
    localPixels[localBase + 3] = inputImage[pixelIndex + 3];

    barrier(CLK_LOCAL_MEM_FENCE);

    int radius = kernelSize / 2;
    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;

    for (int k = -radius; k <= radius; k++) {
        int sampleIdx = localIdx + k;

        if (sampleIdx < 0) {
            sampleIdx = 0;
        } else if (sampleIdx >= lineLength) {
            sampleIdx = lineLength - 1;
        }

        int sampleBase = sampleIdx * 4;
        float weight = filterKernel[k + radius];

        sumR += (float)localPixels[sampleBase + 0] * weight;
        sumG += (float)localPixels[sampleBase + 1] * weight;
        sumB += (float)localPixels[sampleBase + 2] * weight;
    }

    int outputIndex = (y * width + x) * 4;
    outputImage[outputIndex + 0] = (unsigned char)sumR;
    outputImage[outputIndex + 1] = (unsigned char)sumG;
    outputImage[outputIndex + 2] = (unsigned char)sumB;
    outputImage[outputIndex + 3] = localPixels[localBase + 3];
}