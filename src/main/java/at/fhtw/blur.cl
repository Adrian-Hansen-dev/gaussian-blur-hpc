__kernel void gaussianBlur1D(
    __global const unsigned char* inputImage,
    __global unsigned char* outputImage,
    const int width,
    const int height,
    __global const float* filterKernel,
    const int kernelSize,
    const int horizontal,
    __local unsigned char* localPixels
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    bool inBounds = (x < width && y < height);

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

    if (inBounds) {
        localPixels[localBase + 0] = inputImage[pixelIndex + 0];
        localPixels[localBase + 1] = inputImage[pixelIndex + 1];
        localPixels[localBase + 2] = inputImage[pixelIndex + 2];
        localPixels[localBase + 3] = inputImage[pixelIndex + 3];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (inBounds) {
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

        outputImage[outputIndex + 0] = (unsigned char)(sumR + 0.5f);
        outputImage[outputIndex + 1] = (unsigned char)(sumG + 0.5f);
        outputImage[outputIndex + 2] = (unsigned char)(sumB + 0.5f);
        outputImage[outputIndex + 3] = localPixels[localBase + 3];
    }
}