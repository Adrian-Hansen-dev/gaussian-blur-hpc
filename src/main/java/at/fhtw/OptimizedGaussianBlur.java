package at.fhtw;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import org.jocl.*;

import static org.jocl.CL.*;

public class OptimizedGaussianBlur {

    private static final float[] GAUSSIAN_KERNEL_1D = {
            0.028532f, 0.067234f, 0.124009f, 0.179044f, 0.202363f,
            0.179044f, 0.124009f, 0.067234f, 0.028532f
    };

    private static final int KERNEL_SIZE = GAUSSIAN_KERNEL_1D.length;

    public static void main(String[] args) {
        String inputPath = "input/shuttle.jpg";
        String outputPath = "output/shuttle_blurred.jpg";

        BufferedImage image = null;
        try {
            image = ImageIO.read(new File(inputPath));
            if (image == null) {
                System.err.println("Could not read image: " + inputPath);
                return;
            }
        } catch (IOException e) {
            System.err.println("Error loading image: " + e.getMessage());
            return;
        }

        int width = image.getWidth();
        int height = image.getHeight();
        System.out.println("Image loaded: " + width + "x" + height);

        CL.setExceptionsEnabled(true);

        int[] numPlatformsArray = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        cl_platform_id[] platforms = new cl_platform_id[numPlatformsArray[0]];
        clGetPlatformIDs(numPlatformsArray[0], platforms, null);
        cl_platform_id platform = platforms[0];

        int[] numDevicesArray = new int[1];
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, null, numDevicesArray);
        cl_device_id[] devices = new cl_device_id[numDevicesArray[0]];
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevicesArray[0], devices, null);
        cl_device_id device = devices[0];

        cl_context context = clCreateContext(null, 1, new cl_device_id[]{device}, null, null, null);
        cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, null);

        String programSource = readKernelSource("src/main/java/at/fhtw/blur.cl");
        cl_program program = clCreateProgramWithSource(context, 1, new String[]{programSource}, null, null);
        clBuildProgram(program, 0, null, null, null, null);

        cl_kernel kernel = clCreateKernel(program, "gaussianBlur1D", null);

        long[] maxWorkGroupSize = new long[1];
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, Sizeof.size_t, Pointer.to(maxWorkGroupSize), null);

        long[] maxWorkItemSizes = new long[3];
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3 * Sizeof.size_t, Pointer.to(maxWorkItemSizes), null);

        // Required check for system capabilities [cite: 12, 21]
        if (width > maxWorkGroupSize[0] || height > maxWorkGroupSize[0]) {
            System.err.println("Error: The image dimensions exceed the device's maximum work-group size.");
            System.exit(1);
        }

        int[] argbPixels = image.getRGB(0, 0, width, height, null, 0, width);
        byte[] inputPixels = new byte[width * height * 4];
        for (int i = 0; i < argbPixels.length; i++) {
            int pixel = argbPixels[i];
            inputPixels[i * 4]     = (byte) ((pixel >> 16) & 0xFF);
            inputPixels[i * 4 + 1] = (byte) ((pixel >> 8) & 0xFF);
            inputPixels[i * 4 + 2] = (byte) (pixel & 0xFF);
            inputPixels[i * 4 + 3] = (byte) ((pixel >> 24) & 0xFF);
        }
        byte[] outputPixels = new byte[width * height * 4];

        long pixelBufferSize = (long) width * height * 4;

        cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, pixelBufferSize, null, null);
        cl_mem intermediateBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, pixelBufferSize, null, null);
        cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, pixelBufferSize, null, null);

        clEnqueueWriteBuffer(commandQueue, inputBuffer, CL_TRUE, 0, pixelBufferSize, Pointer.to(inputPixels), 0, null, null);
        cl_mem filterBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (long) KERNEL_SIZE * Sizeof.cl_float, Pointer.to(GAUSSIAN_KERNEL_1D), null);

        long[] globalWorkSize = new long[]{width, height};

        // Horizontal pass [cite: 11]
        long[] localWorkSizeH = new long[]{width, 1};
        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(inputBuffer));
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(intermediateBuffer));
        clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{width}));
        clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{height}));
        clSetKernelArg(kernel, 4, Sizeof.cl_mem, Pointer.to(filterBuffer));
        clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[]{KERNEL_SIZE}));
        clSetKernelArg(kernel, 6, Sizeof.cl_int, Pointer.to(new int[]{1})); // 1 = horizontal
        clSetKernelArg(kernel, 7, (long) width * 4, null);

        clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, globalWorkSize, localWorkSizeH, 0, null, null);

        // Vertical pass [cite: 11]
        long[] localWorkSizeV = new long[]{1, height};
        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(intermediateBuffer));
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(outputBuffer));
        clSetKernelArg(kernel, 6, Sizeof.cl_int, Pointer.to(new int[]{0})); // 0 = vertical
        clSetKernelArg(kernel, 7, (long) height * 4, null);

        clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, globalWorkSize, localWorkSizeV, 0, null, null);

        clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, pixelBufferSize, Pointer.to(outputPixels), 0, null, null);

        int[] resultPixels = new int[width * height];
        for (int i = 0; i < resultPixels.length; i++) {
            int r = outputPixels[i * 4]     & 0xFF;
            int g = outputPixels[i * 4 + 1] & 0xFF;
            int b = outputPixels[i * 4 + 2] & 0xFF;
            int a = outputPixels[i * 4 + 3] & 0xFF;
            resultPixels[i] = (a << 24) | (r << 16) | (g << 8) | b;
        }

        BufferedImage outputImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        outputImage.setRGB(0, 0, width, height, resultPixels, 0, width);
        try {
            ImageIO.write(outputImage, "jpg", new File(outputPath));
            System.out.println("Blurred image saved to: " + outputPath);
        } catch (IOException e) {
            System.err.println("Error saving output image: " + e.getMessage());
        }

        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(intermediateBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseMemObject(filterBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
    }

    private static String readKernelSource(String filePath) {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append("\n");
            }
        } catch (IOException e) {
            System.err.println("Could not read kernel file: " + filePath);
            System.exit(1);
        }
        return sb.toString();
    }
}