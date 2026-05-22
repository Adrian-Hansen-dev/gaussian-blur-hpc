package at.fhtw;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import org.jocl.*;
import static org.jocl.CL.*;

public class SimpleGaussianBlur {

    public static void main(String[] args) {
        String inputPath = "input/shuttle.jpg";
        String outputPath = "output/shuttle_blurred.jpg";

        // Load input image
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

        // Convert int[] ARGB to byte[] RGBA for the kernel
        int[] argbPixels = image.getRGB(0, 0, width, height, null, 0, width);
        byte[] inputPixels = new byte[width * height * 4];
        for (int i = 0; i < argbPixels.length; i++) {
            int pixel = argbPixels[i];
            inputPixels[i * 4]     = (byte) ((pixel >> 16) & 0xFF); // R
            inputPixels[i * 4 + 1] = (byte) ((pixel >> 8) & 0xFF);  // G
            inputPixels[i * 4 + 2] = (byte) (pixel & 0xFF);         // B
            inputPixels[i * 4 + 3] = (byte) ((pixel >> 24) & 0xFF); // A
        }
        byte[] outputPixels = new byte[width * height * 4];

        // OpenCL setup
        CL.setExceptionsEnabled(true);

        int[] numPlatformsArray = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        if (numPlatformsArray[0] == 0) {
            System.err.println("No OpenCL platform found!");
            return;
        }
        cl_platform_id[] platforms = new cl_platform_id[numPlatformsArray[0]];
        clGetPlatformIDs(1, platforms, null);
        cl_platform_id platform = platforms[0];

        int[] numDevicesArray = new int[1];
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, null, numDevicesArray);
        if (numDevicesArray[0] == 0) {
            System.err.println("No OpenCL device found!");
            return;
        }
        cl_device_id[] devices = new cl_device_id[numDevicesArray[0]];
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevicesArray[0], devices, null);
        cl_device_id device = devices[0];

        cl_context context = clCreateContext(null, 1, new cl_device_id[]{device}, null, null, null);
        cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, null);

        String programSource = readKernelSource("src/main/java/at/fhtw/blur.cl");
        cl_program program = clCreateProgramWithSource(context, 1, new String[]{programSource}, null, null);
        clBuildProgram(program, 0, null, null, null, null);

        cl_kernel kernel = clCreateKernel(program, "gaussianBlur2D", null);

        // Query and validate device capabilities
        long[] maxWorkGroupSize = new long[1];
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, Sizeof.cl_long, Pointer.to(maxWorkGroupSize), null);
        System.out.println("Max work group size: " + maxWorkGroupSize[0]);

       
        // Create buffers and upload input data to GPU
        Pointer ptrInputPixels = Pointer.to(inputPixels);
        Pointer ptrOutputPixels = Pointer.to(outputPixels);
        long pixelBufferSize = (long) width * height * 4;

        cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, pixelBufferSize, null, null);
        cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, pixelBufferSize, null, null);

        clEnqueueWriteBuffer(commandQueue, inputBuffer, CL_TRUE, 0, pixelBufferSize, ptrInputPixels, 0, null, null);

        // Set kernel arguments: gaussianBlur2D(inputImage, outputImage, width, height)
        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(inputBuffer));
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(outputBuffer));
        clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{width}));
        clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{height}));

        // Execute kernel with 2D NDRange (one work item per pixel)
        long[] globalWorkSize = new long[]{width, height};
        clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, globalWorkSize, null, 0, null, null);

        // Read results back from GPU
        clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, pixelBufferSize, ptrOutputPixels, 0, null, null);

        // Convert byte[] RGBA back to int[] ARGB and save
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

        // Cleanup
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
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
