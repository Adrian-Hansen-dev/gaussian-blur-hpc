package org.example;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import org.jocl.*;
import static org.jocl.CL.*;

public class SimpleGaussianBlur {
    public static void main(String[] args) {
        String inputPath = "input/shuttle.jpg";
        String outputPath = "output/shuttle_blurred.jpg"; // Aktuell noch Platzhalter

        BufferedImage image = null;
        try {
            image = ImageIO.read(new File(inputPath));
            if (image == null) {
                System.err.println("Fehler: Konnte Bild nicht lesen. Existiert " + inputPath + "?");
                return;
            }
        } catch (IOException e) {
            System.err.println("Fehler beim Laden des Bildes: " + e.getMessage());
            return;
        }

        int width = image.getWidth();
        int height = image.getHeight();

        // Pixel als eindimensionales int-Array holen (ARGB)
        int[] inputPixels = image.getRGB(0, 0, width, height, null, 0, width);
        int[] outputPixels = new int[width * height];
        System.out.println(inputPixels.length + " Pixel geladen.");

        float[] filterValues = new float[]{
                0.000000f, 0.000001f, 0.000007f, 0.000032f, 0.000053f, 0.000032f, 0.000007f, 0.000001f, 0.000000f,
                0.000001f, 0.000020f, 0.000239f, 0.001072f, 0.001768f, 0.001072f, 0.000239f, 0.000020f, 0.000001f,
                0.000007f, 0.000239f, 0.002915f, 0.013064f, 0.021539f, 0.013064f, 0.002915f, 0.000239f, 0.000007f,
                0.000032f, 0.001072f, 0.013064f, 0.058550f, 0.096533f, 0.058550f, 0.013064f, 0.001072f, 0.000032f,
                0.000053f, 0.001768f, 0.021539f, 0.096533f, 0.159156f, 0.096533f, 0.021539f, 0.001768f, 0.000053f,
                0.000032f, 0.001072f, 0.013064f, 0.058550f, 0.096533f, 0.058550f, 0.013064f, 0.001072f, 0.000032f,
                0.000007f, 0.000239f, 0.002915f, 0.013064f, 0.021539f, 0.013064f, 0.002915f, 0.000239f, 0.000007f,
                0.000001f, 0.000020f, 0.000239f, 0.001072f, 0.001768f, 0.001072f, 0.000239f, 0.000020f, 0.000001f,
                0.000000f, 0.000001f, 0.000007f, 0.000032f, 0.000053f, 0.000032f, 0.000007f, 0.000001f, 0.000000f
        }; // Aktuell noch Platzhalter
        int filterSize = 9;

        // ===== OpenCL Setup =====

        // Step 0: Enable exceptions so OpenCL errors become Java exceptions
        CL.setExceptionsEnabled(true);

        // Step 1: Get a platform
        // A "platform" = an OpenCL implementation installed on this machine
        // (e.g. Apple's, NVIDIA's, Intel's). We just take the first one
        int[] numPlatformsArray = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        if (numPlatformsArray[0] == 0) {
            System.err.println("No OpenCL platform found!");
            return;
        }
        cl_platform_id[] platforms = new cl_platform_id[numPlatformsArray[0]];
        clGetPlatformIDs(1, platforms, null);
        cl_platform_id platform = platforms[0];

        // Step 2: Get a device
        int[] numDevicesArray = new int[1];
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, null, numDevicesArray);
        if (numDevicesArray[0] == 0) {
            System.err.println("No OpenCL device found!");
            return;
        }
        cl_device_id[] devices = new cl_device_id[numDevicesArray[0]];
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevicesArray[0], devices, null);
        cl_device_id device = devices[0];

        // Step 3: Create a context
        // A "context" groups together a device + its memory objects + programs
        cl_context context = clCreateContext(null, 1, new cl_device_id[]{device}, null, null, null);

        // Step 4: Create a command queue
        // The queue is how we send instructions to the device
        // Commands execute in order (FIFO)
        cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, null);

        // Step 5: Load the kernel source code from a .cl file, compile it
        // Note: placeholder for now, we will read the actual kernel source from a file later
        String programSource = readKernelSource("src/main/resources/gaussian_blur.cl");
        cl_program program = clCreateProgramWithSource(context, 1, new String[]{programSource}, null, null);
        clBuildProgram(program, 0, null, null, null, null);

        // Step 6: Create the kernel
        // A program can contain multiple __kernel functions
        // clCreateKernel picks one by name — this must match the function name in the .cl file
        cl_kernel kernel = clCreateKernel(program, "gaussian_blur", null);

        // Step 7: Query device capabilities (required by the assignment)
        // We need to check max work group size and max work item sizes to make sure our 2D NDRange (width x height) is valid
        long[] maxWorkGroupSize = new long[1];
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, Sizeof.cl_long, Pointer.to(maxWorkGroupSize), null);
        System.out.println("Max work group size: " + maxWorkGroupSize[0]);

        long[] maxWorkItemDimensions = new long[1];
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, Sizeof.cl_long, Pointer.to(maxWorkItemDimensions), null);

        long[] maxWorkItemSizes = new long[(int) maxWorkItemDimensions[0]];
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, maxWorkItemDimensions[0] * Sizeof.cl_long, Pointer.to(maxWorkItemSizes), null);
        System.out.println("Max work item sizes: dim0=" + maxWorkItemSizes[0] + " dim1=" + maxWorkItemSizes[1]);

        // TODO: Next steps — create buffers, set kernel args, execute, read back results

        // Cleanup: release all OpenCL resources in reverse order
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
