package org.example;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

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
        int filterSize = 9; // Aktuell noch Platzhalter
    }
}
