package at.fhtw;


import org.jocl.CL;

import static org.jocl.CL.clGetPlatformIDs;

public class JOCLSetupTest {
    public static void main(String[] args) {
        // Aktiviere JOCL Exceptions, damit Fehler direkt als Java-Exceptions geworfen werden
        CL.setExceptionsEnabled(true);

        // Array vorbereiten, um die Anzahl der Plattformen zu speichern
        int[] numPlatformsArray = new int[1];

        // OpenCL Plattformen abfragen
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        System.out.println("Erfolgreich geladen!");
        System.out.println("Gefundene OpenCL Plattformen: " + numPlatforms);

        if (numPlatforms > 0) {
            System.out.println("Dein Setup ist perfekt. Du kannst jetzt OpenCL programmieren.");
        } else {
            System.out.println("JOCL funktioniert, aber es wurde kein OpenCL-Treiber auf dem System gefunden.");
        }
    }
}