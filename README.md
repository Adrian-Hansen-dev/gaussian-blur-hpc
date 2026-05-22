# Gaussian Blur HPC

A Java/OpenCL project using JOCL for GPU-accelerated image processing.

## Prerequisites

### All Platforms
- **Java JDK 17+** — [Download](https://adoptium.net/)
- **Apache Maven 3.6+** — [Download](https://maven.apache.org/download.cgi)
- **OpenCL-capable GPU** with up-to-date drivers

### Windows-specific
- **GPU drivers with OpenCL support:**
  - NVIDIA: Install the latest [NVIDIA Game Ready or Studio drivers](https://www.nvidia.com/Download/index.aspx) — OpenCL is included
  - AMD: Install [AMD Adrenalin drivers](https://www.amd.com/en/support) — OpenCL is included
  - Intel (integrated graphics): Install [Intel Graphics Driver](https://www.intel.com/content/www/us/en/download-center/home.html) — includes Intel OpenCL runtime
- **No separate OpenCL SDK is needed** — the drivers ship the runtime

> To verify OpenCL is available on your system, you can use [GPU-Z](https://www.techpowerup.com/gpuz/) or [CPU-Z](https://www.cpuid.com/softwares/cpu-z.html) and check the OpenCL entry under the Graphics tab.

## Running

### Windows (Command Prompt or PowerShell)

```cmd
mvn compile exec:java -Dexec.mainClass="at.fhtw.JOCLSetupTest"
```

### macOS / Linux

```bash
mvn compile exec:java -Dexec.mainClass="at.fhtw.JOCLSetupTest"
```

Expected output if everything is set up correctly:

```
Erfolgreich geladen!
Gefundene OpenCL Plattformen: 1
Dein Setup ist perfekt. Du kannst jetzt OpenCL programmieren.
```

If you see `0 Plattformen`, your GPU drivers are missing or do not expose an OpenCL runtime.
