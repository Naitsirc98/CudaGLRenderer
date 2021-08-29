# CudaGLRenderer

A simple OpenGL gltf2 Physically Based Renderer that uses CUDA for Post effects rendering operations.

The CUDA pipeline writes directly to the OpenGL framebuffer color attachment to avoiding intermediate copies.

You can select multiple post processing effects at runtime with the UI window on the left, as well as the camera exposure.

![Gamma Correction + Tonemapping](glcuda_gamma.PNG?raw=true "Gamma Correction + Tonemapping")

![Color Inversion](glcuda_inversion.PNG?raw=true "Color Inversion")

![Grayscale + Sharpening](glcuda_grayscale_sharp.PNG?raw=true "Grayscale + Sharpening")
