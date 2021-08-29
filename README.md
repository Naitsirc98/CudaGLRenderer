# CudaGLRenderer

A simple OpenGL gltf2 Physically Based Renderer that uses CUDA for Post effects rendering operations.

The CUDA pipeline writes directly to the OpenGL framebuffer color attachment to avoiding intermediate copies.
