// ═══════════════════════════════════════════════════════════════════════════
// PLAYLIST CONFIG — edit this file to change video order, titles, and media.
// Open playlist/index.html and point it at this config file.
// Reorder, add, or remove entries in `items` to change the playlist.
// ═══════════════════════════════════════════════════════════════════════════

const PLAYLIST_CONFIG = {
  title: "Arivu Repositories",
  subtitle:
    "An introduction to the Arivu repositories and how to navigate the knowledge path.",
  // Path from playlist/index.html to this folder (where .mp4 and thumbnails live)
  mediaBase: "../videos/",
  items: [
    {
      video: "How GPU Programs Work.mp4",
      thumbnail: "How GPUs Programs Work.png",
      title: "How GPU Programs Work",
      description:
        "Understand how a GPU runs work — from kernels and Streaming Multiprocessors to grids, blocks, threads, and warps. #GPU #CUDA #GPUProgramming #ParallelComputing #AI",
    },
    {
      video: "Single-GPU Programming Model.mp4",
      thumbnail: "Single-GPU Programming Model.png",
      title: "Single-GPU Programming Model",
      description:
        "A deep dive into the CUDA programming model: grids, blocks, threads, and warps, and how they map to GPU hardware. #CUDA #GPUProgramming #CUDAProgramming #ParallelComputing #DeepLearning",
    },
    {
      video: "Multi-GPU Programming Model.mp4",
      thumbnail: "Multi-GPU Programming Model.png",
      title: "Multi-GPU Programming Model",
      description:
        "Learn how work is split across multiple GPU devices for data-parallel batch processing. #MultiGPU #GPUProgramming #CUDA #DistributedComputing #AI",
    },
    {
      video: "GPU System Limits — How They Constrain and Shape Parallelization.mp4",
      thumbnail: "GPU System Limits — How They Constrain and Shape Parallelization.png",
      title: "GPU System Limits — How They Constrain and Shape Parallelization",
      description:
        "Explore how hardware limits (from P100, V100, A100 whitepapers) constrain and shape parallelization strategies. #GPU #CUDA #Parallelization #NvidiaGPU #AIHardware",
    },
    {
      video: "GPU Performance Tuning Knobs For Beginners.mp4",
      thumbnail: "GPU Performance Tuning Knobs For Beginners.png",
      title: "GPU Performance Tuning Knobs for Beginners",
      description:
        "Learn the key performance tuning knobs — block size, occupancy, and warp divergence — tied back to the GPU hardware hierarchy. #GPUOptimization #CUDA #PerformanceTuning #GPUProgramming #AI",
    },
  ],
};
