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
      video: "the_arivu_repositories.mp4",
      thumbnail: "Arivu_Learning_Project_Overview.png",
      title: "The Arivu Repositories",
      description:
        "Discover the Arivu knowledge path and how the repositories are organized.",
    },
  ],
};
