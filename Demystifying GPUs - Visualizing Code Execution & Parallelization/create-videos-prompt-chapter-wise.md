# For each of  the file listed below that are available in sources, use its content & create **a dedicated educational video**, with the same name (except for .mp4 extension).

- how-gpu-programs-work.md
- gpu-programming-model-single-gpu-scenario.md
- gpu-programming-model-multi-gpu-scenario.md
- walk_through_vector_add_example_and_tuning.md
- performance-tuning-knobs-for-beginners.md
- how-gpu-system-limits-constrain-and-shape-parallelization.md
- vector-add-example-in-python.md

## Overall Requirements

- Audience: AUDIENCE_DESCRIPTION (e.g., “beginner AI engineers”).
- Tone: calm, confident, non-hype, slightly conversational but technically accurate.
- Visual style: dark background, high-contrast light text, clean layout.
- Narration: professional male voice, steady pace, clear articulation.
- Use ONLY content from the lesson plan and attached official sources. Do not invent specifications or claims.
- The generated videos will be put together as a plalist on youtube.com. Hence keep narrative across them woven together as one larger consistent story.

## Video Structure

For EACH chapter in LESSON_PLAN_FILENAME:

1. **On-screen visuals**
   - Generate slides and simple animations directly in the video:
     - Slide title matching the chapter title.
     - 3–5 short bullet points or phrases (no long sentences).
     - Simple diagrams, icons, or callouts as described in the lesson plan (e.g., stacks, timelines, comparison charts).
   - Use dark-mode slides, minimal clutter, and consistent typography.
   - When diagrams are in context, show them visually (arrows, boxes, labels).

2. **Narration / Audio**
   - Automatically generate spoken narration that:
     - Introduces the chapter briefly.
     - Explains each bullet or visual element in clear, simple language.
     - Defines acronyms the first time they appear.
   - Narration must stay tightly aligned with subject matter.

3. **Flow Between Chapters**
   - Insert short transitions (simple fades or slide transitions) between chapters.
   - Briefly recap or bridge if the lesson plan indicates dependencies between chapters.

## Thank You / Outro Section

At the **end of the video**, add a short “Thank You / Next Steps” segment:

- Visual:
  - A final slide with a dark background and large “Thank You” text.
  - 2–3 short bullet points summarizing key takeaways or next steps (e.g., “Use this framework before buying”, “Apply it to A100 vs H100”, “Avoid spec-sheet traps”).
- Narration:
  - 2–3 sentences thanking the viewer.
  - A brief reminder of the main message: evaluating GPUs holistically, not by single headline numbers.

## Technical & Formatting Notes

- Maintain consistent fonts, colors, and layout throughout.
- Ensure all on-screen text is readable at typical YouTube resolutions.
- Do NOT include any NotebookLM watermark, logo, or branding.
- Do NOT add unrelated marketing or tool promotion; keep the video purely educational.