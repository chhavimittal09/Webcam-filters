# Lumina Studio

### Real-Time Webcam Filters & Creative Computer Vision Studio

Lumina Studio is an interactive **Computer Vision web application** that applies real-time visual effects to a webcam feed using **OpenCV, Streamlit, and WebRTC**.

It goes beyond basic webcam filters by combining live image processing with filter combinations, adjustable intensity, keyboard controls, snapshots, photobooth mode, photo strips, video recording, and an in-browser gallery.

---

## ✨ Features

- 10 real-time webcam filters
- Filter Combo Lab to layer two filters together
- Adjustable filter intensity
- Mirror preview
- Instant frame capture
- 3-2-1 photobooth countdown
- Automatic 3-shot photo strips
- Processed video recording
- Keyboard-controlled interaction
- Built-in gallery for saved media
- Download all captures as a ZIP
- Delete individual captures or clear the gallery
- Light and dark interface
- Local execution

---

## 🎨 Available Filters

| # | Filter | Description |
|---|---|---|
| 1 | Original | Unmodified webcam feed |
| 2 | Grayscale | Converts the frame to grayscale |
| 3 | Sepia | Applies a warm sepia tone |
| 4 | Vintage | Combines sepia, vignette and image noise |
| 5 | Sketch | Creates a pencil-sketch effect |
| 6 | Cartoon | Combines smoothing and edge processing |
| 7 | Warm Tone | Shifts the image towards warmer tones |
| 8 | Cool Tone | Shifts the image towards cooler tones |
| 9 | Neon Glow | Creates glowing colored edges |
| 10 | Signature Duotone | Maps brightness to a custom duotone palette |

---

## 🧪 Combo Lab

The Combo Lab allows a second filter to be applied on top of the active filter.

For example:

```text
Webcam Frame
     |
     v
Neon Glow
     |
     +
   Cartoon
     |
     v
Blended Result
