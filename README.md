# 水墨 · Ink in Water

A real-time, physics-based ink-wash (水墨画) toy. Black ink is stirred through
dark, softly lit water, and the fluid simulation runs entirely on the GPU
with WebGPU.

## Using it

| Gesture | Effect |
| --- | --- |
| Drag | Paint. A slow brush lays heavy, dark ink (浓); a fast one leaves a pale wash (淡). |
| Hold still | Ink slowly blooms around the brush. |
| Tap / click | Drop a blob of ink. |
| Two fingers / right-drag / shift-drag | Stir the water without adding ink. |

- **Ink lasts**: how long ink stays. It holds at full strength for the first ~65% of that time, then fades. Choices range from 10 s to 5 min, or ∞. The default is 60 s.
- **Pause** (Space): freezes the water and stops the fade, so you can look at or save a composition.
- **Save PNG** (S) and **Clear** (C).

The water settles within a few seconds after you stop stirring, so a pattern
stays put while you look at it.

Requires a browser with WebGPU (current Chrome, Edge, Safari on iOS/macOS 26+).
Other browsers see a short "not supported" message.

## How it works

`src/fluid.ts` runs a 2D incompressible (stable-fluids) solver each frame:
brush splats → vorticity confinement → pressure projection (Jacobi) →
velocity advection. Velocity has a short half-life so the water calms quickly.

The ink runs on a finer grid than the velocity. It is advected with
MacCormack (error-corrected, clamped), so strokes stay crisp instead of
smearing into grey. A small diffusion term feathers the edges the way ink
bleeds into wet xuan paper. Each texel also stores a mass-weighted "life"
value, which drives the hold-then-fade behaviour.

`src/shaders.ts` holds all the WGSL, including the renderer. The renderer
draws the slate water lit from below, with drifting pockets of deep green,
and applies an ink-wash tone curve that keeps pale washes readable.

## Development

```sh
npm install
npm run dev      # local dev server
npm run build    # typecheck + production build into dist/
```

Pushing to `main` deploys `dist/` to GitHub Pages through
`.github/workflows/deploy.yml`. To turn it on, set Settings → Pages → Source to "GitHub Actions".
