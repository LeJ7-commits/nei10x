import './style.css';
import { Fluid, FADE_FRACTION, type Splat } from './fluid';
import { renderWGSL } from './shaders';

const LIFETIMES = [10, 20, 30, 45, 60, 90, 120, 180, 300, Infinity];
const DEFAULT_LIFETIME_INDEX = 4; // 60 s

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

function fail(message: string): void {
  const el = $('unsupported');
  el.querySelector('p')!.textContent = message;
  el.hidden = false;
  $('ui').hidden = true;
  $('hint').hidden = true;
}

async function main(): Promise<void> {
  if (!navigator.gpu) {
    fail('This browser does not support WebGPU. Try a recent Chrome, Edge or Safari.');
    return;
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    fail('WebGPU is available but no suitable GPU adapter was found.');
    return;
  }
  const device = await adapter.requestDevice();
  device.addEventListener('uncapturederror', (e) => console.error('WebGPU:', e.error.message));
  device.lost.then((info) => {
    if (info.reason !== 'destroyed') fail(`The GPU device was lost: ${info.message}`);
  });

  const canvas = $<HTMLCanvasElement>('canvas');
  const context = canvas.getContext('webgpu')!;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  const coarse = matchMedia('(pointer: coarse)').matches;
  const SIM_SHORT = coarse ? 224 : 400;
  const DYE_SHORT = coarse ? 640 : 1024;

  // Canvas sizing -----------------------------------------------------------

  const resizeCanvas = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(canvas.clientWidth * dpr));
    canvas.height = Math.max(1, Math.round(canvas.clientHeight * dpr));
  };
  resizeCanvas();

  const makeFluid = () => {
    const aspect = canvas.clientWidth / canvas.clientHeight || 1;
    const shortPx = Math.min(canvas.width, canvas.height);
    const dyeShort = Math.min(DYE_SHORT, shortPx);
    return new Fluid(device, aspect, Math.min(SIM_SHORT, Math.round(dyeShort / 2)), dyeShort);
  };
  let fluid = makeFluid();

  // Rendering ----------------------------------------------------------------

  const renderModule = device.createShaderModule({ label: 'render', code: renderWGSL });
  const renderPipeline = device.createRenderPipeline({
    label: 'render',
    layout: 'auto',
    vertex: { module: renderModule, entryPoint: 'vs' },
    fragment: { module: renderModule, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  const viewBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  let renderGroups = new Map<number, GPUBindGroup>();

  const draw = (encoder: GPUCommandEncoder, target: GPUTextureView) => {
    let group = renderGroups.get(fluid.inkIndex);
    if (!group) {
      group = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: fluid.paramsBuffer } },
          { binding: 1, resource: { buffer: viewBuffer } },
          { binding: 2, resource: fluid.inkTexture.createView() },
        ],
      });
      renderGroups.set(fluid.inkIndex, group);
    }
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    pass.setPipeline(renderPipeline);
    pass.setBindGroup(0, group);
    pass.draw(3);
    pass.end();
  };

  const resetFluid = () => {
    fluid.destroy();
    fluid = makeFluid();
    renderGroups = new Map();
  };

  new ResizeObserver(() => {
    resizeCanvas();
    // Keep the painting through small resizes (mobile toolbars); only a real
    // change of shape, like rotating the phone, starts a fresh sheet.
    const aspect = canvas.clientWidth / canvas.clientHeight || 1;
    if (Math.abs(aspect / fluid.aspect - 1) > 0.2) resetFluid();
  }).observe(canvas);

  // State and UI -------------------------------------------------------------

  let paused = false;
  let lifetimeIndex = DEFAULT_LIFETIME_INDEX;
  let time = 0;

  const lifetimeInput = $<HTMLInputElement>('lifetime');
  const lifetimeLabel = $('lifetime-value');
  lifetimeInput.max = String(LIFETIMES.length - 1);
  lifetimeInput.value = String(lifetimeIndex);
  const showLifetime = () => {
    const v = LIFETIMES[lifetimeIndex];
    lifetimeLabel.textContent = Number.isFinite(v) ? `${v}s` : '∞';
  };
  showLifetime();
  lifetimeInput.addEventListener('input', () => {
    lifetimeIndex = Number(lifetimeInput.value);
    showLifetime();
  });

  const pauseButton = $<HTMLButtonElement>('pause');
  const setPaused = (p: boolean) => {
    paused = p;
    pauseButton.textContent = p ? 'Resume' : 'Pause';
    pauseButton.setAttribute('aria-pressed', String(p));
    document.body.classList.toggle('paused', p);
  };
  pauseButton.addEventListener('click', () => setPaused(!paused));
  $('clear').addEventListener('click', resetFluid);
  $('snapshot').addEventListener('click', () => void snapshot());
  window.addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLInputElement) return;
    if (e.code === 'Space') { e.preventDefault(); setPaused(!paused); }
    if (e.key === 'c') resetFluid();
    if (e.key === 's') void snapshot();
  });

  $('hint').textContent = coarse
    ? 'Drag to paint · tap to drop ink · two fingers to stir'
    : 'Drag to paint · click to drop ink · right-drag or shift-drag to stir';

  // Input --------------------------------------------------------------------

  interface Pointer {
    x: number; y: number; lastX: number; lastY: number;
    downAt: number; still: number; travel: number;
    multi: boolean; stirOnly: boolean;
  }
  const pointers = new Map<number, Pointer>();
  const pending: Splat[] = [];

  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => {
    try { canvas.setPointerCapture(e.pointerId); } catch { /* pointer already gone */ }
    document.body.classList.add('touched');
    pointers.set(e.pointerId, {
      x: e.offsetX, y: e.offsetY, lastX: e.offsetX, lastY: e.offsetY,
      downAt: performance.now(), still: 0, travel: 0,
      multi: pointers.size > 0, stirOnly: e.button === 2 || e.shiftKey,
    });
    if (pointers.size > 1) for (const p of pointers.values()) p.multi = true;
  });
  canvas.addEventListener('pointermove', (e) => {
    const p = pointers.get(e.pointerId);
    if (!p) return;
    p.travel += Math.hypot(e.offsetX - p.x, e.offsetY - p.y);
    p.x = e.offsetX;
    p.y = e.offsetY;
  });
  const release = (e: PointerEvent) => {
    const p = pointers.get(e.pointerId);
    if (!p) return;
    pointers.delete(e.pointerId);
    const tap = !p.multi && !p.stirOnly && p.travel < 10 && performance.now() - p.downAt < 300;
    if (tap && !paused && e.type === 'pointerup') {
      pending.push({
        x: p.x / canvas.clientWidth, y: p.y / canvas.clientHeight,
        fx: 0, fy: 0, ink: 1.6, radius: 0.022,
      });
    }
  };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);

  const smoothstep = (a: number, b: number, x: number) => {
    const t = Math.min(Math.max((x - a) / (b - a), 0), 1);
    return t * t * (3 - 2 * t);
  };

  /** Turns this frame's pointer motion into brush splats. */
  const collectSplats = (dt: number): Splat[] => {
    const W = canvas.clientWidth;
    const H = canvas.clientHeight;
    const out = pending.splice(0);
    const stirAll = pointers.size >= 2;
    for (const p of pointers.values()) {
      const dx = p.x - p.lastX;
      const dy = p.y - p.lastY;
      const dist = Math.hypot(dx, dy);
      p.lastX = p.x;
      p.lastY = p.y;
      const stir = stirAll || p.stirOnly;

      if (dist < 0.5) {
        // A brush resting in the water slowly blooms ink around it.
        p.still += dt;
        if (!stir && !p.multi && p.still > 0.25) {
          out.push({ x: p.x / W, y: p.y / H, fx: 0, fy: 0, ink: 1.4 * dt, radius: 0.02 });
        }
        continue;
      }
      p.still = 0;

      const speed = dist / dt; // css px / s
      // A painting brush only nudges the water; stirring really moves it.
      const follow = stir ? 0.8 : 0.3;
      const maxV = fluid.simH * (stir ? 2.5 : 0.8);
      let fx = (dx / dt / W) * fluid.simW * follow;
      let fy = (dy / dt / H) * fluid.simH * follow;
      const mag = Math.hypot(fx, fy);
      if (mag > maxV) { fx *= maxV / mag; fy *= maxV / mag; }

      // Stirring moves more water than painting does.
      const forceR = stir ? 0.05 : 0.035;
      const forceSteps = Math.min(Math.ceil(dist / (forceR * 0.5 * H)), 8);
      for (let i = 1; i <= forceSteps; i++) {
        const t = i / forceSteps;
        out.push({ x: (p.lastX - dx + dx * t) / W, y: (p.lastY - dy + dy * t) / H, fx, fy, ink: 0, radius: forceR, drag: stir ? 0.85 : 0.5 });
      }
      if (stir) continue;

      // 浓淡: a slow brush lays heavy, dark ink; a fast one leaves a pale wash.
      const fast = smoothstep(150, 2200, speed);
      const tone = 1.3 - 1.0 * fast;
      const inkR = 0.016 - 0.007 * fast;
      const spacing = inkR * 0.4 * H;
      const inkSteps = Math.min(Math.ceil(dist / spacing), 48);
      const perSplat = 0.25 * tone * Math.min(1, dist / inkSteps / spacing);
      for (let i = 1; i <= inkSteps; i++) {
        const t = i / inkSteps;
        out.push({ x: (p.lastX - dx + dx * t) / W, y: (p.lastY - dy + dy * t) / H, fx: 0, fy: 0, ink: perSplat, radius: inkR });
      }
    }
    return out;
  };

  // Snapshot -----------------------------------------------------------------

  async function snapshot(): Promise<void> {
    const w = canvas.width;
    const h = canvas.height;
    const target = device.createTexture({
      size: [w, h], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const bytesPerRow = Math.ceil((w * 4) / 256) * 256;
    const readback = device.createBuffer({ size: bytesPerRow * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    writeView();
    draw(encoder, target.createView());
    encoder.copyTextureToBuffer({ texture: target }, { buffer: readback, bytesPerRow }, [w, h]);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(readback.getMappedRange());
    const pixels = new Uint8ClampedArray(w * h * 4);
    const bgra = format === 'bgra8unorm';
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const s = y * bytesPerRow + x * 4;
        const d = (y * w + x) * 4;
        pixels[d] = src[s + (bgra ? 2 : 0)];
        pixels[d + 1] = src[s + 1];
        pixels[d + 2] = src[s + (bgra ? 0 : 2)];
        pixels[d + 3] = 255;
      }
    }
    readback.unmap();
    readback.destroy();
    target.destroy();

    const out = document.createElement('canvas');
    out.width = w;
    out.height = h;
    out.getContext('2d')!.putImageData(new ImageData(pixels, w, h), 0, 0);
    const blob = await new Promise<Blob | null>((r) => out.toBlob(r, 'image/png'));
    if (!blob) return;
    const stamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, '');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `ink-${stamp}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  // Frame loop ---------------------------------------------------------------

  const viewData = new Float32Array(4);
  const writeView = () => {
    const life = LIFETIMES[lifetimeIndex];
    viewData.set([canvas.width, canvas.height, time, Number.isFinite(life) ? FADE_FRACTION : 1]);
    device.queue.writeBuffer(viewBuffer, 0, viewData);
  };

  let last = performance.now();
  const frame = (now: number) => {
    const dt = Math.min((now - last) / 1000, 1 / 30);
    last = now;
    const encoder = device.createCommandEncoder();
    if (paused) {
      // Swallow pointer motion so resuming doesn't fling the water.
      for (const p of pointers.values()) { p.lastX = p.x; p.lastY = p.y; }
      pending.length = 0;
      fluid.writeParams(0, LIFETIMES[lifetimeIndex]);
    } else {
      time += dt;
      fluid.step(encoder, collectSplats(dt), { dt, inkLifetime: LIFETIMES[lifetimeIndex] });
    }
    writeView();
    draw(encoder, context.getCurrentTexture().createView());
    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

main().catch((err) => {
  console.error(err);
  fail(`Could not start WebGPU: ${err instanceof Error ? err.message : String(err)}`);
});
